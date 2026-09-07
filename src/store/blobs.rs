//! Image bytes: the `blobs` table, referenced from message rows by content hash.
//!
//! An image enters meka as bytes and leaves for a provider as bytes, but it rests here once: a
//! message row holds a reference, and the same screenshot read twice costs one row. The store makes
//! the exchange at its own two doors, so nothing above it handles the reference form except the
//! readers that want it, `GET /v1/sessions/{id}/messages` and an export.

use base64::Engine;
use sha2::Digest;

use super::*;
use crate::{
    conversation::{ContentBlock, Event, Message, ToolResultContent},
    image::ImageSource,
};

/// Bytes taken out of a message on its way to a row, not yet written.
pub(super) struct NewBlob {
    pub(super) hash: String,
    pub(super) media_type: String,
    pub(super) bytes: Vec<u8>,
}

/// One stored image, as an export carries it and an import brings it back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StoredBlob {
    pub(crate) hash: String,
    pub(crate) media_type: String,
    pub(crate) bytes: Vec<u8>,
}

/// The name a blob is stored under: the SHA-256 of its bytes, in hex. Content-addressed, so two
/// reads of one file are one row and a reference can be checked against what it names.
pub(crate) fn content_hash(bytes: &[u8]) -> String {
    let digest = sha2::Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// Every message an event carries, for the walkers below. A redaction carries none: it names
/// positions in messages other events carry.
fn messages_in(event: &mut Event) -> Vec<&mut Message> {
    match event {
        Event::Append(message) => vec![message],
        Event::CompactBoundary { summary, .. } => vec![summary],
        Event::Repair { messages, .. } => messages.iter_mut().collect(),
        Event::Redact { .. } => Vec::new(),
    }
}

fn sources_in(message: &mut Message) -> Vec<&mut ImageSource> {
    let mut sources = Vec::new();
    for block in &mut message.content {
        match block {
            ContentBlock::Image { source } => sources.push(source),
            ContentBlock::ToolResult { content, .. } => {
                for item in content {
                    if let ToolResultContent::Image { source } = item {
                        sources.push(source);
                    }
                }
            }
            _ => {}
        }
    }
    sources
}

/// `event` with every inline image replaced by a reference, and the bytes to store for them.
///
/// A payload that is not base64 stays inline: the store cannot name bytes it cannot decode, and a
/// provider will say what is wrong with them where a silent drop would not.
pub(super) fn externalize_images(event: &Event) -> (Event, Vec<NewBlob>) {
    let mut event = event.clone();
    let mut blobs = Vec::new();
    for message in messages_in(&mut event) {
        for source in sources_in(message) {
            let ImageSource::Base64 { media_type, data } = source else {
                continue;
            };
            let bytes = match base64::engine::general_purpose::STANDARD.decode(data.as_bytes()) {
                Ok(bytes) => bytes,
                Err(error) => {
                    tracing::warn!(
                        "an image payload is not base64 and stays inline in the message row: {error}"
                    );
                    continue;
                }
            };
            let hash = content_hash(&bytes);
            let size = bytes.len() as u64;
            blobs.push(NewBlob {
                hash: hash.clone(),
                media_type: media_type.clone(),
                bytes,
            });
            *source = ImageSource::Blob {
                hash,
                media_type: media_type.clone(),
                size,
            };
        }
    }
    (event, blobs)
}

/// Every blob `event` references, in order of appearance, duplicates included.
pub(super) fn blob_references(event: &Event) -> Vec<String> {
    let mut event = event.clone();
    let mut hashes = Vec::new();
    for message in messages_in(&mut event) {
        for source in sources_in(message) {
            if let ImageSource::Blob { hash, .. } = source {
                hashes.push(hash.clone());
            }
        }
    }
    hashes
}

/// Write blobs that are not there yet. A hash already present is the same bytes by construction.
pub(super) fn insert_blobs(
    transaction: &rusqlite::Transaction<'_>,
    blobs: &[NewBlob],
    now: &str,
) -> rusqlite::Result<()> {
    if blobs.is_empty() {
        return Ok(());
    }
    let mut insert = transaction.prepare(
        "INSERT OR IGNORE INTO blobs (hash, media_type, bytes, size, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5)",
    )?;
    for blob in blobs {
        insert.execute(rusqlite::params![
            blob.hash,
            blob.media_type,
            blob.bytes,
            blob.bytes.len() as i64,
            now
        ])?;
    }
    Ok(())
}

/// Record that message row `message_id` references `hashes`, which is what keeps them from the
/// sweep and what scopes `GET /v1/sessions/{id}/blobs/{hash}` to a session.
pub(super) fn link_message_blobs(
    transaction: &rusqlite::Transaction<'_>,
    message_id: i64,
    hashes: &[String],
) -> rusqlite::Result<()> {
    if hashes.is_empty() {
        return Ok(());
    }
    let mut insert = transaction
        .prepare("INSERT OR IGNORE INTO message_blobs (message_id, hash) VALUES (?1, ?2)")?;
    for hash in hashes {
        insert.execute(rusqlite::params![message_id, hash])?;
    }
    Ok(())
}

/// Whether every hash in `hashes` names a stored blob, or the first that does not.
pub(super) fn missing_blob(
    connection: &rusqlite::Connection,
    hashes: &[String],
) -> rusqlite::Result<Option<String>> {
    let mut lookup = connection.prepare("SELECT 1 FROM blobs WHERE hash = ?1")?;
    for hash in hashes {
        if !lookup.exists(rusqlite::params![hash])? {
            return Ok(Some(hash.clone()));
        }
    }
    Ok(None)
}

/// Delete every blob no message row references any more. Run after a session delete, whose
/// cascade takes the references with the rows.
pub(super) fn sweep_unreferenced_blobs(
    connection: &rusqlite::Connection,
) -> rusqlite::Result<usize> {
    connection.execute(
        "DELETE FROM blobs WHERE hash NOT IN (SELECT hash FROM message_blobs)",
        [],
    )
}

impl Store {
    /// Put the bytes back into every blob reference in `events`, which is what hydrating a
    /// conversation for a turn needs: providers, the request budget and a replay all take bytes.
    ///
    /// A reference to a blob the store no longer holds is left as it is and warned about, rather
    /// than replaced with something invented; the provider then refuses the block and the turn's
    /// own degrade path takes it out.
    pub(crate) async fn inline_blobs(&self, events: &mut [Event]) -> Result<()> {
        let mut wanted: Vec<String> = Vec::new();
        for event in events.iter() {
            for hash in blob_references(event) {
                if !wanted.contains(&hash) {
                    wanted.push(hash);
                }
            }
        }
        if wanted.is_empty() {
            return Ok(());
        }
        let stored = self.load_blobs(wanted.clone()).await?;
        let by_hash: std::collections::HashMap<&str, &StoredBlob> = stored
            .iter()
            .map(|blob| (blob.hash.as_str(), blob))
            .collect();
        for event in events.iter_mut() {
            for message in messages_in(event) {
                for source in sources_in(message) {
                    let (hash, media_type) = match &*source {
                        ImageSource::Blob {
                            hash, media_type, ..
                        } => (hash.clone(), media_type.clone()),
                        ImageSource::Base64 { .. } => continue,
                    };
                    match by_hash.get(hash.as_str()) {
                        // Declared as the block was written, not as the row says: `blobs` keeps
                        // the first writer's `media_type` for the same bytes, and a later block
                        // that declared them differently would come back wearing a type it never
                        // carried, which `repair_invalid_images` then replaces with a note.
                        Some(blob) => {
                            *source = ImageSource::Base64 {
                                media_type,
                                data: base64::engine::general_purpose::STANDARD.encode(&blob.bytes),
                            };
                        }
                        None => tracing::warn!(
                            "image blob {hash} is referenced by this conversation but is not in the \
                             store; the block is sent as a reference"
                        ),
                    }
                }
            }
        }
        Ok(())
    }

    /// The first of `hashes` that names no stored blob, or `None` when every one does. What an
    /// import asks before it writes, so an archive naming bytes nobody holds is refused in the
    /// caller's words rather than inside the transaction.
    pub(super) async fn first_missing_blob(&self, hashes: Vec<String>) -> Result<Option<String>> {
        if hashes.is_empty() {
            return Ok(None);
        }
        self.connection
            .call(move |connection| missing_blob(connection, &hashes))
            .await
            .map_err(|error| MekaError::Database(format!("failed to check image blobs: {error}")))
    }

    /// The stored blobs among `hashes`. One that is missing is simply absent from the result.
    pub(crate) async fn load_blobs(&self, hashes: Vec<String>) -> Result<Vec<StoredBlob>> {
        if hashes.is_empty() {
            return Ok(Vec::new());
        }
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection
                    .prepare("SELECT hash, media_type, bytes FROM blobs WHERE hash = ?1")?;
                let mut blobs = Vec::with_capacity(hashes.len());
                for hash in &hashes {
                    let mut rows = statement.query_map(rusqlite::params![hash], |row| {
                        Ok(StoredBlob {
                            hash: row.get(0)?,
                            media_type: row.get(1)?,
                            bytes: row.get(2)?,
                        })
                    })?;
                    if let Some(blob) = rows.next() {
                        blobs.push(blob?);
                    }
                }
                Ok(blobs)
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to load image blobs: {error}")))
    }

    /// One blob, only if a message of `session_id` references it: the scope the HTTP API serves
    /// it under, so a token that may read one session cannot enumerate every image in the store.
    pub(crate) async fn load_session_blob(
        &self,
        session_id: Uuid,
        hash: &str,
    ) -> Result<Option<StoredBlob>> {
        let hash = hash.to_string();
        self.connection
            .call(move |connection| -> rusqlite::Result<_> {
                let mut statement = connection.prepare(
                    "SELECT b.hash, b.media_type, b.bytes FROM blobs b
                     JOIN message_blobs mb ON mb.hash = b.hash
                     JOIN messages m ON m.id = mb.message_id
                     WHERE m.session_id = ?1 AND b.hash = ?2
                     LIMIT 1",
                )?;
                let mut rows = statement.query_map(
                    rusqlite::params![session_id.to_string(), hash],
                    |row| {
                        Ok(StoredBlob {
                            hash: row.get(0)?,
                            media_type: row.get(1)?,
                            bytes: row.get(2)?,
                        })
                    },
                )?;
                match rows.next() {
                    Some(blob) => Ok(Some(blob?)),
                    None => Ok(None),
                }
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to load an image blob: {error}")))
    }

    /// How many blobs the store holds, for the tests that assert one image is one row.
    #[cfg(test)]
    pub(crate) async fn blob_count(&self) -> Result<i64> {
        self.connection
            .call(|connection| -> rusqlite::Result<_> {
                connection.query_row("SELECT count(*) FROM blobs", [], |row| row.get(0))
            })
            .await
            .map_err(|error| MekaError::Database(format!("failed to count blobs: {error}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(data: &str) -> ImageSource {
        ImageSource::Base64 {
            media_type: "image/png".to_string(),
            data: data.to_string(),
        }
    }

    /// The same bytes are one row however many messages carry them, every reference is recorded,
    /// and hydration puts the bytes back exactly.
    #[tokio::test]
    async fn an_image_is_stored_once_and_comes_back_whole() {
        let store = Store::for_test().await;
        let session = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("session");
        let first = Event::Append(Message::user_with_images("look", vec![image("aGVsbG8=")]));
        let second = Event::Append(Message {
            role: crate::conversation::Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "u1".to_string(),
                content: vec![ToolResultContent::Image {
                    source: image("aGVsbG8="),
                }],
                is_error: false,
            }],
        });
        store.save_event(session, &first).await.expect("save");
        store
            .save_events_atomic(session, vec![second.clone()])
            .await
            .expect("save batch");

        assert_eq!(store.blob_count().await.expect("count"), 1);
        let hash = content_hash(b"hello");
        let blob = store
            .load_session_blob(session, &hash)
            .await
            .expect("load")
            .expect("referenced by this session");
        assert_eq!(blob.bytes, b"hello");
        assert_eq!(blob.media_type, "image/png");

        let mut events = store.load_events(session).await.expect("load events");
        assert_eq!(blob_references(&events[0]), vec![hash.clone()]);
        assert_eq!(blob_references(&events[1]), vec![hash.clone()]);
        store.inline_blobs(&mut events).await.expect("inline");
        let as_json = |event: &Event| serde_json::to_string(event).expect("serialize");
        assert_eq!(as_json(&events[0]), as_json(&first));
        assert_eq!(as_json(&events[1]), as_json(&second));
    }

    /// The scope: a session that never referenced a blob cannot read it, and deleting the last
    /// session that did takes the blob with it.
    #[tokio::test]
    async fn a_blob_is_scoped_to_its_sessions_and_swept_with_the_last() {
        let store = Store::for_test().await;
        let owner = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("session");
        let stranger = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("session");
        store
            .save_event(
                owner,
                &Event::Append(Message::user_with_images("look", vec![image("aGk=")])),
            )
            .await
            .expect("save");
        let hash = content_hash(b"hi");
        assert!(
            store
                .load_session_blob(stranger, &hash)
                .await
                .expect("load")
                .is_none(),
            "another session must not reach it"
        );
        assert!(store.delete_session(owner).await.expect("delete"));
        assert_eq!(
            store.blob_count().await.expect("count"),
            0,
            "nothing references it any more"
        );
    }

    /// A block comes back declared as it was written. Two blocks can name the same bytes under
    /// different types and the row keeps only the first writer's, so hydrating from the row would
    /// hand the second block a type it never declared.
    #[tokio::test]
    async fn a_block_keeps_its_own_media_type_when_its_bytes_are_shared() {
        let store = Store::for_test().await;
        let session = store
            .create_session(None, "test-profile".to_string())
            .await
            .expect("session");
        let as_png = Event::Append(Message::user_with_images("first", vec![image("aGVsbG8=")]));
        let as_jpeg = Event::Append(Message::user_with_images("second", vec![
            ImageSource::Base64 {
                media_type: "image/jpeg".to_string(),
                data: "aGVsbG8=".to_string(),
            },
        ]));
        store.save_event(session, &as_png).await.expect("save");
        store.save_event(session, &as_jpeg).await.expect("save");
        assert_eq!(
            store.blob_count().await.expect("count"),
            1,
            "one row for the bytes"
        );

        let mut events = store.load_events(session).await.expect("load");
        store.inline_blobs(&mut events).await.expect("inline");
        let declared = |event: &Event| match event {
            Event::Append(message) => match &message.content[1] {
                ContentBlock::Image { source } => source.media_type().to_string(),
                other => panic!("expected the image block, got {other:?}"),
            },
            other => panic!("expected an append, got {other:?}"),
        };
        assert_eq!(declared(&events[0]), "image/png");
        assert_eq!(declared(&events[1]), "image/jpeg");
    }

    /// A payload that is not base64 is left inline rather than stored under a hash of nothing.
    #[test]
    fn a_payload_that_is_not_base64_stays_inline() {
        let event = Event::Append(Message::user_with_images("look", vec![image(
            "!!not base64!!",
        )]));
        let (externalized, blobs) = externalize_images(&event);
        assert!(blobs.is_empty());
        assert_eq!(
            serde_json::to_string(&externalized).expect("serialize"),
            serde_json::to_string(&event).expect("serialize")
        );
    }
}
