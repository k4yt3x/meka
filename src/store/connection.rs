//! The store's connection to its database: one SQLite connection on a thread of its own, which runs
//! the closures it is handed one at a time, in the order they were handed over. A caller awaits
//! SQLite without blocking the runtime, and no two callers interleave on the connection.

use std::{path::Path, sync::mpsc};

/// One closure the connection's thread runs against the connection.
type Job = Box<dyn FnOnce(&mut rusqlite::Connection) + Send + 'static>;

/// A handle to the connection's thread. Clones share the one connection; the thread, and the
/// connection with it, ends once the last handle is dropped and every job handed over has run.
#[derive(Clone, Debug)]
pub(crate) struct Connection {
    jobs: mpsc::Sender<Job>,
}

/// Why a call came back without the closure's answer.
#[derive(Debug)]
pub(crate) enum Error<E = rusqlite::Error> {
    /// The connection's thread is gone, which a closure that panicked on it leaves behind.
    ConnectionClosed,
    /// The closure ran and answered with this error.
    Error(E),
}

impl<E: std::fmt::Display> std::fmt::Display for Error<E> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConnectionClosed => formatter.write_str("the database connection is closed"),
            Self::Error(error) => error.fmt(formatter),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for Error<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ConnectionClosed => None,
            Self::Error(error) => Some(error),
        }
    }
}

/// Why a connection could not be opened.
#[derive(Debug)]
pub(crate) enum OpenError {
    /// The thread the connection would live on did not start, or stopped before it opened one.
    Thread(String),
    /// SQLite failed to open the database.
    Sqlite(rusqlite::Error),
}

impl std::fmt::Display for OpenError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Thread(reason) => write!(formatter, "the database thread {reason}"),
            Self::Sqlite(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for OpenError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Thread(_) => None,
            Self::Sqlite(error) => Some(error),
        }
    }
}

impl Connection {
    /// Open the database at `path` on a thread of its own.
    pub(crate) async fn open(path: impl AsRef<Path>) -> Result<Self, OpenError> {
        let path = path.as_ref().to_owned();
        Self::start(move || rusqlite::Connection::open(path)).await
    }

    /// Open a database in memory on a thread of its own.
    #[cfg(test)]
    pub(crate) async fn open_in_memory() -> Result<Self, OpenError> {
        Self::start(rusqlite::Connection::open_in_memory).await
    }

    /// Start the thread, open the connection on it with `open`, and hand back the handle once it
    /// is open.
    async fn start(
        open: impl FnOnce() -> rusqlite::Result<rusqlite::Connection> + Send + 'static,
    ) -> Result<Self, OpenError> {
        let (jobs, received) = mpsc::channel::<Job>();
        let (opened, opening) = tokio::sync::oneshot::channel::<rusqlite::Result<()>>();
        std::thread::Builder::new()
            .name("meka-store".to_string())
            .spawn(move || {
                let mut connection = match open() {
                    Ok(connection) => connection,
                    Err(error) => {
                        if opened.send(Err(error)).is_err() {
                            tracing::debug!("the store's opener stopped waiting for its refusal");
                        }
                        return;
                    }
                };
                if opened.send(Ok(())).is_err() {
                    tracing::debug!("the store's opener stopped waiting for its connection");
                    return;
                }
                while let Ok(job) = received.recv() {
                    job(&mut connection);
                }
            })
            .map_err(|error| OpenError::Thread(format!("did not start: {error}")))?;
        match opening.await {
            Ok(Ok(())) => Ok(Self { jobs }),
            Ok(Err(error)) => Err(OpenError::Sqlite(error)),
            Err(_) => Err(OpenError::Thread("stopped before it opened".to_string())),
        }
    }

    /// Run `function` against the connection, after every closure handed over before it, and
    /// answer with what it answered.
    pub(crate) async fn call<F, R, E>(&self, function: F) -> Result<R, Error<E>>
    where
        F: FnOnce(&mut rusqlite::Connection) -> Result<R, E> + Send + 'static,
        R: Send + 'static,
        E: Send + 'static,
    {
        let (answer, answered) = tokio::sync::oneshot::channel();
        self.jobs
            .send(Box::new(move |connection| {
                // A caller that stopped waiting has its closure run all the same, since a write
                // half done is worse than one done unread.
                if answer.send(function(connection)).is_err() {
                    tracing::trace!("a store call's caller stopped waiting for its answer");
                }
            }))
            .map_err(|_| Error::ConnectionClosed)?;
        answered
            .await
            .map_err(|_| Error::ConnectionClosed)?
            .map_err(Error::Error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Calls queued behind a busy connection run in the order they were handed over, not in
    /// whatever order the thread finds them.
    #[tokio::test]
    async fn queued_calls_run_in_the_order_they_were_handed_over() {
        use std::future::Future;

        let connection = Connection::open_in_memory().await.expect("open");
        connection
            .call(|connection| connection.execute_batch("CREATE TABLE t (n INTEGER)"))
            .await
            .expect("create");
        // Holds the thread until every call below is queued behind it. Each call hands its
        // closure over on its first poll, so polling them once, in order, queues them in order.
        let (release, gate) = mpsc::channel::<()>();
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        let mut held = Box::pin(connection.call(move |_| gate.recv()));
        assert!(held.as_mut().poll(&mut context).is_pending());
        let mut queued: Vec<_> = (0..10)
            .map(|n| {
                Box::pin(connection.call(move |connection| {
                    connection.execute("INSERT INTO t (n) VALUES (?1)", [n])
                }))
            })
            .collect();
        for call in &mut queued {
            assert!(call.as_mut().poll(&mut context).is_pending());
        }
        release.send(()).expect("release");
        held.await.expect("held");
        for call in queued {
            call.await.expect("insert");
        }
        let order: Vec<i64> = connection
            .call(|connection| {
                let mut statement = connection.prepare("SELECT n FROM t ORDER BY rowid")?;
                let order = statement
                    .query_map([], |row| row.get(0))?
                    .collect::<rusqlite::Result<_>>()?;
                Ok::<_, rusqlite::Error>(order)
            })
            .await
            .expect("read");
        assert_eq!(order, (0..10).collect::<Vec<i64>>());
    }

    /// A closure's own error comes back as the closure's, not as a closed connection.
    #[tokio::test]
    async fn a_closure_s_error_comes_back_as_its_own() {
        let connection = Connection::open_in_memory().await.expect("open");
        let failed = connection
            .call(|connection| connection.execute_batch("SELECT * FROM missing"))
            .await;
        assert!(matches!(failed, Err(Error::Error(_))), "{failed:?}");
    }

    /// A closure that panics takes the connection's thread with it; every later call is told the
    /// connection is closed rather than waiting forever.
    #[tokio::test]
    async fn a_panicking_closure_closes_the_connection() {
        let connection = Connection::open_in_memory().await.expect("open");
        let panicked = connection
            .call(|_| -> rusqlite::Result<()> { panic!("a closure that panics") })
            .await;
        assert!(
            matches!(panicked, Err(Error::ConnectionClosed)),
            "{panicked:?}"
        );
        let after = connection
            .call(|_| -> rusqlite::Result<()> { Ok(()) })
            .await;
        assert!(matches!(after, Err(Error::ConnectionClosed)), "{after:?}");
    }

    /// A database SQLite cannot open fails with SQLite's own error.
    #[tokio::test]
    async fn a_database_that_cannot_be_opened_fails_with_sqlite_s_error() {
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("missing").join("meka.db");
        let failed = Connection::open(&path).await;
        assert!(matches!(failed, Err(OpenError::Sqlite(_))), "{failed:?}");
    }
}
