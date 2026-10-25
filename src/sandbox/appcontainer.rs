//! The AppContainer sandbox backend: the policy meka describes and the process security
//! environment Windows enforces from it. What does not touch the OS, the policy, its encoding and
//! the UI restrictions, is compiled and tested everywhere; the API itself is Windows-only.
//!
//! The environment is the OS's: `CreateProcessSecurityEnvironment` in `processmodel.dll` takes a
//! FlatBuffers blob (file identifier `PSEC`, schema 1.0 or 1.1) and hands back a handle that a
//! `CreateProcessW` attribute attaches to a child, which then runs as an AppContainer under the
//! blob's filesystem grants, network policy and UI restrictions, descendants included. The schema
//! is Microsoft's, vendored by mxc as `ProcessSecurityEnvironment.fbs`; the facts the policy
//! rests on were measured on build 26300 and are recorded in the shell page.
#![cfg_attr(
    not(windows),
    allow(
        dead_code,
        reason = "only the Windows spawn calls this module; what does not touch the OS is tested \
                  on every platform"
    )
)]

use std::path::PathBuf;

use crate::config::Network;

/// The `JOB_OBJECT_UILIMIT_*` bits every supported build takes, as one mask: handles to USER
/// objects outside the job, reading and writing the clipboard, system parameters, display
/// settings, global atoms, desktop switches and exiting Windows.
pub(crate) const UI_RESTRICTIONS_CLASSIC: u32 = 0xff;
/// `JOB_OBJECT_UILIMIT_IME`: the input method editor is out of reach.
pub(crate) const UI_RESTRICTION_IME: u32 = 0x100;
/// The build that introduced [`UI_RESTRICTION_IME`], Windows 11 22H2.
const IME_BUILD: u32 = 22621;
/// `JOB_OBJECT_UILIMIT_INJECTION`: `SendInput` and its kin are refused.
pub(crate) const UI_RESTRICTION_INJECTION: u32 = 0x200;
/// The build that introduced [`UI_RESTRICTION_INJECTION`], Windows 11 24H2.
const INJECTION_BUILD: u32 = 26100;

/// The UI restrictions a confined command runs under on OS build `build`: the classic bits on
/// every build, and each newer bit from the build that introduced it, because the kernel refuses
/// a bit it does not know and the spawn would fail with it. One definition for the token backend's
/// job and the AppContainer spec, so a command is kept off the desktop the same way on both.
pub(crate) fn ui_restrictions_for_build(build: u32) -> u32 {
    let mut restrictions = UI_RESTRICTIONS_CLASSIC;
    if build >= IME_BUILD {
        restrictions |= UI_RESTRICTION_IME;
    }
    if build >= INJECTION_BUILD {
        restrictions |= UI_RESTRICTION_INJECTION;
    }
    restrictions
}

/// The UI restrictions for the build this process runs on; the classic bits alone where the build
/// cannot be read.
#[cfg(windows)]
pub(crate) fn ui_restrictions_here() -> u32 {
    ui_restrictions_for_build(os_build_number().unwrap_or(0))
}

/// This OS's build number from `RtlGetVersion`, which answers for the kernel where `GetVersionExW`
/// answers for the process's manifest. `None` where ntdll does not export it.
#[cfg(windows)]
pub(crate) fn os_build_number() -> Option<u32> {
    use windows_sys::Win32::System::{
        LibraryLoader::{GetModuleHandleW, GetProcAddress},
        SystemInformation::OSVERSIONINFOW,
    };
    type RtlGetVersion = unsafe extern "system" fn(*mut OSVERSIONINFOW) -> i32;
    let ntdll: Vec<u16> = "ntdll.dll\0".encode_utf16().collect();
    // SAFETY: the name is a NUL-terminated UTF-16 string, and ntdll is mapped into every process.
    let module = unsafe { GetModuleHandleW(ntdll.as_ptr()) };
    if module.is_null() {
        return None;
    }
    // SAFETY: the export name is NUL-terminated and the module handle is live for the process.
    let function = unsafe { GetProcAddress(module, c"RtlGetVersion".as_ptr().cast()) }?;
    // SAFETY: `RtlGetVersion` has this signature on every Windows that exports it.
    let function: RtlGetVersion = unsafe { std::mem::transmute(function) };
    // SAFETY: an all-zero block with its size set is what the function fills in.
    let mut information: OSVERSIONINFOW = unsafe { std::mem::zeroed() };
    information.dwOSVersionInfoSize = std::mem::size_of::<OSVERSIONINFOW>() as u32;
    // SAFETY: the block is complete, correctly sized and outlives the call.
    if unsafe { function(&raw mut information) } != 0 {
        return None;
    }
    Some(information.dwBuildNumber)
}

/// The schema major version this backend speaks; the minor is negotiated at probe time.
pub(crate) const SCHEMA_MAJOR: u16 = 1;
/// The first minor with the ingress table, whose loopback exemption is what lets a command reach
/// this machine's own addresses: what `host` needs, so auto-pick requires it.
pub(crate) const SCHEMA_MINOR_WITH_INGRESS: u16 = 1;
/// `QueryProcessSecurityEnvironmentSupport` flag: `fs_deny` is honored. Without it meka's
/// directories cannot be hidden, so the probe refuses the backend.
pub(crate) const SUPPORT_FILESYSTEM_DENY: u64 = 0x1;

/// The AppContainer capabilities a command with network access carries: the internet and the
/// private network as Windows classifies them. Measured: traffic needs both the capability and an
/// egress table that allows it, and the capability alone or the table alone cuts everything off,
/// DNS included.
const NETWORK_CAPABILITIES: [&str; 2] = ["internetClient", "privateNetworkClientServer"];

/// What the environment enforces for one command, in meka's own terms, encoded by [`encode`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Policy {
    /// Schema major and minor.
    pub(crate) version: (u16, u16),
    /// AppContainer capability names, in the order they are joined.
    pub(crate) capabilities: Vec<String>,
    /// The `JOB_OBJECT_UILIMIT_*` mask; see [`ui_restrictions_for_build`].
    pub(crate) ui_restrictions: u32,
    /// Where the command writes: the workspace roots at `workspace`, nothing at `read`.
    pub(crate) read_write: Vec<String>,
    /// Where the command reads: every drive root where reads are not confined, else the readable
    /// paths. A root grant cascades over the whole drive, measured.
    pub(crate) read_only: Vec<String>,
    /// What the command cannot reach even under a grant above it: meka's own directories. A deny
    /// beats a read-only grant and hides the listing, measured.
    pub(crate) deny: Vec<String>,
    /// The egress and ingress tables.
    pub(crate) network: NetworkTables,
}

/// The network policy's two tables. Egress is default-allow or default-deny with no rules, since
/// meka's setting is all or nothing; ingress exists from schema 1.1 and is what carries the
/// loopback exemption.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NetworkTables {
    pub(crate) egress_allow: bool,
    pub(crate) ingress: Option<Ingress>,
}

/// The ingress table: inbound connections, and this machine's own addresses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Ingress {
    pub(crate) allow: bool,
    pub(crate) host_loopback: bool,
}

/// What a policy is built from: the confinement's paths, the machine's drives, meka's own
/// directories and the network the backend holds (`sandbox::held_network`), with the negotiated
/// schema version and the build's UI mask.
pub(crate) struct PolicyInputs<'a> {
    pub(crate) version: (u16, u16),
    pub(crate) network: Network,
    pub(crate) writable: &'a [PathBuf],
    /// `None` where reads reach everywhere.
    pub(crate) readable: Option<&'a [PathBuf]>,
    pub(crate) drive_roots: &'a [PathBuf],
    pub(crate) private: &'a [PathBuf],
    pub(crate) ui_restrictions: u32,
}

/// The policy for one command: writes beneath the workspace roots alone, reads everywhere or
/// beneath the readable paths, meka's directories denied either way, and the network as held.
///
/// A path a read-write root covers is left off the read-only list: the readable set carries the
/// workspace roots too, and the environment holds a path named on both lists as read-only
/// (measured), which would deny the workspace its own writes. A read-only grant above a
/// read-write root is fine, and is what reads everywhere with a writable workspace looks like.
pub(crate) fn policy_for(inputs: &PolicyInputs<'_>) -> Policy {
    let paths = |list: &[PathBuf]| -> Vec<String> {
        list.iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect()
    };
    let read_only: Vec<PathBuf> = inputs
        .readable
        .unwrap_or(inputs.drive_roots)
        .iter()
        .filter(|path| !inputs.writable.iter().any(|root| path.starts_with(root)))
        .cloned()
        .collect();
    let reaches_network = inputs.network != Network::None;
    let capabilities = if reaches_network {
        NETWORK_CAPABILITIES.map(str::to_string).to_vec()
    } else {
        Vec::new()
    };
    // The ingress table is what tells `host` from `remote`: with the loopback exemption a command
    // reaches this machine's own addresses, without it only other machines. Schema 1.0 has no
    // table, so there `host` is held as `remote`, which `held_network` says before this runs.
    let ingress = match inputs.network {
        Network::Host if inputs.version.1 >= SCHEMA_MINOR_WITH_INGRESS => Some(Ingress {
            allow: true,
            host_loopback: true,
        }),
        Network::Remote if inputs.version.1 >= SCHEMA_MINOR_WITH_INGRESS => Some(Ingress {
            allow: false,
            host_loopback: false,
        }),
        _ => None,
    };
    Policy {
        version: inputs.version,
        capabilities,
        ui_restrictions: inputs.ui_restrictions,
        read_write: paths(inputs.writable),
        read_only: paths(&read_only),
        deny: paths(inputs.private),
        network: NetworkTables {
            egress_allow: reaches_network,
            ingress,
        },
    }
}

/// The blob's file identifier, the four bytes after the root offset.
const FILE_IDENTIFIER: &[u8; 4] = b"PSEC";
/// `FilterAction::allow`; `deny` is 0 and the default, so it is never written.
const FILTER_ALLOW: i8 = 1;

/// A FlatBuffers writer for this one schema, written forward from the buffer's start rather than
/// back to front as a general builder does: a table's offsets to its children are unsigned and
/// point forward, so each table is written with placeholders and patched once its children have
/// a position. One schema with four tables, scalars, strings and string vectors is small enough
/// to write by hand, and the alternative, the `flatbuffers` crate with generated bindings, would
/// add a dependency for a blob the test reader below can check.
struct Writer {
    bytes: Vec<u8>,
}

impl Writer {
    fn position(&self) -> usize {
        self.bytes.len()
    }

    fn align(&mut self, alignment: usize) {
        while !self.bytes.len().is_multiple_of(alignment) {
            self.bytes.push(0);
        }
    }

    fn put_u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn put_u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn put_u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn put_u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    /// A placeholder for an offset to a child written later; its position, for [`Self::patch`].
    fn reserve(&mut self) -> usize {
        let at = self.position();
        self.put_u32(0);
        at
    }

    /// Point the placeholder at `at` to the object at `target`: a `uoffset` is relative to its
    /// own position.
    fn patch(&mut self, at: usize, target: usize) {
        let offset = (target - at) as u32;
        if let Some(slot) = self.bytes.get_mut(at..at + 4) {
            slot.copy_from_slice(&offset.to_le_bytes());
        }
    }

    /// A vtable followed by the start of its table: the table's `soffset` back to the vtable is
    /// written, and the caller writes the inline fields right after, up to `inline_size`. `slots`
    /// are each field's offset within the table in schema order, 0 for an absent field.
    fn put_table(&mut self, slots: &[u16], inline_size: u16, alignment: usize) -> usize {
        self.align(2);
        let vtable = self.position();
        self.put_u16((4 + 2 * slots.len()) as u16);
        self.put_u16(inline_size);
        for slot in slots {
            self.put_u16(*slot);
        }
        self.align(alignment);
        let table = self.position();
        self.bytes
            .extend_from_slice(&((table - vtable) as i32).to_le_bytes());
        table
    }

    fn put_string(&mut self, text: &str) -> usize {
        self.align(4);
        let at = self.position();
        self.put_u32(text.len() as u32);
        self.bytes.extend_from_slice(text.as_bytes());
        self.put_u8(0);
        at
    }

    /// A vector of strings: the length, one offset per element, then the strings.
    fn put_strings(&mut self, items: &[String]) -> usize {
        self.align(4);
        let at = self.position();
        self.put_u32(items.len() as u32);
        let slots: Vec<usize> = items.iter().map(|_| self.reserve()).collect();
        for (slot, item) in slots.into_iter().zip(items) {
            let target = self.put_string(item);
            self.patch(slot, target);
        }
        at
    }
}

/// The root table's inline layout: the `soffset`, the version struct, the 8-aligned mask, then
/// one offset slot per child. Written whole whether or not a child is present; the vtable says
/// which slots count.
const ROOT_VERSION: u16 = 4;
const ROOT_UI_RESTRICTIONS: u16 = 8;
const ROOT_CAPABILITIES: u16 = 16;
const ROOT_READ_WRITE: u16 = 20;
const ROOT_READ_ONLY: u16 = 24;
const ROOT_DENY: u16 = 28;
const ROOT_NETWORK_POLICY: u16 = 32;
const ROOT_INLINE_SIZE: u16 = 36;

/// `policy` as the blob `CreateProcessSecurityEnvironment` takes.
pub(crate) fn encode(policy: &Policy) -> Vec<u8> {
    let mut writer = Writer { bytes: Vec::new() };
    let root_slot = writer.reserve();
    writer.bytes.extend_from_slice(FILE_IDENTIFIER);

    let present = |has: bool, at: u16| if has { at } else { 0 };
    // Schema order: version, capabilities, disallow_win32k_system_calls (left at its default:
    // it kills PowerShell outright, measured), ui_restrictions, fs_read_write, fs_read_only,
    // fs_deny, network_policy, fs_enumerate (unused).
    let slots = [
        ROOT_VERSION,
        present(!policy.capabilities.is_empty(), ROOT_CAPABILITIES),
        0,
        ROOT_UI_RESTRICTIONS,
        present(!policy.read_write.is_empty(), ROOT_READ_WRITE),
        present(!policy.read_only.is_empty(), ROOT_READ_ONLY),
        present(!policy.deny.is_empty(), ROOT_DENY),
        ROOT_NETWORK_POLICY,
        0,
    ];
    let root = writer.put_table(&slots, ROOT_INLINE_SIZE, 8);
    writer.patch(root_slot, root);
    writer.put_u16(policy.version.0);
    writer.put_u16(policy.version.1);
    writer.put_u64(u64::from(policy.ui_restrictions));
    let capabilities_slot = writer.reserve();
    let read_write_slot = writer.reserve();
    let read_only_slot = writer.reserve();
    let deny_slot = writer.reserve();
    let network_slot = writer.reserve();

    if !policy.capabilities.is_empty() {
        let target = writer.put_string(&policy.capabilities.join(","));
        writer.patch(capabilities_slot, target);
    }
    for (slot, items) in [
        (read_write_slot, &policy.read_write),
        (read_only_slot, &policy.read_only),
        (deny_slot, &policy.deny),
    ] {
        if !items.is_empty() {
            let target = writer.put_strings(items);
            writer.patch(slot, target);
        }
    }
    let target = put_network_policy(&mut writer, &policy.network);
    writer.patch(network_slot, target);
    writer.bytes
}

/// The `NetworkPolicy` table: `proxy` and `allowed_appcontainer_peer` absent, `egress` always,
/// `ingress` where the tables say.
fn put_network_policy(writer: &mut Writer, tables: &NetworkTables) -> usize {
    let table = writer.put_table(&[0, 4, 0, present_slot(tables.ingress.is_some(), 8)], 12, 4);
    let egress_slot = writer.reserve();
    let ingress_slot = writer.reserve();

    // `EndpointPolicy`: `default_action`, with `allow` and `deny` rule lists absent. The byte is
    // written either way so the table holds the inline size its vtable states; the vtable alone
    // says whether it counts.
    let egress = writer.put_table(&[present_slot(tables.egress_allow, 4), 0, 0], 5, 4);
    writer.put_u8(if tables.egress_allow {
        FILTER_ALLOW as u8
    } else {
        0
    });
    writer.patch(egress_slot, egress);

    // `IngressPolicy`: `default_action` and `host_loopback`, each written only as `allow`.
    if let Some(ingress) = tables.ingress {
        let table = writer.put_table(
            &[
                present_slot(ingress.allow, 4),
                present_slot(ingress.host_loopback, 5),
            ],
            6,
            4,
        );
        writer.put_u8(if ingress.allow { FILTER_ALLOW as u8 } else { 0 });
        writer.put_u8(if ingress.host_loopback {
            FILTER_ALLOW as u8
        } else {
            0
        });
        writer.patch(ingress_slot, table);
    }
    table
}

fn present_slot(has: bool, at: u16) -> u16 {
    if has { at } else { 0 }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    /// Each newer bit from the build that introduced it, since the kernel refuses a bit it does
    /// not know: the classic eight alone on 22H1, the IME bit from 22H2, injection from 24H2.
    #[test]
    fn the_ui_restrictions_grow_with_the_build() {
        assert_eq!(ui_restrictions_for_build(0), 0xff);
        assert_eq!(ui_restrictions_for_build(22000), 0xff);
        assert_eq!(ui_restrictions_for_build(22621), 0x1ff);
        assert_eq!(ui_restrictions_for_build(26099), 0x1ff);
        assert_eq!(ui_restrictions_for_build(26100), 0x3ff);
        assert_eq!(ui_restrictions_for_build(26300), 0x3ff);
    }

    fn paths(items: &[&str]) -> Vec<PathBuf> {
        items.iter().map(PathBuf::from).collect()
    }

    fn inputs<'a>(
        version: (u16, u16),
        network: Network,
        writable: &'a [PathBuf],
        readable: Option<&'a [PathBuf]>,
        drive_roots: &'a [PathBuf],
        private: &'a [PathBuf],
    ) -> PolicyInputs<'a> {
        PolicyInputs {
            version,
            network,
            writable,
            readable,
            drive_roots,
            private,
            ui_restrictions: 0x3ff,
        }
    }

    /// `read` reads every drive and writes nowhere, `workspace` writes beneath its roots, a
    /// confined read boundary replaces the drive roots with the readable paths, and meka's
    /// directories are denied at every level.
    #[test]
    fn the_policy_follows_the_level_and_the_read_boundary() {
        let drives = paths(&["C:\\", "D:\\"]);
        let private = paths(&["C:\\Users\\me\\AppData\\Roaming\\meka"]);
        let work = paths(&["C:\\work"]);
        let readable = paths(&["C:\\Windows", "C:\\Program Files", "C:\\work"]);

        let read = policy_for(&inputs((1, 1), Network::Host, &[], None, &drives, &private));
        assert!(read.read_write.is_empty());
        assert_eq!(read.read_only, ["C:\\", "D:\\"]);
        assert_eq!(read.deny, ["C:\\Users\\me\\AppData\\Roaming\\meka"]);
        assert_eq!(read.ui_restrictions, 0x3ff);
        assert_eq!(read.version, (1, 1));

        let workspace = policy_for(&inputs(
            (1, 1),
            Network::Host,
            &work,
            None,
            &drives,
            &private,
        ));
        assert_eq!(workspace.read_write, ["C:\\work"]);
        assert_eq!(workspace.read_only, ["C:\\", "D:\\"]);

        let confined = policy_for(&inputs(
            (1, 1),
            Network::Host,
            &work,
            Some(&readable),
            &drives,
            &private,
        ));
        assert_eq!(
            confined.read_only,
            ["C:\\Windows", "C:\\Program Files"],
            "the readable paths replace the drive roots, and a writable root is not among them"
        );
        assert_eq!(confined.read_write, ["C:\\work"]);
        // Joined rather than spelled with a backslash, so the path is beneath the root on the
        // platform the test runs on: a backslash is not a separator on Unix.
        let nested = [
            PathBuf::from("C:\\work").join("sub"),
            PathBuf::from("C:\\work"),
            PathBuf::from("D:\\"),
        ];
        let nested = policy_for(&inputs(
            (1, 1),
            Network::Host,
            &work,
            Some(&nested),
            &drives,
            &private,
        ));
        assert_eq!(
            nested.read_only,
            ["D:\\"],
            "a path beneath a writable root is covered by it and stays off the read-only list"
        );
    }

    /// `none` is no capability and an egress table that denies; `host` and `remote` carry the two
    /// network capabilities and an allowing table, told apart by the 1.1 ingress table's loopback
    /// exemption; on 1.0 there is no ingress table at all.
    #[test]
    fn the_network_tables_follow_the_setting_and_the_schema() {
        let policy =
            |version, network| policy_for(&inputs(version, network, &[], None, &[], &[])).clone();
        let none = policy((1, 1), Network::None);
        assert!(none.capabilities.is_empty());
        assert_eq!(none.network, NetworkTables {
            egress_allow: false,
            ingress: None
        });
        let host = policy((1, 1), Network::Host);
        assert_eq!(host.capabilities, [
            "internetClient",
            "privateNetworkClientServer"
        ]);
        assert_eq!(host.network, NetworkTables {
            egress_allow: true,
            ingress: Some(Ingress {
                allow: true,
                host_loopback: true
            })
        });
        let remote = policy((1, 1), Network::Remote);
        assert_eq!(remote.capabilities, host.capabilities);
        assert_eq!(remote.network, NetworkTables {
            egress_allow: true,
            ingress: Some(Ingress {
                allow: false,
                host_loopback: false
            })
        });
        for network in [Network::Host, Network::Remote] {
            let old = policy((1, 0), network);
            assert_eq!(old.capabilities, host.capabilities);
            assert_eq!(
                old.network,
                NetworkTables {
                    egress_allow: true,
                    ingress: None
                },
                "{network}: schema 1.0 has no ingress table"
            );
        }
    }

    /// A reader for the blobs the tests decode: the root offset, the identifier, and tables by
    /// their vtables, the way any FlatBuffers reader walks them. Enough of the format to read
    /// every field this schema has, and nothing the encoder could share with it.
    struct Reader<'a> {
        bytes: &'a [u8],
    }

    impl<'a> Reader<'a> {
        fn u8_at(&self, at: usize) -> u8 {
            *self.bytes.get(at).expect("in bounds")
        }

        fn u16_at(&self, at: usize) -> u16 {
            let bytes = self.bytes.get(at..at + 2).expect("in bounds");
            u16::from_le_bytes([bytes[0], bytes[1]])
        }

        fn u32_at(&self, at: usize) -> u32 {
            let bytes = self.bytes.get(at..at + 4).expect("in bounds");
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
        }

        fn u64_at(&self, at: usize) -> u64 {
            let bytes: [u8; 8] = self
                .bytes
                .get(at..at + 8)
                .expect("in bounds")
                .try_into()
                .expect("eight bytes");
            u64::from_le_bytes(bytes)
        }

        fn identifier(&self) -> &'a [u8] {
            self.bytes.get(4..8).expect("an identifier")
        }

        fn root(&self) -> usize {
            self.u32_at(0) as usize
        }

        /// The position of field `index` of the table at `table`, or `None` when absent.
        fn field(&self, table: usize, index: usize) -> Option<usize> {
            let soffset = self.u32_at(table) as i32;
            let vtable = (table as i64 - i64::from(soffset)) as usize;
            let vtable_size = self.u16_at(vtable) as usize;
            let slot = 4 + 2 * index;
            if slot + 2 > vtable_size {
                return None;
            }
            let offset = self.u16_at(vtable + slot) as usize;
            (offset != 0).then_some(table + offset)
        }

        /// The target of the offset field at `at`.
        fn follow(&self, at: usize) -> usize {
            at + self.u32_at(at) as usize
        }

        fn string_at(&self, at: usize) -> String {
            let length = self.u32_at(at) as usize;
            let bytes = self.bytes.get(at + 4..at + 4 + length).expect("in bounds");
            assert_eq!(self.u8_at(at + 4 + length), 0, "a string ends in NUL");
            String::from_utf8(bytes.to_vec()).expect("UTF-8")
        }

        fn strings_at(&self, at: usize) -> Vec<String> {
            let length = self.u32_at(at) as usize;
            (0..length)
                .map(|index| {
                    let slot = at + 4 + 4 * index;
                    self.string_at(self.follow(slot))
                })
                .collect()
        }

        fn table_field(&self, table: usize, index: usize) -> Option<usize> {
            self.field(table, index).map(|at| self.follow(at))
        }

        fn string_field(&self, table: usize, index: usize) -> Option<String> {
            self.table_field(table, index).map(|at| self.string_at(at))
        }

        fn strings_field(&self, table: usize, index: usize) -> Vec<String> {
            self.table_field(table, index)
                .map(|at| self.strings_at(at))
                .unwrap_or_default()
        }

        fn byte_field(&self, table: usize, index: usize) -> u8 {
            self.field(table, index).map_or(0, |at| self.u8_at(at))
        }

        /// The whole blob as the policy it encodes.
        fn decode(&self) -> Policy {
            assert_eq!(self.identifier(), b"PSEC");
            let root = self.root();
            let version = self.field(root, 0).expect("the version is required");
            let network = self
                .table_field(root, 7)
                .expect("the network policy is written");
            let egress = self
                .table_field(network, 1)
                .expect("the egress table is written");
            let ingress = self.table_field(network, 3).map(|ingress| Ingress {
                allow: self.byte_field(ingress, 0) == 1,
                host_loopback: self.byte_field(ingress, 1) == 1,
            });
            assert!(self.field(network, 0).is_none(), "no proxy");
            assert!(self.field(network, 2).is_none(), "no peer");
            assert!(self.field(root, 2).is_none(), "win32k calls stay allowed");
            assert!(self.field(root, 8).is_none(), "no enumerate list");
            Policy {
                version: (self.u16_at(version), self.u16_at(version + 2)),
                capabilities: self
                    .string_field(root, 1)
                    .map(|joined| joined.split(',').map(str::to_string).collect())
                    .unwrap_or_default(),
                ui_restrictions: self.field(root, 3).map_or(0, |at| self.u64_at(at) as u32),
                read_write: self.strings_field(root, 4),
                read_only: self.strings_field(root, 5),
                deny: self.strings_field(root, 6),
                network: NetworkTables {
                    egress_allow: self.byte_field(egress, 0) == 1,
                    ingress,
                },
            }
        }
    }

    fn sample_policies() -> Vec<Policy> {
        let drives = paths(&["C:\\", "D:\\"]);
        let private = paths(&[
            "C:\\Users\\me\\AppData\\Roaming\\meka",
            "C:\\Users\\me\\AppData\\Local\\meka",
        ]);
        let work = paths(&["C:\\work", "D:\\more work"]);
        let readable = paths(&[
            "C:\\Windows",
            "C:\\Program Files",
            "C:\\Program Files (x86)",
        ]);
        vec![
            policy_for(&inputs((1, 1), Network::Host, &[], None, &drives, &private)),
            policy_for(&inputs(
                (1, 1),
                Network::Remote,
                &work,
                Some(&readable),
                &drives,
                &private,
            )),
            policy_for(&inputs((1, 0), Network::None, &work, None, &drives, &[])),
            policy_for(&inputs((1, 0), Network::Remote, &[], None, &[], &[])),
        ]
    }

    /// Every field survives a round trip through the blob, including an absent vector, an
    /// absent ingress table and a denying egress table, which the encoder leaves out rather than
    /// writes as defaults.
    #[test]
    fn the_blob_decodes_to_the_policy_it_encodes() {
        for policy in sample_policies() {
            let bytes = encode(&policy);
            let decoded = Reader { bytes: &bytes }.decode();
            assert_eq!(decoded, policy);
        }
    }

    /// Not a test but the recipe that checks the encoder against the official reader after a
    /// change: prints each sample blob as hex, which the spike's `spec-tool verify` parses with
    /// mxc's generated bindings over the `flatbuffers` crate. Run with `--ignored --nocapture`.
    #[test]
    #[ignore = "a recipe for the official verifier, not a check"]
    fn dump_sample_blobs_for_the_official_verifier() {
        for (index, policy) in sample_policies().iter().enumerate() {
            let hex: Vec<String> = encode(policy).iter().map(|b| format!("{b:02x}")).collect();
            println!("BLOB {index} {}", hex.join(" "));
        }
    }

    /// The scalars sit where a reader expects them: the 64-bit mask on an 8-byte boundary from the
    /// buffer's start, every table on a 4-byte one, which is what the verifier in the official
    /// reader checks before it reads anything.
    #[test]
    fn the_blob_s_scalars_are_aligned() {
        for policy in sample_policies() {
            let bytes = encode(&policy);
            let reader = Reader { bytes: &bytes };
            let root = reader.root();
            assert_eq!(root % 4, 0);
            let mask = reader.field(root, 3).expect("the mask is written");
            assert_eq!(mask % 8, 0, "the u64 at {mask} is 8-aligned");
            for index in [4, 5, 6] {
                if let Some(vector) = reader.table_field(root, index) {
                    assert_eq!(vector % 4, 0);
                }
            }
            let network = reader.table_field(root, 7).expect("network");
            assert_eq!(network % 4, 0);
        }
    }

    /// The blobs mxc's generated encoder produces for the same four policies (the spike's
    /// `spec-tool dump`), decoded by the same reader, carry the same fields as meka's: the two
    /// encoders agree on the schema's slots, enums and defaults, whatever bytes they lay them
    /// out in.
    #[test]
    fn the_official_encoder_s_blobs_decode_to_the_same_policies() {
        let goldens: [&str; 4] = [
            include_str!("appcontainer/golden-read-host-11.hex"),
            include_str!("appcontainer/golden-workspace-remote-confined-11.hex"),
            include_str!("appcontainer/golden-workspace-none-10.hex"),
            include_str!("appcontainer/golden-bare-remote-10.hex"),
        ];
        for (golden, policy) in goldens.iter().zip(sample_policies()) {
            let bytes: Vec<u8> = golden
                .split_whitespace()
                .map(|pair| u8::from_str_radix(pair, 16).expect("hex"))
                .collect();
            assert_eq!(Reader { bytes: &bytes }.decode(), policy);
        }
    }
}

/// Every drive letter in a `GetLogicalDrives` mask, as a root path: bit 0 is `A:`.
pub(crate) fn drive_roots_from(mask: u32) -> Vec<PathBuf> {
    (0..26u32)
        .filter(|bit| mask & (1 << bit) != 0)
        .map(|bit| PathBuf::from(format!("{}:\\", char::from(b'A' + bit as u8))))
        .collect()
}

/// Whether a window station needs the AppContainer grant: every one but the interactive
/// `WinSta0`, which already carries the ALL APPLICATION PACKAGES access and the Low label a
/// confined child needs to load `user32`, and which meka never touches.
pub(crate) fn needs_window_station_grant(name: &str) -> bool {
    !name.eq_ignore_ascii_case("WinSta0")
}

/// The drive roots this machine reports, read per command: what a command reads where reads are
/// not confined.
#[cfg(windows)]
pub(crate) fn drive_roots() -> Vec<PathBuf> {
    // SAFETY: no arguments, no preconditions.
    drive_roots_from(unsafe { windows_sys::Win32::Storage::FileSystem::GetLogicalDrives() })
}

/// `CreateProcessSecurityEnvironment(spec, size, flags, out handle) -> HRESULT`.
#[cfg(windows)]
type CreateFn = unsafe extern "system" fn(
    *const core::ffi::c_void,
    u32,
    u32,
    *mut windows_sys::Win32::Foundation::HANDLE,
) -> i32;
/// `CloseProcessSecurityEnvironment(handle)`.
#[cfg(windows)]
type CloseFn = unsafe extern "system" fn(windows_sys::Win32::Foundation::HANDLE);
/// `QueryProcessSecurityEnvironmentSupport(out flags) -> HRESULT`.
#[cfg(windows)]
type QueryFn = unsafe extern "system" fn(*mut u64) -> i32;
/// `IsProcessSecurityEnvironmentVersionSupported(major, out available, out minor) -> HRESULT`.
#[cfg(windows)]
type VersionFn = unsafe extern "system" fn(u32, *mut u8, *mut u32) -> i32;

/// The four exports of `processmodel.dll`, resolved once per process. The module is never freed:
/// the function pointers are held for the process's lifetime.
#[cfg(windows)]
#[derive(Clone, Copy)]
pub(crate) struct Api {
    create: CreateFn,
    close: CloseFn,
    query: QueryFn,
    version: VersionFn,
}

#[cfg(windows)]
impl Api {
    /// The exports, resolved on first use and kept; `Err` names what is missing, which is the
    /// probe's reason on a build without the API.
    pub(crate) fn load() -> Result<Api, String> {
        static API: std::sync::OnceLock<Result<Api, String>> = std::sync::OnceLock::new();
        API.get_or_init(Self::resolve).clone()
    }

    fn resolve() -> Result<Api, String> {
        use windows_sys::Win32::System::LibraryLoader::{
            GetProcAddress, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW,
        };
        let name: Vec<u16> = "processmodel.dll\0".encode_utf16().collect();
        // SAFETY: the name is NUL-terminated; the search flag confines the lookup to System32,
        // so no module of that name on the path is taken for the OS's.
        let module = unsafe {
            LoadLibraryExW(
                name.as_ptr(),
                std::ptr::null_mut(),
                LOAD_LIBRARY_SEARCH_SYSTEM32,
            )
        };
        if module.is_null() {
            return Err(format!(
                "processmodel.dll is not loadable ({})",
                std::io::Error::last_os_error()
            ));
        }
        let export = |symbol: &core::ffi::CStr| -> Result<usize, String> {
            // SAFETY: the module is loaded for the process's lifetime and the name is
            // NUL-terminated.
            unsafe { GetProcAddress(module, symbol.as_ptr().cast()) }
                .map(|function| function as usize)
                .ok_or_else(|| {
                    format!(
                        "processmodel.dll has no {}: this Windows build lacks the process \
                         security environment API",
                        symbol.to_string_lossy()
                    )
                })
        };
        let create = export(c"CreateProcessSecurityEnvironment")?;
        let close = export(c"CloseProcessSecurityEnvironment")?;
        let query = export(c"QueryProcessSecurityEnvironmentSupport")?;
        let version = export(c"IsProcessSecurityEnvironmentVersionSupported")?;
        // SAFETY: the four exports have these signatures on every build that has them, as mxc's
        // bindings state and the spike measured.
        Ok(unsafe {
            Api {
                create: std::mem::transmute::<usize, CreateFn>(create),
                close: std::mem::transmute::<usize, CloseFn>(close),
                query: std::mem::transmute::<usize, QueryFn>(query),
                version: std::mem::transmute::<usize, VersionFn>(version),
            }
        })
    }

    /// The schema minor the OS supports under [`SCHEMA_MAJOR`], or the reason it supports none.
    pub(crate) fn supported_minor(&self) -> Result<u16, String> {
        let mut available = 0u8;
        let mut minor = 0u32;
        // SAFETY: both out-parameters are live for the call.
        let result =
            unsafe { (self.version)(u32::from(SCHEMA_MAJOR), &raw mut available, &raw mut minor) };
        if result < 0 {
            return Err(format!(
                "IsProcessSecurityEnvironmentVersionSupported failed: 0x{result:08x}"
            ));
        }
        if available == 0 {
            return Err(format!(
                "schema {SCHEMA_MAJOR}.x is not supported on this build"
            ));
        }
        Ok(minor as u16)
    }

    /// `QueryProcessSecurityEnvironmentSupport`'s flags, or the reason they could not be read.
    pub(crate) fn support_flags(&self) -> Result<u64, String> {
        let mut flags = 0u64;
        // SAFETY: the out-parameter is live for the call.
        let result = unsafe { (self.query)(&raw mut flags) };
        if result < 0 {
            return Err(format!(
                "QueryProcessSecurityEnvironmentSupport failed: 0x{result:08x}"
            ));
        }
        Ok(flags)
    }
}

/// One process security environment, created from a policy and closed on drop. One per command,
/// as Bubblewrap's private `/tmp` is per command: a sub-agent's scope differs from its parent's,
/// and the OS removes the environment's redirected temp space on close.
#[cfg(windows)]
pub(crate) struct Environment {
    api: Api,
    handle: windows_sys::Win32::Foundation::HANDLE,
}

// SAFETY: the handle is an opaque kernel object usable from any thread, as every other handle the
// spawn keeps is; the child that holds it is reaped from a tokio task.
#[cfg(windows)]
unsafe impl Send for Environment {}
#[cfg(windows)]
unsafe impl Sync for Environment {}

#[cfg(windows)]
impl Environment {
    /// Create the environment `policy` describes. The roots are named in the error because the
    /// API reports an `HRESULT` alone, and a root it refuses (a drive it cannot grant) is the
    /// likeliest cause.
    pub(crate) fn create(api: Api, policy: &Policy) -> std::io::Result<Self> {
        let spec = encode(policy);
        let mut handle: windows_sys::Win32::Foundation::HANDLE = std::ptr::null_mut();
        // SAFETY: the spec buffer is live for the call with its exact length; the out-parameter
        // is live too; flags are none.
        let result =
            unsafe { (api.create)(spec.as_ptr().cast(), spec.len() as u32, 0, &raw mut handle) };
        if result < 0 || handle.is_null() {
            return Err(std::io::Error::other(format!(
                "CreateProcessSecurityEnvironment failed with 0x{result:08x} (read-only {}; \
                 read-write {}; denied {})",
                policy.read_only.join(", "),
                policy.read_write.join(", "),
                policy.deny.join(", ")
            )));
        }
        Ok(Self { api, handle })
    }

    /// The handle the spawn's security-environment attribute names.
    pub(crate) fn handle(&self) -> windows_sys::Win32::Foundation::HANDLE {
        self.handle
    }
}

#[cfg(windows)]
impl Drop for Environment {
    fn drop(&mut self) {
        // SAFETY: the handle came from a successful create and is closed exactly once, here.
        unsafe { (self.api.close)(self.handle) };
    }
}

/// Probe the backend: the exports, schema 1.x with its minor, the deny flag, and one environment
/// created and closed, so a build that has the API but refuses a spec is found at startup rather
/// than at the first command. The window station is granted on the way, since a confined child
/// on a non-interactive one cannot load `user32` without it; a failed grant is warned about once
/// and does not refuse the backend, because `cmd` and Python still run and the warning says what
/// does not.
#[cfg(windows)]
pub(crate) fn probe() -> super::BackendProbe {
    use super::{BackendProbe, SandboxCapability};
    let missing = |reason: String| BackendProbe::Missing { reason };
    let api = match Api::load() {
        Ok(api) => api,
        Err(reason) => return missing(reason),
    };
    let minor = match api.supported_minor() {
        Ok(minor) => minor,
        Err(reason) => return missing(reason),
    };
    let support = match api.support_flags() {
        Ok(support) => support,
        Err(reason) => return missing(reason),
    };
    if support & SUPPORT_FILESYSTEM_DENY == 0 {
        return missing(format!(
            "this build does not honor fs_deny (support flags 0x{support:x}), so meka's store \
             could not be hidden"
        ));
    }
    let smoke = policy_for(&PolicyInputs {
        version: (SCHEMA_MAJOR, minor),
        network: Network::None,
        writable: &[],
        readable: None,
        drive_roots: &[],
        private: &[],
        ui_restrictions: ui_restrictions_here(),
    });
    if let Err(error) = Environment::create(api, &smoke) {
        return missing(format!("an environment cannot be created: {error}"));
    }
    match ensure_window_station_access() {
        Ok(Some(name)) => tracing::debug!(
            "granted AppContainer access on window station '{name}', which is not the \
             interactive one"
        ),
        Ok(None) => {}
        Err(error) => tracing::warn!(
            "failed to grant AppContainer access on this session's window station ({error}); \
             under the AppContainer backend a command that loads user32, PowerShell included, \
             will fail to start"
        ),
    }
    tracing::debug!("AppContainer schema 1.{minor}, support flags 0x{support:x}");
    BackendProbe::Ok(SandboxCapability::AppContainer { minor })
}

/// Give ALL APPLICATION PACKAGES full access on this process's window station and desktop and
/// label both Low, as the interactive `WinSta0` is set up: a confined child on a service or SSH
/// session's window station cannot load `user32` otherwise and dies with
/// `STATUS_DLL_INIT_FAILED`, measured. The process owns its session's window station, so no
/// elevation is needed; `WinSta0` is never touched. Returns the station's name when a grant was
/// made, `None` on `WinSta0`.
#[cfg(windows)]
pub(crate) fn ensure_window_station_access() -> std::io::Result<Option<String>> {
    use windows_sys::Win32::{
        Foundation::{GENERIC_ALL, LocalFree},
        Security::{
            ACL, ACL_REVISION, AddMandatoryAce,
            Authorization::{
                ConvertStringSidToSidW, EXPLICIT_ACCESS_W, GetSecurityInfo, SE_WINDOW_OBJECT,
                SET_ACCESS, SetEntriesInAclW, SetSecurityInfo, TRUSTEE_IS_GROUP, TRUSTEE_IS_SID,
                TRUSTEE_W,
            },
            DACL_SECURITY_INFORMATION, InitializeAcl, LABEL_SECURITY_INFORMATION, PSID,
        },
        System::{
            StationsAndDesktops::{
                CloseDesktop, CloseWindowStation, GetProcessWindowStation, GetThreadDesktop,
                GetUserObjectInformationW, OpenDesktopW, OpenWindowStationW, UOI_NAME,
            },
            Threading::GetCurrentThreadId,
        },
    };
    /// `READ_CONTROL | WRITE_DAC | WRITE_OWNER`: what rewriting the DACL and the label needs.
    const REWRITE_SECURITY: u32 = 0x0002_0000 | 0x0004_0000 | 0x0008_0000;
    /// The AppContainer side: a label ACE's inheritance is none here.
    const NO_INHERITANCE: u32 = 0;
    const ALL_APPLICATION_PACKAGES: &str = "S-1-15-2-1";
    const LOW_INTEGRITY_LABEL: &str = "S-1-16-4096";

    // SAFETY: both return the calling process's own objects; no preconditions.
    let winsta = unsafe { GetProcessWindowStation() };
    let desktop = unsafe { GetThreadDesktop(GetCurrentThreadId()) };
    let user_object_name =
        |handle: windows_sys::Win32::Foundation::HANDLE| -> std::io::Result<String> {
            let mut buffer = [0u16; 256];
            let mut needed = 0u32;
            // SAFETY: the buffer and its byte length are live for the call.
            if unsafe {
                GetUserObjectInformationW(
                    handle,
                    UOI_NAME,
                    buffer.as_mut_ptr().cast(),
                    (buffer.len() * 2) as u32,
                    &raw mut needed,
                )
            } == 0
            {
                return Err(std::io::Error::last_os_error());
            }
            let length = buffer
                .iter()
                .position(|unit| *unit == 0)
                .unwrap_or(buffer.len());
            Ok(String::from_utf16_lossy(
                buffer.get(..length).unwrap_or(&[]),
            ))
        };
    let winsta_name = user_object_name(winsta.cast())?;
    if !needs_window_station_grant(&winsta_name) {
        return Ok(None);
    }
    let desktop_name = user_object_name(desktop.cast())?;
    let wide = |text: &str| -> Vec<u16> { text.encode_utf16().chain(std::iter::once(0)).collect() };
    let sid = |text: &str| -> std::io::Result<super::windows::LocalFreeGuard> {
        let mut sid: PSID = std::ptr::null_mut();
        let text = wide(text);
        // SAFETY: the string is NUL-terminated and the out-parameter is live.
        if unsafe { ConvertStringSidToSidW(text.as_ptr(), &raw mut sid) } == 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(super::windows::LocalFreeGuard(sid))
    };
    let packages = sid(ALL_APPLICATION_PACKAGES)?;
    let low = sid(LOW_INTEGRITY_LABEL)?;
    let winsta_wide = wide(&winsta_name);
    let desktop_wide = wide(&desktop_name);
    // SAFETY: the names are NUL-terminated; the handles are closed below. Each open's error is
    // read before the next call can replace it.
    let own_winsta = unsafe { OpenWindowStationW(winsta_wide.as_ptr(), 0, REWRITE_SECURITY) };
    let winsta_error = std::io::Error::last_os_error();
    let own_desktop = unsafe { OpenDesktopW(desktop_wide.as_ptr(), 0, 0, REWRITE_SECURITY) };
    let desktop_error = std::io::Error::last_os_error();
    // The first failure is the one reported; a later one on the other object adds nothing.
    let mut outcome: std::io::Result<()> = Ok(());
    let mut fail = |error: std::io::Error| {
        if outcome.is_ok() {
            outcome = Err(error);
        }
    };
    for (handle, open_error) in [
        (own_winsta.cast::<core::ffi::c_void>(), winsta_error),
        (own_desktop.cast(), desktop_error),
    ] {
        if handle.is_null() {
            fail(open_error);
            continue;
        }
        let mut old_dacl: *mut ACL = std::ptr::null_mut();
        let mut descriptor: *mut core::ffi::c_void = std::ptr::null_mut();
        // SAFETY: the out-parameters are live; the descriptor is freed below.
        let status = unsafe {
            GetSecurityInfo(
                handle,
                SE_WINDOW_OBJECT,
                DACL_SECURITY_INFORMATION,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &raw mut old_dacl,
                std::ptr::null_mut(),
                &raw mut descriptor,
            )
        };
        if status != 0 {
            fail(std::io::Error::from_raw_os_error(status as i32));
            continue;
        }
        // SAFETY: an all-zero entry with its fields set below is a complete one.
        let mut access: EXPLICIT_ACCESS_W = unsafe { std::mem::zeroed() };
        access.grfAccessPermissions = GENERIC_ALL;
        access.grfAccessMode = SET_ACCESS;
        access.grfInheritance = NO_INHERITANCE;
        access.Trustee = TRUSTEE_W {
            pMultipleTrustee: std::ptr::null_mut(),
            MultipleTrusteeOperation: 0,
            TrusteeForm: TRUSTEE_IS_SID,
            TrusteeType: TRUSTEE_IS_GROUP,
            ptstrName: packages.0.cast(),
        };
        let mut new_dacl: *mut ACL = std::ptr::null_mut();
        // SAFETY: one entry, the old DACL from the call above, a live out-parameter.
        let status = unsafe { SetEntriesInAclW(1, &raw const access, old_dacl, &raw mut new_dacl) };
        if status == 0 {
            // SAFETY: the new DACL is complete and the object handle carries WRITE_DAC.
            let status = unsafe {
                SetSecurityInfo(
                    handle,
                    SE_WINDOW_OBJECT,
                    DACL_SECURITY_INFORMATION,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    new_dacl,
                    std::ptr::null_mut(),
                )
            };
            if status != 0 {
                fail(std::io::Error::from_raw_os_error(status as i32));
            }
            // SAFETY: allocated by SetEntriesInAclW, released once.
            unsafe { LocalFree(new_dacl.cast()) };
        } else {
            fail(std::io::Error::from_raw_os_error(status as i32));
        }
        // SAFETY: allocated by GetSecurityInfo, released once.
        unsafe { LocalFree(descriptor) };

        // `u64` cells so the buffer is aligned for the ACL header and the ACE's mask.
        let mut label = [0u64; 32];
        let label_acl = label.as_mut_ptr().cast::<ACL>();
        let label_bytes = std::mem::size_of_val(&label) as u32;
        // SAFETY: the buffer is live and its size is passed; one mandatory ACE fits.
        if unsafe { InitializeAcl(label_acl, label_bytes, ACL_REVISION) } == 0
            || unsafe { AddMandatoryAce(label_acl, ACL_REVISION, 0, 0, low.0) } == 0
        {
            fail(std::io::Error::last_os_error());
            continue;
        }
        // SAFETY: the label ACL is complete and the handle carries WRITE_OWNER for the label.
        let status = unsafe {
            SetSecurityInfo(
                handle,
                SE_WINDOW_OBJECT,
                LABEL_SECURITY_INFORMATION,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                label_acl,
            )
        };
        if status != 0 {
            fail(std::io::Error::from_raw_os_error(status as i32));
        }
    }
    // SAFETY: opened above; a null handle is skipped by the loop and not closed. The SIDs are
    // freed by their guards.
    unsafe {
        if !own_winsta.is_null() {
            CloseWindowStation(own_winsta);
        }
        if !own_desktop.is_null() {
            CloseDesktop(own_desktop);
        }
    }
    drop((packages, low));
    outcome.map(|()| Some(winsta_name))
}

#[cfg(test)]
mod windows_tests {
    use super::{drive_roots_from, needs_window_station_grant};

    /// Bit 0 is `A:`, each set bit is one root, and the roots come out in letter order.
    #[test]
    fn the_drive_roots_follow_the_mask() {
        let roots: Vec<String> = drive_roots_from(0b1_1100)
            .iter()
            .map(|root| root.to_string_lossy().into_owned())
            .collect();
        assert_eq!(roots, ["C:\\", "D:\\", "E:\\"]);
        assert!(drive_roots_from(0).is_empty());
        assert_eq!(drive_roots_from(1 << 25).len(), 1, "Z: is the last letter");
    }

    /// The interactive window station is never touched, whatever its case; every other one is.
    #[test]
    fn only_the_interactive_window_station_is_left_alone() {
        assert!(!needs_window_station_grant("WinSta0"));
        assert!(!needs_window_station_grant("winsta0"));
        assert!(needs_window_station_grant("Service-0x0-3e7$"));
        assert!(needs_window_station_grant(""));
    }
}
