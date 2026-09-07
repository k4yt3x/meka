#!/usr/bin/env python3
"""Rewrite a 0.45 `config.toml` into the shape meka 0.46 reads.

0.46 splits each `[providers.<name>]` profile into an `[accounts.<name>]` table (backend, endpoint
and OAuth settings) and a `[profiles.<name>]` table (account, model and every model-tied setting),
renames `default_provider` to `default_profile`, replaces the `ask` permission level with the
`approvals` switch, and spells every duration as a string (`request_timeout_seconds = 30` becomes
`request_timeout = "30s"`, `retention_days = 30` becomes `retention = "30d"`). meka carries its
database forward on its own; this file is the one thing it cannot migrate, because `config.toml`
has no version and meka refuses a key it does not know rather than guessing what it meant.

Dry run by default: prints what would change as a unified diff and writes nothing. `--apply`
rewrites the file in place, atomically, keeping every comment and the order of everything it does
not touch.

    python3 migrate-0.45-to-0.46.py            # report
    python3 migrate-0.45-to-0.46.py --apply    # rewrite

Requires Python 3.11+ and `tomlkit` (`pip install tomlkit`), which preserves comments; the standard
library's `tomllib` reads TOML but cannot write it back.
"""

from __future__ import annotations

import argparse
import difflib
import os
import sys
import tempfile
from pathlib import Path

try:
    import tomlkit
except ImportError:  # pragma: no cover - the whole point is to say so
    sys.exit("this script needs `tomlkit`; install it with `pip install tomlkit`")

ACCOUNT_KEYS = ("backend", "base_url", "oauth_token_url", "client_id", "device_id")
# Per table: old key -> (new key, unit suffix for an integer that becomes a duration string, or
# None for a plain rename).
KEY_RENAMES = {
    "web": {
        "request_timeout_seconds": ("request_timeout", "s"),
        "connect_timeout_seconds": ("connect_timeout", "s"),
        "read_timeout_seconds": ("read_timeout", "s"),
    },
    "mcp": {
        "strict": ("default_required", None),
        "grace_seconds": ("grace", "s"),
        "connect_timeout_seconds": ("connect_timeout", "s"),
    },
    "session": {"retention_days": ("retention", "d")},
    "thinking": {"budget_tokens": ("budget", None)},
}
PROFILE_KEYS = (
    "account",
    "model",
    "context_window",
    "max_output_tokens",
    "effort",
    "vision",
    "thinking",
    "thinking_budget",
    "max_request_bytes",
    "redact_thinking",
)


def default_config_path() -> Path:
    """Where meka reads its config: `MEKA_CONFIG_DIR`, else the platform config directory."""
    override = os.environ.get("MEKA_CONFIG_DIR")
    if override and os.path.isabs(override):
        return Path(override) / "config.toml"
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "meka" / "config.toml"


class Report:
    """What the rewrite did, line by line, for the dry run to print and `--apply` to confirm."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.warnings: list[str] = []

    def note(self, line: str) -> None:
        self.lines.append(line)

    def warn(self, line: str) -> None:
        self.warnings.append(line)


class Refused(Exception):
    """The file cannot be converted without guessing, and the reason is the message."""


def with_trivia_of(item, original):
    """`item`, carrying the spacing and trailing comment that sat on `original`."""
    item.trivia.indent = original.trivia.indent
    item.trivia.comment_ws = original.trivia.comment_ws
    item.trivia.comment = original.trivia.comment
    item.trivia.trail = original.trivia.trail
    return item


def rename_keys(table, section: str, report: Report):
    """`table` with this release's key names, in the same order with the same comments, or `None`
    when it holds none of the old ones."""
    renames = KEY_RENAMES[section]
    if not any(old in table for old in renames):
        return None
    rebuilt = tomlkit.inline_table() if isinstance(table, tomlkit.items.InlineTable) else tomlkit.table()
    with_trivia_of(rebuilt, table)
    for key, item in table.value.body:
        if key is None:
            rebuilt.add(item)
            continue
        old = key.key
        if old not in renames:
            rebuilt.add(key, item)
            continue
        new, unit = renames[old]
        if unit is None:
            rebuilt.add(new, item)
            report.note(f"[{section}].{old} -> {new}")
        elif isinstance(item, tomlkit.items.Integer):
            value = int(item)
            if section == "web" and value == 0:
                # `0` meant "use the default" on the integer keys, and 0.46 refuses `"0s"`, so the
                # faithful conversion is no key at all.
                report.note(f"[web].{old} = 0 meant the default; the key is removed")
                continue
            rebuilt.add(new, with_trivia_of(tomlkit.string(f"{value}{unit}"), item))
            report.note(f'[{section}].{old} = {value} -> {new} = "{value}{unit}"')
        else:
            # Not a number, so not a value this script can put a unit on; left under its old name,
            # where meka refuses it by name rather than reading it as something else.
            rebuilt.add(key, item)
            report.warn(
                f"[{section}].{old} is not a whole number; left as is, and meka will refuse it"
            )
    return rebuilt


def convert(document: tomlkit.TOMLDocument, report: Report) -> tuple[tomlkit.TOMLDocument, bool]:
    """The rewritten document, and whether anything changed. `document` itself is not reused."""
    changed = False

    providers = document.get("providers")
    if providers is not None:
        if "accounts" in document or "profiles" in document:
            raise Refused(
                "this file has [providers] beside [accounts] or [profiles]; finish the split by "
                "hand first"
            )
        accounts = tomlkit.table(is_super_table=True)
        profiles = tomlkit.table(is_super_table=True)
        for name, provider in providers.items():
            account = tomlkit.table()
            profile = tomlkit.table()
            profile.add("account", name)
            for key, value in provider.items():
                if key == "type":
                    account.add("backend", value)
                elif key in ACCOUNT_KEYS:
                    account.add(key, value)
                elif key in PROFILE_KEYS:
                    profile.add(key, value)
                else:
                    # Left where meka will name it rather than dropped: an unknown key is refused
                    # by name at load, which is a better outcome than a setting vanishing.
                    profile.add(key, value)
                    report.warn(
                        f"[providers.{name}].{key} is not a key meka 0.46 knows; carried into "
                        f"[profiles.{name}], where meka will refuse it"
                    )
            if "backend" not in account:
                report.warn(f"[providers.{name}] states no `type`; the account has no backend")
            accounts.add(name, account)
            profiles.add(name, profile)
            report.note(f"[providers.{name}] -> [accounts.{name}] + [profiles.{name}]")
        changed = True
    else:
        report.note("no [providers] table; nothing to split")

    if "default_provider" in document:
        report.note(
            f'default_provider = "{document["default_provider"]}" -> default_profile = '
            f'"{document["default_provider"]}"'
        )
        changed = True

    # Rebuilt in order rather than edited in place: tomlkit appends a new table at the end, and
    # the accounts and profiles belong where the profiles they came from were, with the comments
    # that sat above them. Everything else is carried over untouched.
    rebuilt = tomlkit.document()
    for key, item in document.body:
        if key is None:
            rebuilt.add(item)
        elif key.key == "providers":
            rebuilt.add("accounts", accounts)
            rebuilt.add("profiles", profiles)
        elif key.key == "default_provider":
            rebuilt.add("default_profile", item)
        elif key.key in KEY_RENAMES and (renamed := rename_keys(item, key.key, report)) is not None:
            rebuilt.add(key, renamed)
            changed = True
        else:
            rebuilt.add(key, item)

    permissions = rebuilt.get("permissions")
    if permissions is not None:
        enabled = permissions.get("enabled")
        if enabled is not None and "ask" in list(enabled):
            # `none` in its place, not a shorter list: `ask` asked about everything and ran nothing
            # unattended, which is `none` with approvals on, and a default outside the enabled list
            # is what meka would otherwise have to fall back from.
            kept: list[str] = []
            for level in enabled:
                level = "none" if level == "ask" else level
                if level not in kept:
                    kept.append(level)
            permissions["enabled"] = kept
            report.note("[permissions].enabled: `ask` -> `none`")
            changed = True
        if permissions.get("default") == "ask":
            permissions["default"] = "none"
            permissions["approvals"] = True
            report.note(
                '[permissions].default = "ask" -> default = "none" with approvals = true'
            )
            changed = True

    return rebuilt, changed


def write_atomically(path: Path, text: str) -> None:
    """Replace `path` with `text` through a rename, keeping its mode."""
    mode = path.stat().st_mode & 0o777
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        os.unlink(temporary)
        raise


SELF_TEST_INPUT = """# my config
default_provider = "work"

[providers.work]
type = "claude-subscription"
model = "claude-opus-5"   # pinned
context_window = 200000
redact_thinking = false
device_id = "abc"

[providers.local]
type     = "openai-chat-completions"
base_url = "http://localhost:11434/v1"
model    = "llama3"

[permissions]
default = "ask"
enabled = ["read", "ask", "unrestricted"]

[display]
stream = true

[web]
request_timeout_seconds = 60   # slow proxy
connect_timeout_seconds = 0

[session]
retention_days = 30

[thinking]
budget_tokens = 20000

[mcp]
strict = true
grace_seconds = 5
connect_timeout_seconds = 45

[[mcp.servers]]
name = "s"
transport = "stdio"
command = "x"
"""

SELF_TEST_EXPECTED = {
    "default_profile": "work",
    "accounts": {
        "work": {"backend": "claude-subscription", "device_id": "abc"},
        "local": {
            "backend": "openai-chat-completions",
            "base_url": "http://localhost:11434/v1",
        },
    },
    "profiles": {
        "work": {
            "account": "work",
            "model": "claude-opus-5",
            "context_window": 200000,
            "redact_thinking": False,
        },
        "local": {"account": "local", "model": "llama3"},
    },
    "permissions": {
        "default": "none",
        "enabled": ["read", "none", "unrestricted"],
        "approvals": True,
    },
    "display": {"stream": True},
    "web": {"request_timeout": "60s"},
    "session": {"retention": "30d"},
    "thinking": {"budget": 20000},
    "mcp": {
        "default_required": True,
        "grace": "5s",
        "connect_timeout": "45s",
        "servers": [{"name": "s", "transport": "stdio", "command": "x"}],
    },
}


def self_test() -> int:
    document, changed = convert(tomlkit.parse(SELF_TEST_INPUT), Report())
    assert changed, "the fixture has work to do"
    rendered = tomlkit.dumps(document)
    import tomllib

    parsed = tomllib.loads(rendered)
    assert parsed == SELF_TEST_EXPECTED, f"unexpected result:\n{rendered}"
    assert "# pinned" in rendered and "# my config" in rendered, "comments must survive"
    assert "# slow proxy" in rendered, "a comment on a converted key must survive"
    assert rendered.index("[accounts.work]") < rendered.index("[permissions]"), (
        f"the new tables must sit where [providers] was:\n{rendered}"
    )
    # A second run finds nothing to do.
    _, again = convert(tomlkit.parse(rendered), Report())
    assert not again, "a migrated file must be left alone"
    # A file with both shapes is refused rather than merged by guesswork.
    try:
        convert(tomlkit.parse(rendered + '\n[providers.late]\ntype = "x"\n'), Report())
    except Refused:
        pass
    else:
        raise AssertionError("both shapes at once must be refused")
    print("self-test passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None, help="config.toml to rewrite (default: meka's own)")
    parser.add_argument("--apply", action="store_true", help="write the result; the default is a dry run")
    parser.add_argument("--self-test", action="store_true", help="check the script against its own fixture and exit")
    arguments = parser.parse_args()

    if arguments.self_test:
        return self_test()

    path = arguments.config or default_config_path()
    if not path.is_file():
        print(f"no config file at {path}", file=sys.stderr)
        return 1
    before = path.read_text(encoding="utf-8")
    report = Report()
    try:
        document, changed = convert(tomlkit.parse(before), report)
    except Refused as refusal:
        print(f"{path}: {refusal}", file=sys.stderr)
        return 1
    after = tomlkit.dumps(document)

    for line in report.lines:
        print(line)
    for line in report.warnings:
        print(f"warning: {line}", file=sys.stderr)
    if not changed:
        print(f"{path}: nothing to change")
        return 0

    if arguments.apply:
        write_atomically(path, after)
        print(f"{path}: rewritten")
        return 0

    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=str(path),
        tofile=f"{path} (0.46)",
    )
    sys.stdout.writelines(diff)
    print(f"\ndry run; rerun with `--apply` to write {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
