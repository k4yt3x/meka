# Installation

agsh is written in Rust and builds as a single binary.

## Pre-Built Binaries

Download the latest release for your platform from the [GitHub Releases](https://github.com/k4yt3x/agsh/releases/latest) page.

| Platform | Archive |
|----------|---------|
| Linux (x86_64) | `agsh-linux-amd64.tar.gz` |
| macOS (Apple Silicon) | `agsh-macos-arm64.tar.gz` |
| Windows (x86_64) | `agsh-windows-amd64.zip` |

Extract the binary and place it somewhere on your `$PATH`:

```bash
# Linux/macOS
tar -xzf agsh-*.tar.gz
cp agsh ~/.local/bin/
```

## Cargo Install

If you have [Rust](https://www.rust-lang.org/tools/install) installed, you can install agsh directly from the Git repository:

```bash
cargo install --git https://github.com/k4yt3x/agsh.git
```

This builds the latest version from source and installs it to `~/.cargo/bin/`.

## Building from Source

### Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (edition 2024, requires Rust 1.85+)
- A C compiler (for the bundled SQLite)

### Build

```bash
git clone https://github.com/k4yt3x/agsh.git
cd agsh
cargo build --release
```

The binary will be at `target/release/agsh`. Copy it somewhere on your `$PATH`:

```bash
cp target/release/agsh ~/.local/bin/
```

## Verify

```bash
agsh --version
agsh --help
```
