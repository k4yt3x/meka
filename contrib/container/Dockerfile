# syntax=docker/dockerfile:1.7

FROM archlinux:latest AS builder

WORKDIR /src
COPY Cargo.toml Cargo.lock build.rs ./
COPY assets ./assets
COPY src ./src

RUN pacman -Syu --noconfirm \
        base-devel \
        rust \
    && rm -rf /var/cache/pacman/pkg \
    && cargo build --release --locked

FROM archlinux:latest
COPY --from=builder /src/target/release/agsh /usr/local/bin/agsh
WORKDIR /root
ENTRYPOINT ["/usr/local/bin/agsh"]
