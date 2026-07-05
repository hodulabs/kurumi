hdr := "kurumi/include/kurumi.h"

default:
    @just --list

# format c + rust + justfile
format: format-c format-rs format-just

# format the c abi header
format-c:
    clang-format -i {{ hdr }}

# format rust crates
format-rs:
    cargo fmt --all

# format the justfile (just --fmt is still unstable)
format-just:
    just --fmt --unstable

# verify formatting (c + rust + justfile), no writes
format-check:
    clang-format --dry-run --Werror {{ hdr }}
    cargo fmt --all --check
    just --fmt --check --unstable

# lint c + rust + markdown
lint: lint-c lint-rs lint-md

# header compiles standalone as strict c and c++
lint-c:
    clang -x c   -std=c11   -fsyntax-only -Wall -Wextra -Wpedantic -Ikurumi/include {{ hdr }}
    clang -x c++ -std=c++17 -fsyntax-only -Wall -Wextra -Wpedantic -Ikurumi/include {{ hdr }}

# clippy over all targets, warnings denied
lint-rs:
    cargo clippy --workspace --all-targets -- -D warnings

# every relative link in the readmes resolves to a real file (external urls skipped)
lint-md:
    #!/usr/bin/env bash
    set -euo pipefail
    status=0
    for md in README.md kurumi/README.md kurumi_core/README.md kurumi_metal/README.md; do
        dir=$(dirname "$md")
        while read -r link; do
            case "$link" in http://*|https://*|mailto:*|'#'*) continue ;; esac
            path="${link%%#*}"
            [ -z "$path" ] && continue
            [ -e "$dir/$path" ] || { echo "BROKEN: $md -> $link"; status=1; }
        done < <(grep -oE '\]\([^)]+\)' "$md" | sed -E 's/^\]\(//; s/\)$//')
    done
    [ "$status" -eq 0 ] && echo "markdown links ok"
    exit $status

# workspace test suite
test:
    cargo test --workspace

# build the workspace
build:
    cargo build --workspace

# ignored benchmarks (release)
bench:
    cargo test -p kurumi_metal --release -- --ignored --nocapture

# regenerate third-party attribution (LICENSES/ + THIRD-PARTY.md) via cargo-tribute
licenses:
    cargo tribute

# fail if a dependency license is disallowed or the attribution is stale
licenses-check:
    cargo tribute --check

# CI gate: format check, lint, test
check: format-check lint test
