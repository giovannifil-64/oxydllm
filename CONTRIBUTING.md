# Contributing to oxydLLM

Thank you for your interest in contributing! Below is everything you need to get started.

## Prerequisites

- Rust toolchain
- macOS with Apple Silicon for Metal-accelerated development, or Linux with an NVIDIA GPU for CUDA (CPU path works on both macOS and Linux)
- A supported model checkpoint for end-to-end testing (optional but recommended)

## Setting up the project

```bash
git clone https://github.com/giovannifil-64/oxydllm.git
cd oxydllm
git config core.hooksPath .githooks
cargo build
```

For Metal support (macOS default):

```bash
cargo build --features metal
```

For CUDA (Linux + NVIDIA GPU):

```bash
cargo build --features cuda
```

For CPU-only:

```bash
cargo build --no-default-features
```

## Running tests

Unit and integration tests (no GPU required):

```bash
cargo test
```

Architecture regression tests download small model checkpoints on first run and are gated behind the `arch_regression` feature:

```bash
cargo test --test arch_regression
```

## Making a pull request

1. Fork the repo and create a branch from `main`.
2. Write or update tests for any logic you change.
3. Run `cargo clippy --all-targets` and fix all warnings.
4. Run `cargo fmt --check`; format your code with `cargo fmt`.
5. Open a PR against `main` and fill in the pull request template.
> [!IMPORTANT]
> This repository uses Git hooks (located in the `.githooks/` directory). Ensure you have run `git config core.hooksPath .githooks` during setup. If the `pre-commit` or `pre-push` scripts fail, your action will be blocked.


Please keep PRs focused — one logical change per PR makes review much faster.

## Adding a new model architecture

The lowest-friction path for a new architecture is a new entry in `src/models/arch_defaults.rs` and, if the HuggingFace `config.json` uses non-standard field names, a small addition to `src/models/parsers/hf_parser.rs`. The inline comments in those files describe the cost estimate by scenario.

## Code style

- No clippy warnings on `--all-targets`.
- Avoid using `#[allow(dead_code)]` as much as possible; instead, remove the dead code or wire up the unused fields.
- No `println!` in library code; use `tracing::{info,warn,error}`.
- Default to writing no comments. Only add one when the *why* is non-obvious.
- Prefer small, focused commits with a conventional-commits subject line (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`).

## Reporting bugs and requesting features

Please use the issue templates provided in `.github/ISSUE_TEMPLATE/`.

## License

By contributing you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).
