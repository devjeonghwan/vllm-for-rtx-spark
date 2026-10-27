# Contributing

The majority of the rendering and parsing is built in Rust for performance and exposed to Python
through thin [`pyo3`](https://pyo3.rs/) bindings.

```text
┌──────────────────┐      ┌───────────────────────────┐
│  Python code     │      │  Rust core (this repo)    │
│  (dataclasses,   │────► │  • chat / encoding logic  │
│   convenience)   │      │  • tokeniser (tiktoken)   │
└──────────────────┘  FFI └───────────────────────────┘
```

## Repository layout

```text
.
├── src/                  # Rust crate
│   ├── chat.rs           # High-level data-structures (Role, Message, …)
│   ├── encoding.rs       # Rendering & parsing implementation
│   ├── registry.rs       # Built-in encodings
│   ├── tests.rs          # Canonical Rust test-suite
│   └── py_module.rs      # PyO3 bindings ⇒ compiled as openai_harmony.*.so
│
├── python/openai_harmony/ # Pure-Python wrapper around the binding
│   └── __init__.py       # Dataclasses + helper API mirroring chat.rs
│
├── tests/                # Python test-suite (1-to-1 port of tests.rs)
├── Cargo.toml            # Rust package manifest
├── pyproject.toml        # Python build configuration for maturin
└── README.md             # You are here 🖖
```

## Developing locally

### Prerequisites

- Rust tool-chain (stable) – <https://rustup.rs>
- Python ≥ 3.8 + virtualenv/venv
- [`maturin`](https://github.com/PyO3/maturin) – build tool for PyO3 projects

#### 1. Clone & bootstrap

```bash
git clone https://github.com/oss-harmony/harmony.git
cd harmony
# Create & activate a virtualenv
python -m venv .venv
source .venv/bin/activate
# Install maturin and test dependencies
pip install maturin pytest mypy ruff  # tailor to your workflow
# Compile the Rust crate *and* install the Python package in editable mode
maturin develop --release
```

`maturin develop` builds _harmony_ with Cargo, produces a native extension
(`openai_harmony.<abi>.so`) and places it in your virtualenv next to the pure-
Python wrapper – similar to `pip install -e .` for pure Python projects.

#### 2. Running the test-suites

Rust:

```bash
cargo test          # runs src/tests.rs
```

Python:

```bash
pytest              # executes tests/ (mirrors the Rust suite)
```

Run both in one go to ensure parity:

```bash
pytest && cargo test
```

### 3. Type-checking & formatting (optional)

```bash
mypy harmony        # static type analysis
ruff check .        # linting
cargo fmt --all     # Rust formatter
```

[harmony-format]: https://cookbook.openai.com/articles/openai-harmony
[gpt-oss]: https://openai.com/open-models


## Cutting a new release

- bump the version in Cargo.toml

- cargo build

- cargo test

- commit the changes to Cargo.toml and Cargo.lock
  - ex: `git commit -m "Release v0.0.9"`

- create a new git tag matching the version in Cargo.toml
  - ex: `git tag v0.0.9`
  
- push the release to main
  - ex: `git push upstream` (repleace `upstream` with the name of your
    oss-harmony remote)

- push the new git tag
  - ex: `git push upstream v0.0.9` (replace `upstream` with the name
    of your oss-harmony remote)
