# JSON Schema Test Suite Runner

Runs the official [JSON Schema Test Suite](https://github.com/json-schema-org/JSON-Schema-Test-Suite)
against llguidance's JSON Schema compiler with **ratchet-based regression detection**.

## How it works

Each test case is compiled into a grammar and run against test instances. Results are
categorized by distance from correctness (best to worst):

| Category | Meaning |
|---|---|
| `pass` | Instance result matches expectation |
| `false_negative` | Instance was rejected but should have been accepted |
| `compile_error_all_invalid` | Schema failed to compile, all instances are invalid anyway |
| `skip_compile` | Schema uses an unimplemented feature |
| `compile_error_valid` | Schema failed to compile, but has valid instances we can't handle |
| `false_positive` | Instance was accepted but should have been rejected |

A **baseline file** (`expected_json_schema_test_suite.json`) records the expected category for
every test case. The runner compares current results against this baseline:

- **Regressions** (category got worse) → exit with error
- **Improvements** (category got better) → printed as info, still passes
- **Match** → silent pass

This means the test suite **ratchets forward**: improvements are allowed, regressions are not.

## Usage

```bash
# Run against baseline (clones test suite automatically if not provided)
cargo run -p json_schema_test_suite --release -- \
  --expected expected_json_schema_test_suite.json

# Run specific draft(s)
cargo run -p json_schema_test_suite --release -- \
  --draft draft2020-12 --draft draft7 \
  --expected expected_json_schema_test_suite.json

# Point to a local test suite checkout
cargo run -p json_schema_test_suite --release -- \
  --expected expected_json_schema_test_suite.json \
  /path/to/JSON-Schema-Test-Suite

# Update the baseline after fixing bugs
cargo run -p json_schema_test_suite --release -- \
  --expected expected_json_schema_test_suite.json --update
```

## CLI Options

```
Arguments:
  [SUITE_DIR]  Path to JSON-Schema-Test-Suite checkout (auto-cloned if omitted)

Options:
      --expected <FILE>    Baseline file for ratchet comparison
      --draft <DRAFT>      Draft(s) to run (repeatable). Without this: runs all
                           drafts in the baseline, or draft2020-12 if no baseline
      --update             Overwrite the baseline with current results
```

## CI

This runs as a GitHub Actions workflow (`.github/workflows/json-schema-tests.yml`) on every
push, pinned to a specific test suite commit for reproducibility.
