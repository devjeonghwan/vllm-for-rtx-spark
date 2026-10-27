#!/bin/sh

set -e
cargo run data/blog.schema.ll.json --input data/blog.sample.json
cargo run --release data/blog.schema.json --input data/blog.sample.json
cargo run --release --bin minimal data/blog.schema.json data/blog.sample.json
cargo run --release data/rfc.lark --input data/rfc.xml
