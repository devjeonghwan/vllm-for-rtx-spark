#!/bin/sh

set -e

if [ -z "$PERF" ]; then
    cargo build --release
    ../target/release/sample_parser $DEFAULT_ARGS "$@"
else
    PERF='perf record -F 9999 -g'
    RUSTFLAGS='-C force-frame-pointers=y' cargo build --profile perf
    $PERF ../target/perf/sample_parser $DEFAULT_ARGS "$@"
    echo "perf report -g graph,0.05,caller"
    echo "perf report -g graph,0.05,callee"
fi
