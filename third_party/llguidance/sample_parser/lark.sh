#!/bin/sh

mkdir -p tmp

case "$1" in
    *.gbnf )
        echo "Generating Lark grammar from GBNF $1 to tmp/gbnf.lark"
        ../scripts/gbnf_to_lark.py "$1" > tmp/gbnf.lark
        LARK=tmp/gbnf.lark
        ;;
    *.lark )
        LARK="$1"
        ;;
    test )
        set -xe
        ../scripts/gbnf_to_lark.py data/from-llama.cpp/*.gbnf
        cargo build
        for f in data/from-llama.cpp/*.lark ; do
            ../target/debug/sample_parser "$f"
        done
        exit 0
        ;;
    *)
        echo "Usage: $0 <lark_file> [args...]"
        exit 1
        ;;
esac

cargo run -- $LARK
