#!/bin/sh

CLEAN=0
if [ "$1" = "-c" ]; then
    CLEAN=1
    shift
fi

TRG=`rustup show | head -1 | sed -e 's/.*: //'`
#CRATE=`grep "^name =" Cargo.toml  | head -1 | sed -e 's/.*= "//; s/"//'`
CRATE=$1
shift
TRG_DIR=`cargo metadata --format-version 1 | jq -r '.target_directory'`
if [ $CLEAN -eq 1 ]; then
    rm -rf $TRG_DIR/$TRG/release/deps/$CRATE{,-*}.s
fi
RUSTFLAGS="--emit asm $RUSTFLAGS" cargo $RUSTCHANNEL build --release --target $TRG
F=`ls $TRG_DIR/$TRG/release/deps/$CRATE{,-*}.s 2>/dev/null`
# if $F has more than one file
if [ `echo $F | wc -w` -gt 1 ]; then
    echo "More than one file found: $F; removing; try again"
    rm -f $F
    exit 1
fi

mkdir -p tmp
cp $F tmp/full.s
HERE=$(dirname $0)
node $HERE/annotate_asm.js tmp/full.s "$@" | rustfilt -h > tmp/func.s
ls -l tmp/func.s
