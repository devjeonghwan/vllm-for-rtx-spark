"""Build and stage the recipe's original oss-harmony dependency for ARM64."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import runpy
import shutil
import struct
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
_DEPENDENCIES = runpy.run_path(ROOT / "scripts/prepare_dependencies.py")
_SOURCE = _DEPENDENCIES["lock"]()["oss-harmony"]
SOURCE = ROOT / ".build/sources" / ("oss-harmony-" + _SOURCE["version"])
PAYLOAD = ROOT / "assets/python"
DIST_INFO = "oss_harmony-" + _SOURCE["version"] + ".dist-info"
WHEEL = "oss_harmony-" + _SOURCE["version"] + "-cp38-abi3-win_arm64.whl"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def arm64_image(raw):
    if len(raw) < 64 or raw[:2] != b"MZ":
        return False
    at = struct.unpack_from("<I", raw, 60)[0]
    return raw[at:at + 6] == b"PE\0\0\x64\xaa"


def verify_source(*, offline=False):
    source = _DEPENDENCIES["prepare_source"]("oss-harmony", offline=offline)
    if source != SOURCE:
        raise RuntimeError("Harmony build source differs from its pinned location")
    return _DEPENDENCIES["lock"]()["oss-harmony"]


def stage(wheel):
    contents = {}
    with zipfile.ZipFile(wheel) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            relative = Path(entry.filename)
            if relative.is_absolute() or relative.drive or ".." in relative.parts:
                raise ValueError("Unsafe Harmony wheel member")
            if relative.parts[0] not in ("openai_harmony", DIST_INFO):
                raise ValueError("Unexpected Harmony wheel member: " + entry.filename)
            raw = archive.read(entry)
            if relative.suffix == ".pyd" and not arm64_image(raw):
                raise RuntimeError("Harmony native extension is not ARM64")
            name = relative.as_posix()
            if name.endswith(".cyclonedx.json"):
                # Cargo's SBOM embeds its local source checkout in bom-ref.
                # Preserve the dependency graph with a portable identity.
                raw = re.sub(r"path\+file:///[^\"#]+#oss-harmony@" + re.escape(_SOURCE["version"]),
                             "pkg:cargo/oss-harmony@" + _SOURCE["version"], raw.decode("utf-8")).encode("utf-8")
            contents[name] = raw
    if not any(name.endswith(".pyd") for name in contents):
        raise RuntimeError("Harmony wheel lacks a native extension")
    record_name = DIST_INFO + "/RECORD"
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    for name, raw in sorted(contents.items()):
        if name != record_name:
            token = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode()
            writer.writerow((name, "sha256=" + token, len(raw)))
    writer.writerow((record_name, "", ""))
    contents[record_name] = buffer.getvalue().encode()
    for name, raw in contents.items():
        output = PAYLOAD / name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(raw)
    return {name: sha(raw) for name, raw in contents.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, help="Stage a completed native ARM64 wheel instead of rebuilding")
    parser.add_argument("--rust-root", type=Path, help="Optional portable Rust prefix containing bin/rustc.exe")
    parser.add_argument("--maturin", default="maturin")
    parser.add_argument("--cargo-home", type=Path, default=ROOT / "build/cargo-harmony")
    parser.add_argument("--target-dir", type=Path, default=ROOT / "build/harmony-target")
    parser.add_argument("--output", type=Path, default=ROOT / "build/harmony-wheelhouse")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    source = verify_source(offline=args.offline)
    tools = {}
    reused = args.wheel is not None
    if reused:
        wheel = args.wheel.resolve()
    else:
        if sys.platform != "win32" or platform.machine().upper() not in ("ARM64", "AARCH64"):
            raise RuntimeError("Build with native Windows ARM64 Python")
        environment = dict(os.environ)
        if args.rust_root:
            prefix = args.rust_root.resolve()
            environment["PATH"] = str(prefix / "bin") + os.pathsep + environment.get("PATH", "")
            environment["RUSTC"] = str(prefix / "bin/rustc.exe")
            environment["RUSTUP_HOME"] = str(ROOT / "build/rustup-harmony-unused")
        environment.update(CARGO_HOME=str(args.cargo_home.resolve()),
                           CARGO_TARGET_DIR=str(args.target_dir.resolve()),
                           CARGO_BUILD_JOBS="2", PYO3_PYTHON=sys.executable,
                           CARGO_TERM_COLOR="never")
        for name in ("rustc", "cargo"):
            path = shutil.which(name, path=environment["PATH"])
            if path is None or not arm64_image(Path(path).read_bytes()):
                raise RuntimeError("Native ARM64 Rust tool was not found: " + name)
            tools[name] = subprocess.check_output([path, "--version"], env=environment, text=True).strip()
        maturin = shutil.which(args.maturin, path=environment["PATH"]) or args.maturin
        maturin = str(Path(maturin).resolve())
        if not arm64_image(Path(maturin).read_bytes()):
            raise RuntimeError("Use native Windows ARM64 maturin")
        tools["maturin"] = subprocess.check_output([maturin, "--version"], env=environment, text=True).strip()
        args.output.mkdir(parents=True, exist_ok=True)
        command = [maturin, "build", "--manifest-path", str(SOURCE / "Cargo.toml"),
                   "--release", "--locked", "--target", "aarch64-pc-windows-msvc",
                   "--interpreter", sys.executable, "--out", str(args.output.resolve()), "--jobs", "2"]
        if args.offline:
            command.append("--offline")
        with (args.output / "build.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, env=environment, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT, check=True)
        wheel = args.output / WHEEL
    if wheel.name != WHEEL:
        raise ValueError("Expected Harmony ARM64 wheel: " + WHEEL)
    stage(wheel)
    verify_source(offline=args.offline)
    record = {"distribution": "oss-harmony", "version": source["version"],
              "source": "oss-harmony", "source_patched": False,
              "wheel": wheel.name, "wheel_sha256": sha(wheel.read_bytes()), "tools": tools,
              "reused_prebuilt_wheel": reused, "target": "aarch64-pc-windows-msvc",
              "SBOM_local_build_paths_normalized": True}
    provenance = PAYLOAD / "builds.json"
    records = json.loads(provenance.read_text(encoding="utf-8")) if provenance.exists() else {}
    records["oss-harmony"] = record
    provenance.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
