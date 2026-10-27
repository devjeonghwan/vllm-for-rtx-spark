"""Create source and installation archives from the checked-out project."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.1.0"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def archive(output, entries):
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as target:
        for path, relative in sorted(entries, key=lambda entry: entry[1]):
            target.write(path, relative)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(exist_ok=True)
    for name in (f"vllm-for-rtx-spark-{VERSION}-source.zip",
                 f"vllm-for-rtx-spark-{VERSION}-win_arm64.zip", "SHA256SUMS"):
        if (output / name).exists():
            raise FileExistsError("Release output already exists: " + name)
    wheels = sorted(output.glob("*.whl"))
    core = [path for path in wheels if path.name.startswith("vllm-0.32.0+spark.1-")]
    if len(wheels) != 1 or len(core) != 1:
        raise RuntimeError("Build exactly one Windows ARM64 vLLM wheel first")
    subprocess.run([sys.executable, str(ROOT / "scripts/check_wheels.py"),
                    *map(str, core)], cwd=ROOT, check=True)
    checksums = {path.name: digest(path) for path in core}
    wheel_manifest = output / "wheel-sha256.json"
    wheel_manifest.write_text(json.dumps(checksums, indent=2) + "\n", encoding="utf-8")
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    source_entries = [(ROOT / name, name) for name in tracked if name and (ROOT / name).is_file()]
    if not source_entries:
        raise RuntimeError("Stage source files in Git before packaging")
    archive(output / f"vllm-for-rtx-spark-{VERSION}-source.zip", source_entries)
    install_entries = [(path, "wheels/" + path.name) for path in core]
    install_entries.append((wheel_manifest, "wheels/wheel-sha256.json"))
    for name in ("README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "sources.lock.json"):
        install_entries.append((ROOT / name, name))
    build_info = output / "build-info.json"
    if build_info.is_file():
        install_entries.append((build_info, "build-info.json"))
    for folder in ("configs", "requirements"):
        install_entries.extend((path, path.relative_to(ROOT).as_posix())
                               for path in (ROOT / folder).rglob("*") if path.is_file())
    install_entries.append((ROOT / "scripts/install.ps1", "scripts/install.ps1"))
    archive(output / f"vllm-for-rtx-spark-{VERSION}-win_arm64.zip", install_entries)
    records = {path.name: digest(path) for path in output.iterdir()
               if path.is_file() and path.name != "SHA256SUMS"}
    (output / "SHA256SUMS").write_text("".join(f"{value}  {name}\n" for name, value in sorted(records.items())), encoding="utf-8")
    print(json.dumps({"directory": str(output), "artifacts": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
