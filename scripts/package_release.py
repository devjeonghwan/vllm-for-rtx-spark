"""Package the release wheel and installation files in a Windows ARM64 ZIP."""
import argparse
import json
from pathlib import Path
import runpy
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
_VERSION = runpy.run_path(ROOT / "src/version.py")
VERSION = _VERSION["PROJECT_VERSION"]
WHEEL_VERSION = _VERSION["wheel_version"]()
CORE_WHEEL_PREFIX = "vllm-" + WHEEL_VERSION + "-"


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
    install_archive = output / f"vllm-for-rtx-spark-{VERSION}-win_arm64.zip"
    if install_archive.exists():
        raise FileExistsError("Release output already exists: " + install_archive.name)
    wheels = sorted(output.glob("*.whl"))
    core = [path for path in wheels if path.name.startswith(CORE_WHEEL_PREFIX)]
    if len(wheels) != 1 or len(core) != 1:
        raise RuntimeError("Build exactly one Windows ARM64 vLLM wheel first")
    subprocess.run([sys.executable, str(ROOT / "scripts/check_wheels.py"),
                    *map(str, core)], cwd=ROOT, check=True)
    install_entries = [(path, "wheels/" + path.name) for path in core]
    install_entries.append((_VERSION["manifest_path"](), "manifest.json"))
    for name in ("README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "sources.lock.json"):
        install_entries.append((ROOT / name, name))
    build_info = output / "build-info.json"
    if build_info.is_file():
        install_entries.append((build_info, "build-info.json"))
    for folder in ("configs", "requirements"):
        install_entries.extend((path, path.relative_to(ROOT).as_posix())
                               for path in (ROOT / folder).rglob("*") if path.is_file())
    install_entries.append((ROOT / "scripts/install.ps1", "scripts/install.ps1"))
    archive(install_archive, install_entries)
    print(json.dumps({"directory": str(output), "artifacts": [
        install_archive.name, core[0].name,
    ]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
