"""Create source and installation archives from the checked-out project."""
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
MANIFEST = _VERSION["load_manifest"]()
WHEEL_VERSION = _VERSION["wheel_version"]()
CORE_WHEEL_PREFIX = "vllm-" + WHEEL_VERSION + "-"
SOURCE_DIRECTORIES = (
    "src", "third_party", "assets", "patches", "configs",
    "requirements", "scripts", "tests", "docs",
)
SOURCE_FILES = (
    "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "sources.lock.json",
    "pyproject.toml", ".gitignore", ".gitattributes",
)


def archive(output, entries):
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as target:
        for path, relative in sorted(entries, key=lambda entry: entry[1]):
            target.write(path, relative)


def source_entries(root=ROOT):
    """Collect build inputs from the checkout, including unstaged new files."""
    root = Path(root)
    excluded = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache",
                ".cache", ".build", "build", "dist", ".validation", ".env"}
    entries = []
    candidates = (root / name for name in SOURCE_FILES)
    for path in candidates:
        if path.is_file() and not path.is_symlink():
            entries.append((path, path.name))
    for folder in SOURCE_DIRECTORIES:
        for path in (root / folder).rglob("*"):
            relative = path.relative_to(root)
            if (not path.is_file() or path.is_symlink()
                    or path.suffix.lower() in {".pyc", ".pyo", ".log"}
                    or path.name.lower() == "local-build.json"
                    or any(part.lower() in excluded or part.lower().endswith(".egg-info")
                           or part.lower().startswith((".venv", ".env.")) for part in relative.parts)):
                continue
            entries.append((path, relative.as_posix()))
    return sorted(entries, key=lambda item: item[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(exist_ok=True)
    for name in (f"vllm-for-rtx-spark-{VERSION}-source.zip",
                 f"vllm-for-rtx-spark-{VERSION}-win_arm64.zip"):
        if (output / name).exists():
            raise FileExistsError("Release output already exists: " + name)
    wheels = sorted(output.glob("*.whl"))
    core = [path for path in wheels if path.name.startswith(CORE_WHEEL_PREFIX)]
    if len(wheels) != 1 or len(core) != 1:
        raise RuntimeError("Build exactly one Windows ARM64 vLLM wheel first")
    subprocess.run([sys.executable, str(ROOT / "scripts/check_wheels.py"),
                    *map(str, core)], cwd=ROOT, check=True)
    archive(output / f"vllm-for-rtx-spark-{VERSION}-source.zip", source_entries())
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
    archive(output / f"vllm-for-rtx-spark-{VERSION}-win_arm64.zip", install_entries)
    print(json.dumps({"directory": str(output), "artifacts": [
        f"vllm-for-rtx-spark-{VERSION}-source.zip",
        f"vllm-for-rtx-spark-{VERSION}-win_arm64.zip", core[0].name,
    ]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
