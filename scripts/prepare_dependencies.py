"""Prepare the release's fixed external inputs; never compile or install packages."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / ".build/downloads"
RUNTIME = ROOT / ".build/vendor"


def lock():
    return json.loads((ROOT / "sources.lock.json").read_text(encoding="utf-8"))["dependencies"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def files(root):
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and not any(p in (".git", "__pycache__", ".pytest_cache", "target", "build")
                                      or p.endswith(".egg-info") for p in path.relative_to(root).parts):
            yield path


def tree_digest(root):
    result = hashlib.sha256()
    for path in files(root):
        result.update(path.relative_to(root).as_posix().encode() + b"\0")
        result.update(bytes.fromhex(digest(path)))
    return result.hexdigest()


def member(root, name):
    relative = Path(name)
    if not name or relative.is_absolute() or relative.drive or ".." in relative.parts or "\\" in name:
        raise ValueError("Unsafe archive member: " + name)
    target = Path(root) / relative
    if not target.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Archive member escapes destination")
    return target


def fetch(spec, cache=DEFAULT_CACHE, offline=False):
    cache = Path(cache)
    path = cache / spec["filename"]
    member(cache, spec["filename"])
    if not path.exists():
        if offline:
            raise FileNotFoundError("Archive not cached: " + str(path))
        if not spec["url"].startswith("https://"):
            raise ValueError("External archives require HTTPS")
        cache.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".download")
        with urllib.request.urlopen(spec["url"], timeout=120) as response, temporary.open("xb") as output:
            shutil.copyfileobj(response, output)
        if digest(temporary) != spec["sha256"]:
            raise ValueError("Archive checksum mismatch: " + spec["filename"])
        temporary.rename(path)
    if digest(path) != spec["sha256"]:
        raise ValueError("Archive checksum mismatch: " + spec["filename"])
    return path


def write(root, name, raw):
    path = member(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def patches(root, names):
    if not names:
        return
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "--local", "core.autocrlf", "false"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "--local", "core.eol", "lf"], check=True)
    for name in names:
        patch = member(ROOT, name)
        subprocess.run(["git", "apply", "--check", "--whitespace=nowarn", str(patch)], cwd=root, check=True)
        subprocess.run(["git", "apply", "--whitespace=nowarn", str(patch)], cwd=root, check=True)


def cached(output, inputs):
    output = Path(output)
    receipt = output.with_name(output.name + ".json")
    if not output.exists():
        return False
    if not receipt.is_file():
        raise RuntimeError("Incomplete prepared dependency; preserve it and choose a fresh output: " + str(output))
    record = json.loads(receipt.read_text())
    if record["inputs"] != inputs or record["tree_sha256"] != tree_digest(output):
        raise RuntimeError("Prepared dependency differs; choose a fresh output: " + str(output))
    return True


def finish(output, inputs):
    receipt = Path(output).with_name(Path(output).name + ".json")
    with receipt.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump({"inputs": inputs, "tree_sha256": tree_digest(output)}, stream, indent=2)
        stream.write("\n")
    return Path(output)


def prepare_source(name, *, cache=DEFAULT_CACHE, offline=False, output=None):
    spec = lock()[name]
    if spec["kind"] != "sdist":
        raise ValueError("Expected a source archive: " + name)
    output = Path(output or ROOT / ".build/sources" / (name + "-" + spec["version"]))
    if cached(output, spec):
        return output
    archive = fetch(spec, cache, offline)
    output.mkdir(parents=True)
    with tarfile.open(archive) as source:
        prefix = None
        for item in source:
            if item.issym() or item.islnk() or not (item.isfile() or item.isdir()):
                raise ValueError("Unsupported source archive member: " + item.name)
            member(output, item.name)
            top = item.name.split("/")[0]
            if prefix is not None and top != prefix:
                raise ValueError("Source archive must have one root directory")
            prefix = top
            relative = item.name.partition("/")[2]
            if item.isfile():
                write(output, relative, source.extractfile(item).read())
    return finish(output, spec)


def patch_inputs(names):
    return {name: digest(member(ROOT, name)) for name in names}


def refresh_records(root):
    for info in sorted(Path(root).glob("*.dist-info")):
        metadata = info / "METADATA"
        name = next(line[6:] for line in metadata.read_text(encoding="utf-8").splitlines() if line.startswith("Name: "))
        spec = lock().get(name.replace("_", "-").lower())
        if spec is None or not spec.get("runtime"):
            continue  # Native packages keep their validated wheel metadata.
        members = [p for prefix in spec["modules"] for p in files(Path(root) / prefix)]
        members += [p for p in files(info) if p.name != "RECORD"]
        with (info / "RECORD").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            for path in sorted(members):
                token = base64.urlsafe_b64encode(bytes.fromhex(digest(path))).rstrip(b"=").decode()
                writer.writerow((path.relative_to(root).as_posix(), "sha256=" + token, path.stat().st_size))
            writer.writerow((info.name + "/RECORD", "", ""))


def prepare_runtime(*, cache=DEFAULT_CACHE, offline=False, output=RUNTIME):
    specs = {name: spec for name, spec in lock().items() if spec.get("runtime")}
    names = lock()["flashinfer-python"]["patches"]
    inputs = {"archives": specs, "patches": patch_inputs(names), "native": tree_digest(ROOT / "assets/python")}
    output = Path(output)
    if cached(output, inputs):
        return output
    # Fetch and verify everything before creating an output tree.
    archives = {name: fetch(spec, cache, offline) for name, spec in specs.items()}
    output.mkdir(parents=True)
    for path in files(ROOT / "assets/python"):
        if path.relative_to(ROOT / "assets/python").parts[0] != "builds.json":
            write(output, path.relative_to(ROOT / "assets/python").as_posix(), path.read_bytes())
    for name, spec in specs.items():
        with zipfile.ZipFile(archives[name]) as archive:
            for item in archive.infolist():
                member(output, item.filename)
                if item.is_dir():
                    continue
                top = item.filename.split("/")[0]
                if (top in spec["modules"] and Path(item.filename).suffix in (".py", ".json", ".md", ".txt")
                        or top == spec.get("dist_info") and not item.filename.endswith("/RECORD")):
                    write(output, item.filename, archive.read(item))
    patches(output, names)
    refresh_records(output)
    return finish(output, inputs)


def prepare_flashinfer_native(*, cache=DEFAULT_CACHE, offline=False, output=None):
    spec = lock()["flashinfer-python"]
    names = spec["native_patches"]
    inputs = {"archive": spec["sha256"], "patches": patch_inputs(names), "selection": 1}
    output = Path(output or ROOT / ".build/sources" / ("flashinfer-native-" + spec["version"]))
    if cached(output, inputs):
        return output
    archive = fetch(spec, cache, offline)
    output.mkdir(parents=True)
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            member(output, item.filename)
            if item.is_dir() or not item.filename.startswith("flashinfer/data/"):
                continue
            relative = item.filename.removeprefix("flashinfer/data/")
            top = relative.split("/")[0]
            if top not in ("cccl", "csrc", "include", "cutlass", "spdlog"):
                continue
            # Reproduce the existing sampler build snapshot, including CCCL's
            # .inl/Python files but excluding unused CUTLASS generators/templates.
            suffix = Path(relative).suffix
            if (top != "cccl" and suffix in (".inl", ".jinja")
                    or top in ("cutlass", "spdlog") and suffix == ".py"):
                continue
            target = relative if top == "cccl" else "vendor/" + relative
            write(output, target, source.read(item))
            if top == "include":
                write(output, "native-include/" + relative.removeprefix("include/"), source.read(item))
    patches(output, names)
    return finish(output, inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=("runtime", "oss-harmony", "tiktoken", "llguidance", "flashinfer-native"))
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    options = {"cache": args.cache, "offline": args.offline}
    if args.output is not None:
        options["output"] = args.output
    if args.component == "runtime":
        output = prepare_runtime(**options)
    elif args.component == "flashinfer-native":
        output = prepare_flashinfer_native(**options)
    else:
        output = prepare_source(args.component, **options)
    print(output)


if __name__ == "__main__":
    main()
