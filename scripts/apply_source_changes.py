"""Apply one library's declared patches and copies to its prepared source tree."""
import json
from pathlib import Path
import subprocess


def safe_relative(name):
    path = Path(name)
    if path.is_absolute() or path.drive or not path.parts or ".." in path.parts:
        raise ValueError("Expected a repository-relative source path: " + str(name))
    return path


def contained_path(root, name):
    root = Path(root).resolve()
    path = (root / safe_relative(name)).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Source path escapes its root: " + str(name))
    return path


def payload_files(root):
    root = Path(root)
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (not path.is_file() or path.suffix in (".pyc", ".pyo")
                or any(part in ("__pycache__", ".git", ".pytest_cache", ".ruff_cache")
                       or part.endswith(".egg-info") for part in relative.parts)):
            continue
        yield path, relative.as_posix()


def copy_inputs(manifest, *, source_root):
    """Plan repository-root copies without changing either tree."""
    source_root = Path(source_root).resolve()
    destinations = []
    entries = []
    for rule in manifest.get("copy", ()):
        original = contained_path(source_root, rule["source"])
        placed = safe_relative(rule["destination"])
        if not original.exists():
            raise FileNotFoundError(original)
        key = placed.as_posix().casefold()
        if any(key == old or key.startswith(old + "/") or old.startswith(key + "/") for old in destinations):
            raise ValueError("Overlapping source copy destinations: " + rule["destination"])
        destinations.append(key)
        paths = payload_files(original) if original.is_dir() else [(original, None)]
        for path, relative in paths:
            if not path.resolve().is_relative_to(source_root):
                raise ValueError("Project copy source escapes its directory: " + str(path))
            entries.append((path, (placed / relative if relative is not None else placed).as_posix()))
    return entries


def copy_sources(source, manifest, *, source_root):
    """Preflight all destinations, then copy files and folders without overwrite."""
    source = Path(source).resolve()
    source_root = Path(source_root).resolve()
    entries = copy_inputs(manifest, source_root=source_root)
    for rule in manifest.get("copy", ()):
        destination = contained_path(source, rule["destination"])
        if destination.exists():
            raise RuntimeError("Source copy destination already exists: " + rule["destination"])
        for parent in destination.parents:
            if parent == source.parent:
                break
            if parent.exists() and not parent.is_dir():
                raise RuntimeError("Source copy parent is not a directory: " + str(parent))
    for path, relative in entries:
        destination = contained_path(source, relative)
        if destination.exists():
            raise RuntimeError("Source copy file already exists: " + relative)
    for rule in manifest.get("copy", ()):
        if contained_path(source_root, rule["source"]).is_dir():
            contained_path(source, rule["destination"]).mkdir(parents=True, exist_ok=False)
    for path, relative in entries:
        destination = contained_path(source, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(path.read_bytes())
    return len(entries)


def apply_source_changes(source, manifest_path, *, source_root):
    """Read one explicit manifest; its caller owns fetching and building.

    The prepared source must have its own Git root so git apply cannot silently
    interpret paths against a surrounding checkout. Failures keep the source
    tree for inspection. No library discovery, compilation or hashes are used.
    """
    source = Path(source).resolve()
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not (source / ".git").exists():
        raise ValueError("Prepared source needs its own Git root: " + str(source))
    patches = [contained_path(manifest_path.parent, name) for name in manifest.get("patches", ())]
    for patch in patches:
        if not patch.is_file():
            raise FileNotFoundError(patch)
    copy_inputs(manifest, source_root=source_root)
    for patch in patches:
        subprocess.run(["git", "apply", "--check", "--whitespace=nowarn", str(patch)], cwd=source, check=True)
        subprocess.run(["git", "apply", "--whitespace=nowarn", str(patch)], cwd=source, check=True)
    copied = copy_sources(source, manifest, source_root=source_root)
    return {"patches_applied": len(patches), "copied_files": copied}
