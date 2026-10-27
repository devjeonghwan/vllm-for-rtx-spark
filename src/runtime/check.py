"""Installation checks. Model serving uses the normal vllm command."""
import argparse
import importlib.metadata
import json
import platform
import struct
import sys

from packaging.version import InvalidVersion, Version


def inspect_installation():
    from . import resources_root
    from ..version import load_manifest, wheel_version

    checks = []
    target_platform = load_manifest()["platform"]

    def check(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    expected_os = {"windows": "win32"}.get(target_platform["os"], target_platform["os"])
    expected_cpus = {"arm64", "aarch64"} if target_platform["cpu"] == "arm64" else {target_platform["cpu"]}
    check("platform", sys.platform == expected_os and platform.machine().lower() in expected_cpus,
          f"{sys.platform}/{platform.machine()}")
    expected_python = tuple(int(part) for part in target_platform["python"].split(".")[:2])
    check("python", sys.version_info[:2] == expected_python, platform.python_version())
    expected = {"vllm": wheel_version(), "torch": target_platform["torch"],
                "triton-windows": target_platform["triton_windows"],
                "flash-attn": target_platform["flash_attention"]}
    for name, version in expected.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        try:
            matches = Version(actual) == Version(version)
        except InvalidVersion:
            matches = False
        check(name, matches, actual)
    root = resources_root()
    dlls = list(root.rglob("*.dll"))
    check("native libraries", bool(dlls), str(len(dlls)))
    for dll in dlls:
        with dll.open("rb") as stream:
            header = stream.read(64)
            machine = None
            if len(header) == 64 and header[:2] == b"MZ":
                stream.seek(struct.unpack_from("<I", header, 60)[0])
                signature = stream.read(6)
                if signature[:4] == b"PE\0\0":
                    machine = struct.unpack_from("<H", signature, 4)[0]
        check(str(dll.relative_to(root)), machine == 0xAA64, hex(machine or 0))
    return {"ok": all(row["ok"] for row in checks), "checks": checks}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["doctor"])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = inspect_installation()
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for row in result["checks"]:
            print(f"{'OK' if row['ok'] else 'FAIL'} {row['name']}: {row['detail']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
