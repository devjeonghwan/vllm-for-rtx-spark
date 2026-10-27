"""Export original CuTe artifacts without running a CUDA kernel."""
import hashlib
from pathlib import Path
import re
import shutil
import struct


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(compiled, directory, name):
    if getattr(compiled, "engine", None) is not None:
        raise RuntimeError("AOT compilation unexpectedly created a JIT execution engine")
    ir = directory / (name + ".mlir")
    ir.write_text(str(compiled.ir_module), encoding="utf-8")
    artifacts = {}
    for kind, value in (("ptx", compiled.__ptx__), ("cubin", compiled.__cubin__)):
        path = directory / (name + "." + kind)
        if isinstance(value, str) and kind == "ptx" and ("\n" in value or ".version" in value):
            path.write_text(value, encoding="utf-8")
        elif isinstance(value, str) and Path(value).is_file():
            shutil.copyfile(value, path)
        elif isinstance(value, bytes):
            path.write_bytes(value)
        else:
            raise RuntimeError("Missing compiler artifact: " + kind)
        artifacts[kind] = dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path))
    raw = Path(artifacts["cubin"]["path"]).read_bytes()
    if raw[:4] != b"\x7fELF" or struct.unpack_from("<H", raw, 18)[0] != 190:
        raise RuntimeError("Expected CUDA device code")
    if not re.search(r"\.target\s+sm_121a\b", Path(artifacts["ptx"]["path"]).read_text()):
        raise RuntimeError("Unexpected GPU target")
    return {"artifacts": artifacts, "lowered_IR": str(ir), "lowered_IR_sha256": digest(ir),
            "host_signature": str(compiled.execution_args.signature),
            "kernel_info": {str(k): [{"name": str(getattr(a, "name", "")), "value": str(getattr(a, "value", a))}
                                     for a in v] for k, v in compiled.kernel_info.items()},
            "JIT_engine_created": False}
