"""Internal compiler-host entry point. Never used by inference."""
import importlib.metadata
import json
from pathlib import Path
import runpy
import sys


def main():
    settings = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
    runpy.run_path(str(Path(__file__).with_name("process_arch.py")))["require_windows_x64_process"]()
    for path in reversed(settings.get("python_paths", [])):
        sys.path.insert(0, str(Path(path).resolve()))
    if importlib.metadata.version("nvidia-cutlass-dsl") != "4.8.1":
        raise RuntimeError("This source pack requires CuTe DSL 4.8.1")
    script = Path(sys.argv[2]).resolve()
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *sys.argv[3:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
