"""Stage the kernel pack, or rebuild it through its compiler provider."""
import argparse
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "scripts/source_tree.py"))["load"]()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", default="prebuilt")
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--install", action="store_true", help="Replace checkout assets after validation; preserve the old pack in the output directory")
    args = parser.parse_args()
    from _spark.kernels import build_kernels
    result = build_kernels(args.output, source_root=ROOT, provider=args.provider, settings=args.settings, install=args.install)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
