# Tests

Use native Windows ARM64 Python with the project dependencies installed.
Build-flow tests also require CMake, Ninja and Git; they do not compile native code.

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:VLLM_RTX_SPARK_SOURCE_DIR = 'C:\path\to\prepared-vllm'
python -I -X utf8 -B -m unittest discover -s tests -p '*_test.py'
```

Prepare the source with `python scripts/build_vllm.py prepare --source-dir <path>`. Add `--upstream-dir <pinned-checkout>` to use a local upstream checkout.

Without the source variable, source-based tests are skipped. An invalid configured path fails.

`smoke_api.py` is a manual HTTP check against a running model server.
