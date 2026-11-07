# vLLM for RTX Spark

Optimized vLLM inference for NVIDIA RTX Spark N1X on Windows ARM64.

## Supported models

| Model | Precision | Default recipe |
|---|---|---|
| [Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) | NVFP4 / MXFP8 | [FAST MTP, 96 GB](configs/qwen3.8-flash-next/gpu-96gb-draft-kv-bf16.yaml) |
| [Gemma 4 12B IT FP8 Dynamic](https://huggingface.co/RedHatAI/gemma-4-12B-it-FP8-Dynamic) | FP8, text only | [8 GB BF16 KV](configs/gemma4-12b-fp8/text.yaml) |

## Install

Requirements: ARM64 Python 3.13 and an NVIDIA driver compatible with CUDA 13.4.

- Set GPU memory to 96 GB, leaving 32 GB for Windows.
- Set the Windows paging file to 98,304 MB initial / 131,072 MB maximum ([details](https://forums.developer.nvidia.com/t/rtx-spark-on-windows-cuda-allocations-consume-system-commit-even-with-96-gib-dedicated-gpu-memory/386453)).
- Qwen3.8-Flash-Next-FP4-Local needs at least 32 GB of free NVMe space for its PLE cache.

Download and extract the ARM64 ZIP from [Releases](https://github.com/devjeonghwan/vllm-for-rtx-spark/releases), then install with Python/pip:

```powershell
.\scripts\install.ps1 -Python python
```

Or, with uv installed:

```powershell
uv python install cpython-3.13-windows-aarch64-none
.\scripts\install.ps1 -UseUv
```

`-UseUv` uses the active virtual environment or creates `.venv`; `-Environment <path>` overrides it. Download model weights separately.

## Run

OpenAI-compatible API: `http://127.0.0.1:8000/v1`.

### Qwen3.8-Flash-Next-FP4-Local profiles

```powershell
.\.venv\Scripts\vllm.exe serve C:\models\Qwen3.8-Flash-Next-FP4-Local --config configs\qwen3.8-flash-next\gpu-96gb-draft-kv-bf16.yaml
```

The first start prepares the PLE cache and can take several minutes. Target KV is FP8; select the MTP draft KV precision below.

| Estimated GPU use | KV budget | BF16 draft KV | FP8 draft KV |
|---|---:|---|---|
| 90 GB | 12 GB | [Config](configs/qwen3.8-flash-next/gpu-90gb-draft-kv-bf16.yaml) | [Config](configs/qwen3.8-flash-next/gpu-90gb-draft-kv-fp8.yaml) |
| 96 GB | 18 GB | [Default](configs/qwen3.8-flash-next/gpu-96gb-draft-kv-bf16.yaml) | [Config](configs/qwen3.8-flash-next/gpu-96gb-draft-kv-fp8.yaml) |
| 108 GB | 27 GB | [Config](configs/qwen3.8-flash-next/gpu-108gb-draft-kv-bf16.yaml) | [Config](configs/qwen3.8-flash-next/gpu-108gb-draft-kv-fp8.yaml) |

The 108 GB profiles may use shared memory. [DFlash2](configs/qwen3.8-flash-next/gpu-96gb-dflash2.yaml) is experimental; FAST MTP is the default.

### Gemma 4 12B IT FP8 Dynamic profiles

```powershell
.\.venv\Scripts\vllm.exe serve C:\models\gemma-4-12B-it-FP8-Dynamic --config configs\gemma4-12b-fp8\text.yaml
```

| Estimated GPU use | KV budget | Profile |
|---|---:|---|
| 25 GB | 8 GB | [Default](configs/gemma4-12b-fp8/text.yaml) |

## Benchmarks

### NVIDIA reference

| Model | Benchmark | This project, NVFP4 | NVIDIA llama.cpp, IQ4_XS |
|---|---|---:|---:|
| Qwen3.8-Flash-Next-FP4-Local | AA-AgentPerf-local, full replay | 734.51 s | 1,118.16 s |
| Qwen3.8-Flash-Next-FP4-Local | SPEED-Bench coding, 80 conversations | 70.44 tok/s | 35.37 tok/s |

NVIDIA figures come from its [Windows recipe](https://nvidia.github.io/local-ai-inference-recipes/hardware/rtx-spark/index.html?model=qwen%2Fqwen3.8-flash-next&os=windows&backend=llama-cpp&memory=78&precision=iq4-xs&variant=vision-mtp&memoryConfig=128). Quantization and benchmark clients differ.

### Workload throughput

| Model | Concurrency | Mixed tok/s | Code tok/s | JSON schema tok/s | Prose tok/s |
|---|---:|---:|---:|---:|---:|
| Qwen3.8-Flash-Next-FP4-Local | 1 | 62.85 | 60.40 | 47.20 | 42.38 |
| Qwen3.8-Flash-Next-FP4-Local | 16 | 233.63 | 295.50 | 242.33 | 215.75 |
| Gemma 4 12B IT FP8 Dynamic | 1 | 17.58 | 17.38 | 17.43 | 17.38 |
| Gemma 4 12B IT FP8 Dynamic | 16 | 244.50 | 244.90 | 245.30 | 245.30 |

[See prompts, settings and runner](benchmarks/README.md).

## Build

Requires native ARM64 Python 3.13, Visual Studio ARM64 C++ tools and CUDA Toolkit 13.4.

```powershell
python -m pip install --only-binary=:all: -r requirements\runtime.txt -r requirements\vendor.txt -r requirements\build.txt
.\scripts\build-vllm.ps1 -Python python
```

The project targets vLLM 0.32.0. Source and build requirements are pinned in [manifest.json](patches/vllm/manifest.json); prebuilt native assets are included. See [source preparation](scripts/README.md) for source archives and patches.

[Tests](tests/README.md) · [Source revisions](sources.lock.json) · [Third-party notices](THIRD_PARTY_NOTICES.md)
