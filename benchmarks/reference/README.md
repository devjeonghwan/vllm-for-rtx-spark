# NVIDIA reference benchmarks

The [main README](../../README.md#nvidia-reference) quotes NVIDIA's [Windows llama.cpp recipe](https://nvidia.github.io/local-ai-inference-recipes/hardware/rtx-spark/index.html?model=qwen%2Fqwen3.8-flash-next&os=windows&backend=llama-cpp&memory=78&precision=iq4-xs&variant=vision-mtp&memoryConfig=128). Its IQ4_XS measurements are reference results; this project's NVFP4 measurements use vLLM. Keep backend, precision, context limit and client metric with each result.

## SPEED-Bench coding

Use the [SPEED-Bench dataset](https://huggingface.co/datasets/nvidia/SPEED-Bench) and NVIDIA's [measurement framework and data preparation instructions](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/specdec_bench). The reported workload is the full coding category: 80 conversations and 89 requests in canonical order, concurrency 1, natural EOS and an 8,192-token output cap. Follow-up turns retain generated assistant history.

Some dataset prompts are masked and must be fetched from their original sources by the measurement framework. Running the unprepared dataset does not reproduce the full workload. The project measurements used this [dataset snapshot](https://huggingface.co/datasets/nvidia/SPEED-Bench/tree/454f88454792dfa3ccfd7ef15fff248efde44cd1).

NVIDIA's llama.cpp reference uses the [upstream SPEED client](https://github.com/ggml-org/llama.cpp/blob/efe3626b119ff81033e309c3ba07a770bef2c546/tools/server/bench/speed-bench/speed_bench.py) and server-native decode timings. This project's historical SPEED figures average each conversation's output tokens divided by its vLLM engine decode time. The stock llama.cpp client does not reproduce that vLLM metric; SSE delivery rates and request wall-time throughput are separate measurements.

## AA-AgentPerf-local

Use [AA-AgentPerf-local](https://github.com/ArtificialAnalysis/aa-agentperf-local/tree/70e1b996e05f) with its bundled `agentperf-default-v1` replay: eight tasks and 168 turns with recorded output lengths. Full replay requires at least a 65,536-token context; a shorter context is a different workload.

Run against an existing server using its served model ID:

```powershell
$ApiBase = "http://127.0.0.1:8000/v1"
$ModelId = "MODEL_ID"
git clone https://github.com/ArtificialAnalysis/aa-agentperf-local.git benchmarks/results/aa-agentperf-local
git -C benchmarks/results/aa-agentperf-local checkout 70e1b996e05f
uv run --project benchmarks/results/aa-agentperf-local agentperf-local run --replay agentperf-default-v1 --base-url $ApiBase --model $ModelId --client python --tool-choice none --no-power --no-progress --output-dir benchmarks/results/agentperf
```

Report `measured_duration_ms / 1000` from `summary.json` as completion seconds. This is distinct from summed request latency or the client process's wall time. Replay measures serving performance, not success at executing live agent tasks.

Download upstream tools and data locally; keep them and generated outputs under the ignored `benchmarks/results/` directory.
