# Benchmarks

Reproduce the performance tables in the [main README](../README.md). Run commands from the repository root against an already running server; replace `MODEL_ID` with its served model ID from `/v1/models`.

| Directory | Contents |
|---|---|
| [throughput/](throughput/) | Workload runner and fixed prompts |
| [reference/](reference/README.md) | NVIDIA reference benchmarks and upstream tools |
| [tests/](tests/) | Offline runner tests |
| `results/` | Local outputs, excluded from Git |

## Workload throughput

Install the client dependency and run:

```powershell
python -m pip install -r benchmarks/requirements.txt
python benchmarks/throughput/run.py --base-url http://127.0.0.1:8000 --model MODEL_ID --output benchmarks/results/throughput --execute
```

Or use uv:

```powershell
uv run --no-project --with-requirements benchmarks/requirements.txt python benchmarks/throughput/run.py --base-url http://127.0.0.1:8000 --model MODEL_ID --output benchmarks/results/throughput --execute
```

Omit `--execute` to inspect the plan without sending requests. Use a fresh output directory for each run.

The fixed workloads are Mixed, Code, JSON schema and Prose. Mixed cycles eight prompts. Each workload runs at concurrency 1 and 16 for two 20-second windows after one 32-token warmup request, with a 512-token output cap.

Sampling uses temperature 1, top-p 0.95, thinking off and natural EOS. Other sampling parameters follow the model's generation config. Start the server with `--enable-prefix-caching` to match the reported workload measurements.

Throughput is the number of output token IDs received inside each window divided by 20 seconds, averaged across both runs. Warmup and tokens received after the window are excluded; concurrency 16 reports aggregate throughput. The runner requires SSE token IDs and matching completion usage.

JSON uses schema-constrained generation. Only EOS-complete JSON responses are checked against the schema; capped responses are marked incomplete. Throughput is not a correctness score. Settings, prompts, streams and per-request results are saved in the output directory.

## NVIDIA reference

The main README also includes figures from NVIDIA's Windows llama.cpp recipe. These use separate benchmarks and clients:

| Benchmark | Workload | Metric |
|---|---|---|
| SPEED-Bench coding | 80 coding conversations | Mean decode tok/s, higher is better |
| AA-AgentPerf-local | Full recorded agent replay | Completion seconds, lower is better |

See [reference setup and measurement definitions](reference/README.md). Quantization and timing boundaries differ from this project's workload runner.
