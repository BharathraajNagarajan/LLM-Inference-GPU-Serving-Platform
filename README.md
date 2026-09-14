# LLM Inference & GPU Serving Platform

> **Status: scaffolding phase.** No serving, benchmarking, or model code has been implemented yet. This README is a skeleton describing intent; sections without content are explicitly marked TODO and contain no invented numbers or claims.

## 1. Problem

Serving large language models for inference is a distinct engineering problem from training them. A production LLM-serving system has to answer requests with low latency, sustain throughput under concurrent load, and make efficient use of expensive, memory-constrained GPU hardware — simultaneously.

This is hard for several intertwined reasons:

- **Autoregressive generation is sequential.** Each output token depends on all previous tokens, so generation cannot be trivially parallelized the way a single forward pass over a batch of independent inputs can.
- **Memory is the binding constraint.** The KV cache (attention key/value state for every token in every in-flight sequence) grows with sequence length and batch size, and competes with model weights for limited GPU VRAM.
- **Requests are heterogeneous.** Prompts and desired output lengths vary per request, which makes naive static batching wasteful — short requests wait on long ones, and GPU utilization suffers.
- **Latency and throughput trade off against each other.** Batching more requests together improves GPU utilization and total throughput, but can increase per-request latency; serving systems have to balance both under real, bursty traffic.
- **Correctness and reliability matter under failure.** GPUs can run out of memory, requests can be malformed or unbounded in length, and a serving system needs to degrade predictably rather than silently.

This project intends to explore these problems concretely: standing up an LLM inference server (via vLLM), measuring its actual behavior under load, and documenting real, measured trade-offs rather than assumed ones.

## 2. Architecture

Intended high-level component flow:

```
Client
  |
  |  HTTP request (prompt, generation params)
  v
API layer (src/server/)
  |
  |  translates HTTP request into an inference request
  v
vLLM engine
  |
  |  schedules request into a batch, manages KV cache
  v
Model (weights loaded once, resident in GPU memory)
  |
  |  forward pass(es), token-by-token generation
  v
GPU (compute + memory for weights, activations, KV cache)
  |
  |  generated tokens streamed/returned
  v
API layer -> Client (response)
```

Each stage above is a placeholder for now — `src/server/` currently contains only package scaffolding, not an implemented API or vLLM integration. This diagram describes the intended shape of the system, not a built or verified one.

## 3. Request Flow

At a high level, serving a single generation request through vLLM is expected to involve two distinct phases:

- **Prefill**: the full input prompt is processed in one (or a few) forward pass(es) to populate the initial KV cache for that sequence. This phase is typically compute-bound, since it processes many tokens at once.
- **Decode**: the model then generates output tokens one at a time, each step attending over the growing KV cache (prompt + previously generated tokens) and appending a single new token. This phase is typically memory-bandwidth-bound, since each step does comparatively little compute but must read the full KV cache and model weights.

A serving engine like vLLM is expected to interleave prefill and decode work across many concurrent requests — batching decode steps from multiple in-flight sequences together, and injecting new requests' prefill work alongside ongoing decodes — in order to keep the GPU utilized. The specifics of how this project's deployment behaves (batching policy, scheduling behavior under load, actual prefill/decode timing split) are not yet known and will be described here only once observed from real runs, not assumed in advance.

## 4. Local Setup

This machine has Python 3.14 as the system default, but vLLM/PyTorch do not yet reliably support 3.14. The project venv must be created with Python 3.10 explicitly.

```
py -3.10 -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
```

Then install dependencies (see [GPU Setup](#5-gpu-setup) below for why `torch` needs a non-default install command). This procedure has been verified end-to-end on this machine by running `src/baseline/naive_inference.py` (see [`src/baseline/`](src/baseline/)) — the script loads `Qwen/Qwen2.5-0.5B-Instruct` and runs a single synchronous generation. It is a minimal sanity-check baseline, not the API server (`src/server/` is not yet implemented) and not a formal benchmark run (see [Benchmark Methodology](#7-benchmark-methodology) and [Actual Benchmark Results](#8-actual-benchmark-results), both still TODO).

TODO: fill in remaining local setup instructions (server startup, config via `.env`) once `src/server/` is implemented.

## 5. GPU Setup

This machine has an NVIDIA GeForce GTX 1050 (3 GB VRAM). `nvidia-smi` reports driver version 451.67 with a max supported CUDA version of **11.0** in its header.

**Important — `pip install torch` alone silently installs a CPU-only build on this platform.** Verified directly: a plain `pip install torch transformers` resolved `torch==2.14.0+cpu`, and `torch.cuda.is_available()` returned `False` — no error, no warning, just silent CPU-only operation. Do not rely on the bare command; install a CUDA build explicitly:

```
venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu118
venv\Scripts\python.exe -m pip install transformers
```

This installed `torch==2.7.1+cu118` (CUDA 11.8 build). Note the version tension: the driver header advertises max CUDA **11.0**, while the installed torch build targets CUDA **11.8** — by NVIDIA's published compatibility table this looks like a mismatch that should not work. Verified empirically anyway: `torch.cuda.is_available()` returned `True`, `torch.cuda.get_device_name(0)` correctly reported "GeForce GTX 1050", and `src/baseline/naive_inference.py` ran a real generation on-device with GPU memory actually allocated (peak allocated memory observed in the low hundreds of MiB for this small model, well inside the 3 GB budget). So CUDA execution works in practice on this machine today. Treat this as an observed-but-unexplained mismatch, not a resolved one — it should be re-checked if the driver, torch version, or CUDA index tag ever changes, since a version-table mismatch that happens to work now is not the same as a guaranteed-compatible pairing.

If CUDA is unavailable at run time, code in this project is expected to fall back to CPU with an explicit printed warning (see `src/baseline/naive_inference.py`) rather than failing silently or fabricating GPU results.

TODO: fill in remaining GPU setup notes (e.g. any container/driver-level GPU passthrough config) once `docker/` and `k8s/` have real content.

## 6. API Example

TODO: fill in after implementation.

## 7. Benchmark Methodology

The benchmark harness lives in [`benchmark/run_benchmark.py`](benchmark/run_benchmark.py), with shared measurement/aggregation logic in [`metrics/`](metrics/) (`metrics/records.py`, `metrics/aggregate.py`, `metrics/io.py`). It currently targets only the naive synchronous FastAPI baseline server (`src/server/main.py`) — **not vLLM**. This is deliberately the "before" side of a future naive-vs-vLLM comparison, so every record and output file is explicitly labeled `backend: "naive_fastapi_baseline"` and every output filename is prefixed `naive_baseline_...`, to keep it distinguishable from (and never overwritten or merged with) a future vLLM benchmark run.

**What it measures**, per request:
- success/failure and HTTP status code
- latency (wall-clock time from just before the HTTP POST to just after the response is received)
- input token count and output token count (as reported by the server's `/generate` response body)

And per concurrency level (aggregated from the per-request records, in `metrics/aggregate.py`):
- total/successful/failed request counts
- wall-clock duration of the whole batch at that concurrency level
- throughput, as successful requests per second of that wall-clock duration (not the sum of individual request latencies — this is what actually captures overlap between concurrent requests)
- output tokens per second, computed the same way
- latency mean, median, min, max, and p95 across the batch

All of the above are computed directly from observed per-request data; nothing is estimated, extrapolated, or interpolated across runs.

**How requests are generated:** the harness sends real HTTP POST requests to a `/generate` endpoint on an already-running instance of the server (the harness does not start or stop the server itself — it must be started separately first). Concurrency is implemented with a `ThreadPoolExecutor` sized to the concurrency level being tested; at concurrency level *N*, *N* requests are in flight against the server at once.

**Concurrency levels and the safety cap:** levels to test are configurable via `--concurrency-levels` (default `1,2,4`) and requests-per-level via `--requests-per-level` (default `6`). The harness hard-codes `MAX_VERIFIED_CONCURRENCY = 4` and refuses to run any requested level above it — checking *all* requested levels and aborting before sending a single request if any exceed the cap, rather than partially running and stopping mid-way. This cap exists because the baseline server holds one shared, unlocked model object across all requests (see `src/server/main.py`); concurrency levels 1, 2, and 4 were manually verified beforehand to return correct, non-corrupted, non-cross-contaminated responses under concurrent load, but higher levels have not been tested and are not assumed safe. Raising the cap requires deliberately editing the constant in `benchmark/run_benchmark.py`, not just passing a higher `--concurrency-levels` value — this is intentional friction, not an oversight.

**Prompt set:** a fixed pool of six short, factual, single-turn prompts (distinct "What is the capital of `<country>`?" questions). Requests beyond the pool size cycle through it via modulo indexing. The prompts were originally chosen for a separate concurrency-safety check because each has an independently verifiable, near-deterministic expected answer, making cross-request contamination easy to spot by inspection; the benchmark harness reuses the same pool for continuity, though verifying correctness is not itself part of what this harness measures.

**Generation parameters:** `max_new_tokens` is configurable via `--max-new-tokens` (default `32`) and passed through to the server on every request. Decoding is otherwise whatever the server itself does (currently greedy, `do_sample=False`, hard-coded server-side) — the harness does not control or vary sampling parameters.

**Known confound — no warm-up phase:** the harness does not issue a separate warm-up request before measuring. The first request of a run can include one-time CUDA/kernel warm-up cost on top of actual generation time, which will visibly skew that run's mean/p95 latency at low concurrency (most noticeably at concurrency level 1, where there's only a handful of requests to average over). This is a known limitation of the current harness, not corrected for automatically.

**Output:** each run writes two files into `results/`, timestamped in UTC: a raw per-request CSV (`naive_baseline_<timestamp>_raw.csv`) and an aggregated per-level JSON summary (`naive_baseline_<timestamp>_summary.json`). Both are excluded from version control by `.gitignore` (only `results/.gitkeep` is tracked), so historical runs are not part of the repo's history unless explicitly exported elsewhere.

## 8. Actual Benchmark Results

Data source: [`results/naive_baseline_20260914T041932Z_summary.json`](results/naive_baseline_20260914T041932Z_summary.json), a single run of the naive FastAPI baseline (`backend: "naive_fastapi_baseline"`) with `max_new_tokens=32` and `requests_per_level=6`, at concurrency levels 1, 2, and 4, with GPU telemetry sampling (via `pynvml`, every 200ms) enabled at every level. This is the authoritative naive-baseline dataset — it supersedes the two earlier partial runs (one covering latency/throughput only at 1/2/4, one covering GPU telemetry only at 1/2). All numbers below are copied directly from that one summary file; none are blended from the earlier runs.

| Concurrency | Successful/Total | Wall-clock (s) | Throughput (req/s) | Output tok/s | Latency mean (s) | Latency median (s) | Latency p95 (s) | GPU util % (min/mean/max) | GPU mem MiB (min/mean/max) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 6/6 | 14.4304 | 0.4158 | 3.3263 | 2.4048 | 0.7831 | 8.2584 | 0.0 / 11.34 / 47.0 | 1111.29 / 1456.15 / 1637.92 |
| 2 | 6/6 | 3.4750 | 1.7266 | 13.8131 | 1.0691 | 1.1223 | 1.2356 | 33.0 / 45.5 / 56.0 | 1637.92 / 1644.42 / 1645.92 |
| 4 | 6/6 | 3.4741 | 1.7271 | 13.8166 | 1.7446 | 2.0218 | 2.3405 | 37.0 / 47.375 / 56.0 | 1645.92 / 1667.42 / 1681.92 |

**Caveats — read before drawing conclusions:**
- **Single run, small sample.** Each concurrency level reflects one run of only 6 requests. There is no repetition and no variance/confidence interval; these numbers should be treated as a single observed data point, not a stable estimate. GPU telemetry sample counts also differ across levels (62 samples at concurrency 1 vs. 16 at concurrency 2 and 4) simply because the fixed 200ms polling interval was applied over a longer wall-clock window at concurrency 1 — not because more or fewer measurements were "available."
- **No warm-up (known confound, see [Section 7](#7-benchmark-methodology)).** No warm-up request was issued before measuring, so one-time CUDA/kernel warm-up cost is included in the timings — most visibly at concurrency 1, where `latency_s_max` (10.67s) and the resulting `latency_s_mean`/`latency_s_p95` are pulled well above `latency_s_median`, consistent with one slow first request skewing a small batch. This is why concurrency 1's mean/p95 look disproportionately worse than concurrency 2's and 4's despite less contention. The same cold-start request is also visible in the GPU utilization numbers: concurrency 1's mean utilization (11.34%) is pulled down by idle gaps between the six sequential, mostly-fast requests, while its max (47.0%) reflects a brief compute burst during generation.
- These results are for the naive synchronous baseline only (no batching, no vLLM). They establish a "before" reference point and are not a comparison yet — no vLLM numbers exist at this time.

## 9. Interpretation

TODO: fill in after implementation.

## 10. Kubernetes Deployment

**Status: manifests written and syntax-validated only — NOT applied to any live cluster.** `k8s/deployment.yaml` and `k8s/service.yaml` exist and were validated as well-formed YAML and checked with `kubectl apply --dry-run=client`, but no GPU-enabled Kubernetes cluster is available in this project's environment, so no pod described by them has ever actually started or served a request. See [`k8s/README.md`](k8s/README.md) for the full status note, what's in each manifest, and the known gaps (no `Dockerfile` yet, GPU scheduling unverified in practice) before this could actually be applied.

TODO: fill in an actual apply/run narrative once a GPU-enabled cluster is available — at that point this section must explicitly say whether the manifests were only dry-run/schema-validated versus actually applied to and exercised on a live cluster. These are materially different claims and must not be conflated.

## 11. Failure/Reliability Considerations

TODO: fill in after implementation.

## 12. Limitations

TODO: fill in after implementation.

## 13. Future Work

The following are intentionally out of scope for the current phase of this project:

- Multi-GPU serving (tensor parallelism / pipeline parallelism across multiple devices)
- Custom CUDA kernels
- TensorRT-LLM integration
- Quantization (e.g. INT8/INT4/AWQ/GPTQ) of model weights
- Autoscaling (horizontal scaling of serving replicas based on load)

These may be revisited after the core single-GPU serving and benchmarking loop is implemented and measured.
