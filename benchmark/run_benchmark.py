"""Benchmark harness shared by two backends:

- naive: the naive synchronous FastAPI baseline server (src/server/main.py),
  POSTing to /generate with the {prompt, max_new_tokens} request shape.
- vllm: a vLLM OpenAI-compatible server, POSTing to /v1/chat/completions
  with the {model, messages, max_tokens} request shape.

Select the backend with --backend. Sends a configurable number of requests
to a running server instance at one or more concurrency levels, and
records per-request latency, throughput, and token counts to results/ as
CSV (raw) and JSON (summary).

Safety cap (naive backend only): concurrency levels above 4 have not yet
been verified safe against the naive server's unlocked shared model object
(see prior concurrency-safety testing). This script refuses to run naive
levels above 4 until that is re-verified and this cap is deliberately
raised. This cap does not apply to the vllm backend, which is designed for
real concurrent serving — start with a conservative run manually before
going high, as a matter of practice rather than a code-enforced check.
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import httpx

from metrics.aggregate import summarize_level
from metrics.gpu_telemetry import GpuTelemetryCollector
from metrics.io import write_raw_csv, write_summary_json
from metrics.records import RequestRecord

GPU_TELEMETRY_INTERVAL_S = 0.2

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_VERIFIED_CONCURRENCY = 4

BACKEND_LABELS = {
    "naive": "naive_fastapi_baseline",
    "vllm": "vllm",
}
OUTPUT_PREFIXES = {
    "naive": "naive_baseline_",
    "vllm": "vllm_",
}

PROMPTS = [
    "What is the capital of France?",
    "What is the capital of Japan?",
    "What is the capital of Italy?",
    "What is the capital of Germany?",
    "What is the capital of Spain?",
    "What is the capital of Canada?",
]


def build_request(backend: str, prompt: str, max_new_tokens: int, model_name: str) -> tuple[str, dict]:
    """Return (path, json_body) for the given backend's request shape."""
    if backend == "naive":
        return "/generate", {"prompt": prompt, "max_new_tokens": max_new_tokens}
    return "/v1/chat/completions", {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
    }


def parse_token_counts(backend: str, body: dict) -> tuple[int, int]:
    """Return (input_tokens, output_tokens) from a successful response body."""
    if backend == "naive":
        return body["input_tokens"], body["output_tokens"]
    usage = body["usage"]
    return usage["prompt_tokens"], usage["completion_tokens"]


def send_one(
    backend: str,
    backend_label: str,
    base_url: str,
    request_index: int,
    concurrency_level: int,
    max_new_tokens: int,
    timeout_s: float,
    model_name: str,
) -> RequestRecord:
    prompt = PROMPTS[request_index % len(PROMPTS)]
    path, json_body = build_request(backend, prompt, max_new_tokens, model_name)
    start = time.perf_counter()
    try:
        r = httpx.post(
            f"{base_url}{path}",
            json=json_body,
            timeout=timeout_s,
        )
        latency = time.perf_counter() - start
        if r.status_code == 200:
            body = r.json()
            input_tokens, output_tokens = parse_token_counts(backend, body)
            return RequestRecord(
                backend=backend_label,
                concurrency_level=concurrency_level,
                request_index=request_index,
                prompt=prompt,
                success=True,
                status_code=r.status_code,
                latency_s=latency,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                error=None,
            )
        return RequestRecord(
            backend=backend_label,
            concurrency_level=concurrency_level,
            request_index=request_index,
            prompt=prompt,
            success=False,
            status_code=r.status_code,
            latency_s=latency,
            input_tokens=None,
            output_tokens=None,
            error=f"non-200 status: {r.text[:500]}",
        )
    except Exception as e:
        latency = time.perf_counter() - start
        return RequestRecord(
            backend=backend_label,
            concurrency_level=concurrency_level,
            request_index=request_index,
            prompt=prompt,
            success=False,
            status_code=None,
            latency_s=latency,
            input_tokens=None,
            output_tokens=None,
            error=repr(e),
        )


def run_level(
    backend: str,
    backend_label: str,
    base_url: str,
    concurrency_level: int,
    requests_per_level: int,
    max_new_tokens: int,
    timeout_s: float,
    model_name: str,
) -> tuple[list[RequestRecord], float]:
    print(f"\n=== Running concurrency level {concurrency_level} "
          f"({requests_per_level} requests) against {backend_label} ===")

    records: list[RequestRecord] = [None] * requests_per_level  # type: ignore
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency_level) as ex:
        futures = {
            ex.submit(
                send_one,
                backend,
                backend_label,
                base_url,
                i,
                concurrency_level,
                max_new_tokens,
                timeout_s,
                model_name,
            ): i
            for i in range(requests_per_level)
        }
        for fut in as_completed(futures):
            res = fut.result()
            records[res.request_index] = res
    wall_clock_s = time.perf_counter() - wall_start

    for r in records:
        status = "OK" if r.success else f"FAILED ({r.error})"
        print(
            f"  [{r.request_index}] {status} "
            f"latency={r.latency_s:.3f}s "
            f"in_tokens={r.input_tokens} out_tokens={r.output_tokens}"
        )

    return records, wall_clock_s


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["naive", "vllm"], default="naive")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--concurrency-levels",
        default="1,2,4",
        help="Comma-separated concurrency levels to test. Capped at "
        f"{MAX_VERIFIED_CONCURRENCY} for the naive backend; uncapped for vllm.",
    )
    parser.add_argument("--requests-per-level", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Value sent in the \"model\" field for vllm-mode requests "
        "(the OpenAI-compatible shape requires it). Ignored for naive.",
    )
    args = parser.parse_args()

    backend = args.backend
    backend_label = BACKEND_LABELS[backend]

    levels = [int(x.strip()) for x in args.concurrency_levels.split(",") if x.strip()]

    if backend == "naive":
        for level in levels:
            if level > MAX_VERIFIED_CONCURRENCY:
                print(
                    f"REFUSING TO RUN: concurrency level {level} exceeds the "
                    f"verified-safe cap of {MAX_VERIFIED_CONCURRENCY} for "
                    f"{backend_label} (unlocked shared model object, not yet "
                    "tested above this level). Aborting before sending any "
                    "requests.",
                    file=sys.stderr,
                )
                return 1

    base_url = f"http://{args.host}:{args.port}"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir)

    all_records: list[RequestRecord] = []
    all_summaries: list[dict] = []

    # One collector for the whole run: pynvml init (and its failure warning,
    # if any) happens once here, not once per concurrency level.
    gpu_telemetry = GpuTelemetryCollector(interval_s=GPU_TELEMETRY_INTERVAL_S)
    try:
        for level in levels:
            gpu_telemetry.start()
            records, wall_clock_s = run_level(
                backend,
                backend_label,
                base_url,
                level,
                args.requests_per_level,
                args.max_new_tokens,
                args.timeout_s,
                args.model_name,
            )
            gpu_stats = gpu_telemetry.stop()

            all_records.extend(records)
            summary = summarize_level(
                records, backend_label, level, wall_clock_s, gpu_telemetry=gpu_stats
            )
            all_summaries.append(summary)
            print(f"  -> throughput: {summary['throughput_req_per_s']:.3f} req/s, "
                  f"output tokens/s: {summary['output_tokens_per_s']:.2f}, "
                  f"p95 latency: {summary['latency_s_p95']}")
            if gpu_stats["gpu_telemetry_available"] and gpu_stats["gpu_telemetry_sample_count"] > 0:
                print(
                    f"  -> GPU util % (min/mean/max): "
                    f"{gpu_stats['gpu_util_pct_min']:.1f}/"
                    f"{gpu_stats['gpu_util_pct_mean']:.1f}/"
                    f"{gpu_stats['gpu_util_pct_max']:.1f}, "
                    f"GPU mem MiB (min/mean/max): "
                    f"{gpu_stats['gpu_mem_used_mib_min']:.1f}/"
                    f"{gpu_stats['gpu_mem_used_mib_mean']:.1f}/"
                    f"{gpu_stats['gpu_mem_used_mib_max']:.1f} "
                    f"({gpu_stats['gpu_telemetry_sample_count']} samples)"
                )
    finally:
        gpu_telemetry.shutdown()

    prefix = OUTPUT_PREFIXES[backend]
    raw_csv_path = output_dir / f"{prefix}{timestamp}_raw.csv"
    summary_json_path = output_dir / f"{prefix}{timestamp}_summary.json"

    write_raw_csv(all_records, raw_csv_path)
    write_summary_json(
        {
            "backend": backend_label,
            "timestamp_utc": timestamp,
            "concurrency_levels_tested": levels,
            "requests_per_level": args.requests_per_level,
            "max_new_tokens": args.max_new_tokens,
            "summaries": all_summaries,
        },
        summary_json_path,
    )

    print(f"\nRaw per-request records written to: {raw_csv_path}")
    print(f"Summary written to: {summary_json_path}")

    any_failures = any(not r.success for r in all_records)
    return 1 if any_failures else 0


if __name__ == "__main__":
    sys.exit(main())
