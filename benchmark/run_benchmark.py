"""Benchmark harness for the naive synchronous FastAPI baseline server
(src/server/main.py). Not vLLM.

Sends a configurable number of requests to a running /generate instance
at one or more concurrency levels, and records per-request latency,
throughput, and token counts to results/ as CSV (raw) and JSON (summary).

Safety cap: concurrency levels above 4 have not yet been verified safe
against this server's unlocked shared model object (see prior
concurrency-safety testing). This script refuses to run levels above 4
until that is re-verified and this cap is deliberately raised.
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

BACKEND_LABEL = "naive_fastapi_baseline"
MAX_VERIFIED_CONCURRENCY = 4

PROMPTS = [
    "What is the capital of France?",
    "What is the capital of Japan?",
    "What is the capital of Italy?",
    "What is the capital of Germany?",
    "What is the capital of Spain?",
    "What is the capital of Canada?",
]


def send_one(
    base_url: str,
    request_index: int,
    concurrency_level: int,
    max_new_tokens: int,
    timeout_s: float,
) -> RequestRecord:
    prompt = PROMPTS[request_index % len(PROMPTS)]
    start = time.perf_counter()
    try:
        r = httpx.post(
            f"{base_url}/generate",
            json={"prompt": prompt, "max_new_tokens": max_new_tokens},
            timeout=timeout_s,
        )
        latency = time.perf_counter() - start
        if r.status_code == 200:
            body = r.json()
            return RequestRecord(
                backend=BACKEND_LABEL,
                concurrency_level=concurrency_level,
                request_index=request_index,
                prompt=prompt,
                success=True,
                status_code=r.status_code,
                latency_s=latency,
                input_tokens=body["input_tokens"],
                output_tokens=body["output_tokens"],
                error=None,
            )
        return RequestRecord(
            backend=BACKEND_LABEL,
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
            backend=BACKEND_LABEL,
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
    base_url: str,
    concurrency_level: int,
    requests_per_level: int,
    max_new_tokens: int,
    timeout_s: float,
) -> tuple[list[RequestRecord], float]:
    print(f"\n=== Running concurrency level {concurrency_level} "
          f"({requests_per_level} requests) against {BACKEND_LABEL} ===")

    records: list[RequestRecord] = [None] * requests_per_level  # type: ignore
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency_level) as ex:
        futures = {
            ex.submit(
                send_one, base_url, i, concurrency_level, max_new_tokens, timeout_s
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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--concurrency-levels",
        default="1,2,4",
        help="Comma-separated concurrency levels to test. Capped at "
        f"{MAX_VERIFIED_CONCURRENCY}.",
    )
    parser.add_argument("--requests-per-level", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    levels = [int(x.strip()) for x in args.concurrency_levels.split(",") if x.strip()]

    for level in levels:
        if level > MAX_VERIFIED_CONCURRENCY:
            print(
                f"REFUSING TO RUN: concurrency level {level} exceeds the "
                f"verified-safe cap of {MAX_VERIFIED_CONCURRENCY} for "
                f"{BACKEND_LABEL} (unlocked shared model object, not yet "
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
                base_url, level, args.requests_per_level, args.max_new_tokens, args.timeout_s
            )
            gpu_stats = gpu_telemetry.stop()

            all_records.extend(records)
            summary = summarize_level(
                records, BACKEND_LABEL, level, wall_clock_s, gpu_telemetry=gpu_stats
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

    raw_csv_path = output_dir / f"naive_baseline_{timestamp}_raw.csv"
    summary_json_path = output_dir / f"naive_baseline_{timestamp}_summary.json"

    write_raw_csv(all_records, raw_csv_path)
    write_summary_json(
        {
            "backend": BACKEND_LABEL,
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
