"""Aggregation of raw RequestRecords into per-concurrency-level summaries.

All statistics here are computed directly from the RequestRecords passed
in — nothing here estimates, extrapolates, or fills in missing data.
"""

import statistics

from metrics.records import RequestRecord


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * pct
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1


def summarize_level(
    records: list[RequestRecord],
    backend: str,
    concurrency_level: int,
    wall_clock_s: float,
) -> dict:
    """Summarize all records for one (backend, concurrency_level) run.

    wall_clock_s is the measured wall-clock duration of sending and
    receiving the whole batch of requests at this concurrency level —
    used for throughput, since per-request latency alone does not
    capture overlap between concurrent requests.
    """
    successes = [r for r in records if r.success]
    failures = [r for r in records if not r.success]

    latencies = sorted(r.latency_s for r in successes if r.latency_s is not None)
    total_output_tokens = sum(
        r.output_tokens for r in successes if r.output_tokens is not None
    )
    total_input_tokens = sum(
        r.input_tokens for r in successes if r.input_tokens is not None
    )

    summary = {
        "backend": backend,
        "concurrency_level": concurrency_level,
        "total_requests": len(records),
        "successful_requests": len(successes),
        "failed_requests": len(failures),
        "wall_clock_s": wall_clock_s,
        "throughput_req_per_s": (
            len(successes) / wall_clock_s if wall_clock_s > 0 else 0.0
        ),
        "output_tokens_per_s": (
            total_output_tokens / wall_clock_s if wall_clock_s > 0 else 0.0
        ),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "latency_s_mean": statistics.fmean(latencies) if latencies else None,
        "latency_s_median": statistics.median(latencies) if latencies else None,
        "latency_s_min": min(latencies) if latencies else None,
        "latency_s_max": max(latencies) if latencies else None,
        "latency_s_p95": _percentile(latencies, 0.95) if latencies else None,
        "errors": [r.error for r in failures],
    }
    return summary
