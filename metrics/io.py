"""Writing raw records and summaries to results/ as CSV/JSON."""

import csv
import json
from pathlib import Path

from metrics.records import RequestRecord

RAW_CSV_FIELDS = [
    "backend",
    "concurrency_level",
    "request_index",
    "prompt",
    "success",
    "status_code",
    "latency_s",
    "input_tokens",
    "output_tokens",
    "error",
]


def write_raw_csv(records: list[RequestRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_CSV_FIELDS)
        writer.writeheader()
        for r in records:
            writer.writerow(r.to_dict())


def write_summary_json(summary_payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
