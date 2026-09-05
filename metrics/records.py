"""Shared data shapes for benchmark measurements.

Kept backend-agnostic on purpose: the same RequestRecord shape is meant to
be reused for the future vLLM benchmark run, so that naive-baseline and
vLLM results can eventually be compared side by side. The `backend` field
is what keeps the two distinguishable in results/ — never overwrite or
merge raw files from different backends.
"""

from dataclasses import asdict, dataclass


@dataclass
class RequestRecord:
    backend: str
    concurrency_level: int
    request_index: int
    prompt: str
    success: bool
    status_code: int | None
    latency_s: float | None
    input_tokens: int | None
    output_tokens: int | None
    error: str | None

    def to_dict(self) -> dict:
        return asdict(self)
