"""Background GPU telemetry sampling via pynvml.

Polls GPU utilization (%) and memory used (MiB) on a fixed interval from a
background thread, for the duration of one concurrency level's request
batch. Real captured samples only — nothing here estimates or interpolates
between polls.

If pynvml cannot be initialized (no NVIDIA GPU, missing/broken driver,
package not installed), this degrades gracefully: a clear warning is
printed once, `available` is False, and every summary this collector
produces reports `gpu_telemetry_available: False` with `None` stats rather
than raising. Callers must not let missing telemetry abort the benchmark.
"""

import statistics
import sys
import threading


class GpuTelemetryCollector:
    def __init__(self, device_index: int = 0, interval_s: float = 0.2):
        self.device_index = device_index
        self.interval_s = interval_s
        self.available = False

        self._pynvml = None
        self._handle = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._samples: list[tuple[float, float]] = []  # (util_pct, mem_used_mib)
        self._samples_lock = threading.Lock()

        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._pynvml = pynvml
            self.available = True
        except Exception as e:
            print(
                f"WARNING: GPU telemetry unavailable ({e!r}). Skipping GPU "
                "utilization/memory capture for this benchmark run; "
                "latency/throughput measurement is unaffected.",
                file=sys.stderr,
            )
            self.available = False

    def start(self) -> None:
        """Begin sampling on a background thread. No-op if unavailable."""
        if not self.available:
            return
        self._samples = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _poll_loop(self) -> None:
        pynvml = self._pynvml
        while not self._stop_event.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                with self._samples_lock:
                    self._samples.append((float(util.gpu), mem.used / (1024 * 1024)))
            except Exception:
                pass
            self._stop_event.wait(self.interval_s)

    def stop(self) -> dict:
        """Stop sampling and return a summary dict for this level's batch.

        Always returns the same key set, whether or not telemetry was
        available or any samples were captured, so downstream code can
        merge it into a summary unconditionally.
        """
        empty = {
            "gpu_telemetry_available": self.available,
            "gpu_telemetry_sample_count": 0,
            "gpu_util_pct_min": None,
            "gpu_util_pct_mean": None,
            "gpu_util_pct_max": None,
            "gpu_mem_used_mib_min": None,
            "gpu_mem_used_mib_mean": None,
            "gpu_mem_used_mib_max": None,
        }

        if not self.available:
            return empty

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 5 + 1.0)
            self._thread = None

        with self._samples_lock:
            samples = list(self._samples)

        if not samples:
            return empty

        utils = [s[0] for s in samples]
        mems = [s[1] for s in samples]
        return {
            "gpu_telemetry_available": True,
            "gpu_telemetry_sample_count": len(samples),
            "gpu_util_pct_min": min(utils),
            "gpu_util_pct_mean": statistics.fmean(utils),
            "gpu_util_pct_max": max(utils),
            "gpu_mem_used_mib_min": min(mems),
            "gpu_mem_used_mib_mean": statistics.fmean(mems),
            "gpu_mem_used_mib_max": max(mems),
        }

    def shutdown(self) -> None:
        """Release the pynvml handle. Safe to call even if unavailable."""
        if self.available and self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
