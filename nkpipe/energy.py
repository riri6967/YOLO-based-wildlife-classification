"""Measure GPU energy by sampling NVML power draw in a background thread.

Only GPU board power is counted (CPU, RAM and cooling are not), so the
numbers are a lower bound on the true energy use; the simulation applies a
PUE factor to training energy to account for overheads.
"""

import threading
import time

from . import log, write_json


class EnergyMeter:
    def __init__(self, interval=1.0, device_index=None):
        self.interval = interval
        self.device_index = device_index
        self.joules = 0.0
        self.samples = 0
        self._stop = threading.Event()
        self._handles = []
        try:
            import pynvml
            pynvml.nvmlInit()
            n = pynvml.nvmlDeviceGetCount()
            idx = range(n) if device_index is None else [int(device_index)]
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in idx]
            self._nvml = pynvml
        except Exception as e:  # no NVIDIA GPU / driver: wall time only
            log.warning("NVML unavailable (%s); energy will not be measured", e)
            self._nvml = None

    def _power_w(self):
        return sum(self._nvml.nvmlDeviceGetPowerUsage(h) for h in self._handles) / 1000.0

    def _run(self):
        last = time.time()
        while not self._stop.wait(self.interval):
            now = time.time()
            try:
                self.joules += self._power_w() * (now - last)
                self.samples += 1
            except Exception:
                pass
            last = now

    def __enter__(self):
        self.t0 = time.time()
        if self._nvml:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self.wall_s = time.time() - self.t0
        if self._nvml:
            self._stop.set()
            self._thread.join()
        return False

    def summary(self, **extra):
        kwh = self.joules / 3.6e6 if self._nvml else None
        return {
            "wall_s": self.wall_s,
            "gpu_hours": self.wall_s / 3600 * max(len(self._handles), 1) if self._nvml else 0.0,
            "gpu_kwh": kwh,
            "mean_gpu_watts": self.joules / self.wall_s if self._nvml and self.wall_s else None,
            **extra,
        }

    def save(self, path, **extra):
        s = self.summary(**extra)
        write_json(s, path)
        log.info("energy: %.2f h wall, %s kWh GPU -> %s", s["wall_s"] / 3600,
                 f"{s['gpu_kwh']:.3f}" if s["gpu_kwh"] is not None else "n/a", path)
        return s
