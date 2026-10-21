"""Measure GPU energy by sampling NVML power draw in a background thread.

Only GPU board power is counted (CPU, RAM and cooling are not), so the
numbers are a lower bound on the true energy use; the simulation applies a
PUE factor to training energy to account for overheads.

With `checkpoint=<path>` the running totals are written there every few
minutes (marked "partial"), and a later run with the same checkpoint adds to
them, so energy is still accounted for when a run is interrupted (reboot)
and resumed.
"""

import json
import threading
import time
from pathlib import Path

from . import log, write_json


class EnergyMeter:
    def __init__(self, interval=1.0, device_index=None, checkpoint=None, flush_every=120):
        self.interval = interval
        self.device_index = device_index
        self.joules = 0.0
        self.samples = 0
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.flush_every = flush_every
        self.prev_wall, self.prev_joules, self.sessions = 0.0, 0.0, 1
        if self.checkpoint and self.checkpoint.exists():
            try:
                old = json.loads(self.checkpoint.read_text())
                if old.get("partial"):
                    self.prev_wall = float(old.get("wall_s") or 0)
                    self.prev_joules = float(old.get("gpu_kwh") or 0) * 3.6e6
                    self.sessions = int(old.get("sessions", 1)) + 1
                    log.info("continuing energy log from an interrupted run (%.2f h so far)", self.prev_wall / 3600)
            except Exception:
                pass
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
        last = last_flush = time.time()
        while not self._stop.wait(self.interval):
            now = time.time()
            if self._nvml:
                try:
                    self.joules += self._power_w() * (now - last)
                    self.samples += 1
                except Exception:
                    pass
            last = now
            if self.checkpoint and now - last_flush >= self.flush_every:
                last_flush = now
                try:
                    write_json({**self.summary(), "partial": True}, self.checkpoint)
                except Exception:
                    pass

    def __enter__(self):
        self.t0 = time.time()
        self.wall_s = None
        if self._nvml or self.checkpoint:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self.wall_s = time.time() - self.t0
        if self._nvml or self.checkpoint:
            self._stop.set()
            self._thread.join()
        return False

    def summary(self, **extra):
        wall = self.prev_wall + (self.wall_s if self.wall_s is not None else time.time() - self.t0)
        joules = self.prev_joules + self.joules
        kwh = joules / 3.6e6 if self._nvml else None
        return {
            "wall_s": wall,
            "gpu_hours": wall / 3600 * max(len(self._handles), 1) if self._nvml else 0.0,
            "gpu_kwh": kwh,
            "mean_gpu_watts": joules / wall if self._nvml and wall else None,
            "sessions": self.sessions,
            **extra,
        }

    def save(self, path, **extra):
        s = self.summary(**extra)
        write_json(s, path)
        log.info("energy: %.2f h wall, %s kWh GPU -> %s", s["wall_s"] / 3600,
                 f"{s['gpu_kwh']:.3f}" if s["gpu_kwh"] is not None else "n/a", path)
        return s
