from dataclasses import dataclass
from math import isfinite

import psutil


@dataclass(frozen=True)
class PerformanceStats:
    frame_count: int
    fps: float
    frame_time_ms: float
    process_cpu_percent: float
    system_cpu_percent: float
    process_memory_mb: float
    system_memory_used_gb: float
    system_memory_total_gb: float


class PerformanceMonitor:
    def __init__(self, sample_interval: float = 1.0) -> None:
        if not isfinite(sample_interval) or sample_interval <= 0.0:
            raise ValueError("Интервал измерения должен быть больше нуля.")

        self._process = psutil.Process()
        self._process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        self._process_memory_mb = self._process.memory_info().rss / (1024 * 1024)
        system_memory = psutil.virtual_memory()
        self._system_memory_used_gb = system_memory.used / (1024**3)
        self._system_memory_total_gb = system_memory.total / (1024**3)
        self._sample_interval = sample_interval
        self._sample_elapsed = 0.0
        self._sample_frames = 0
        self._frame_count = 0
        self._fps = 0.0
        self._frame_time_ms = 0.0
        self._process_cpu_percent = 0.0
        self._system_cpu_percent = 0.0

    def update(self, frame_delta_seconds: float) -> PerformanceStats:
        if not isfinite(frame_delta_seconds) or frame_delta_seconds < 0.0:
            raise ValueError("Время кадра не может быть отрицательным.")

        self._frame_count += 1
        self._sample_frames += 1
        self._sample_elapsed += frame_delta_seconds
        self._frame_time_ms = frame_delta_seconds * 1000.0

        if self._sample_elapsed >= self._sample_interval:
            self._fps = self._sample_frames / self._sample_elapsed
            self._sample_frames = 0
            self._sample_elapsed = 0.0
            self._process_cpu_percent = self._process.cpu_percent(interval=None)
            self._system_cpu_percent = psutil.cpu_percent(interval=None)

            process_memory = self._process.memory_info().rss
            system_memory = psutil.virtual_memory()
            self._process_memory_mb = process_memory / (1024 * 1024)
            self._system_memory_used_gb = system_memory.used / (1024**3)
            self._system_memory_total_gb = system_memory.total / (1024**3)

        return PerformanceStats(
            self._frame_count,
            self._fps,
            self._frame_time_ms,
            self._process_cpu_percent,
            self._system_cpu_percent,
            self._process_memory_mb,
            self._system_memory_used_gb,
            self._system_memory_total_gb,
        )
