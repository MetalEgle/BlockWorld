import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from engine.diagnostics import PerformanceMonitor
from engine.graphics.hud import Hud


class PerformanceMonitorTests(unittest.TestCase):
    @patch("engine.diagnostics.psutil.virtual_memory")
    @patch("engine.diagnostics.psutil.cpu_percent", return_value=32.5)
    @patch("engine.diagnostics.psutil.Process")
    def test_update_samples_cpu_memory_and_frame_rate(
        self,
        process_class: Mock,
        _cpu_percent: Mock,
        virtual_memory: Mock,
    ) -> None:
        process = process_class.return_value
        process.cpu_percent.return_value = 12.5
        process.memory_info.return_value = SimpleNamespace(rss=10 * 1024**2)
        virtual_memory.return_value = SimpleNamespace(
            used=8 * 1024**3,
            total=16 * 1024**3,
        )
        monitor = PerformanceMonitor(sample_interval=0.5)

        monitor.update(0.25)
        stats = monitor.update(0.25)

        self.assertEqual(stats.frame_count, 2)
        self.assertEqual(stats.fps, 4.0)
        self.assertEqual(stats.frame_time_ms, 250.0)
        self.assertEqual(stats.process_cpu_percent, 12.5)
        self.assertEqual(stats.system_cpu_percent, 32.5)
        self.assertEqual(stats.process_memory_mb, 10.0)
        self.assertEqual(stats.system_memory_used_gb, 8.0)
        self.assertEqual(stats.system_memory_total_gb, 16.0)
        self.assertEqual(_cpu_percent.call_count, 2)
        process.cpu_percent.assert_has_calls(
            [call(interval=None), call(interval=None)]
        )
        self.assertEqual(virtual_memory.call_count, 2)

    def test_debug_overlay_draws_all_performance_fields(self) -> None:
        monitor = PerformanceMonitor()
        stats = monitor.update(1 / 60)

        vertices = Hud.build_debug_overlay_vertices(1280, 720, stats, block_count=100)

        self.assertEqual(vertices.ndim, 2)
        self.assertEqual(vertices.shape[1], 5)
        self.assertGreater(len(vertices), 0)
        self.assertEqual(len(vertices) % 3, 0)
        self.assertEqual(len(set(map(tuple, vertices[:, 2:]))), 2)

    def test_debug_overlay_returns_no_vertices_for_invalid_viewport(self) -> None:
        stats = PerformanceMonitor().update(1 / 60)

        vertices = Hud.build_debug_overlay_vertices(0, 720, stats, block_count=100)

        self.assertEqual(vertices.shape, (0, 5))

    def test_rejects_invalid_sample_intervals_and_frame_times(self) -> None:
        with self.assertRaises(ValueError):
            PerformanceMonitor(sample_interval=0.0)

        monitor = PerformanceMonitor()
        with self.assertRaises(ValueError):
            monitor.update(-0.1)
