from typing import Dict, Any, Optional
from datetime import datetime
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
import json
from .monitor import SystemMetrics

class MetricsCollector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics: Dict[str, Any] = {
            'system_metrics': [],
            'error_count': 0,
            'face_detections': 0,
            'audio_processed': 0,
            'response_times': []
        }
        self._is_running = False
        self._cleanup_task: Optional[asyncio.Task] = None

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()

    async def start(self):
        """Start metrics collection"""
        async with self.initialize_context() as _:
            pass

    @asynccontextmanager
    async def initialize_context(self):
        """Context manager for metrics collector initialization"""
        try:
            self._is_running = True
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            yield self
        finally:
            await self.stop()

    def record_system_metrics(self, metrics: SystemMetrics):
        """Record system metrics"""
        if not self._is_running:
            return
        self.metrics['system_metrics'].append({
            'memory_used': metrics.memory_used,
            'cpu_used': metrics.cpu_used,
            'disk_used': metrics.disk_used,
            'temperature': metrics.temperature,
            'timestamp': metrics.timestamp
        })

    def increment(self, metric_name: str):
        """Increment counter metrics"""
        if metric_name in self.metrics:
            self.metrics[metric_name] += 1

    @asynccontextmanager
    async def measure_time(self, operation: str):
        """Measure operation execution time"""
        start_time = datetime.now()
        try:
            yield
        finally:
            duration = (datetime.now() - start_time).total_seconds()
            self.metrics['response_times'].append({
                'operation': operation,
                'duration': duration,
                'timestamp': start_time
            })

    async def _periodic_cleanup(self):
        """Cleanup old metrics periodically"""
        while self._is_running:
            try:
                # Keep only last hour of system metrics
                current_time = datetime.now()
                self.metrics['system_metrics'] = [
                    m for m in self.metrics['system_metrics']
                    if (current_time - m['timestamp']).total_seconds() < 3600
                ]
                await asyncio.sleep(300)  # Cleanup every 5 minutes
            except Exception as e:
                self.logger.error(f"Metrics cleanup error: {e}")

    async def stop(self):
        """Stop metrics collection"""
        self._is_running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def save_metrics(self, path: Path = None) -> None:
        """Save metrics to file"""
        if path is None:
            path = Path("metrics.json")
            
        try:
            with open(path, "w") as f:
                json.dump(self.metrics, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")