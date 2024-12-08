from typing import Dict, Any, Optional
import psutil
import asyncio
from datetime import datetime
import gc
import logging
from dataclasses import dataclass
from contextlib import asynccontextmanager

@dataclass
class SystemMetrics:
    memory_used: float
    cpu_used: float
    disk_used: float
    temperature: float
    timestamp: datetime

class SystemMonitor:
    def __init__(self, metrics_collector: Optional["MetricsCollector"] = None):
        self.logger = logging.getLogger(__name__)
        self.metrics_collector = metrics_collector
        self.thresholds = {
            'memory_percent': 90.0,
            'cpu_percent': 80.0,
            'disk_percent': 90.0,
            'temperature': 80.0
        }
        self._is_running = False
        self._monitor_task: Optional[asyncio.Task] = None

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()

    async def start(self) -> None:
        """Start monitoring with proper task management"""
        self._is_running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        
    async def check_health(self) -> bool:
        """Check system health with metrics collection"""
        try:
            metrics = await self._collect_metrics()
            
            # Record metrics if collector exists
            if self.metrics_collector:
                self.metrics_collector.record_system_metrics(metrics)
            
            # Check against thresholds
            health_checks = {
                'memory_percent': metrics.memory_used,
                'cpu_percent': metrics.cpu_used,
                'disk_percent': metrics.disk_used,
                'temperature': metrics.temperature
            }
            
            return all(
                value < self.thresholds[key]
                for key, value in health_checks.items()
            )
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

    async def _monitor_loop(self):
        while self._is_running:
            await self.check_health()
            await asyncio.sleep(5)

    def _get_temperature(self):
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                return max(temp.current for temp in temps['coretemp'])
            return 0.0
        except Exception:
            return 0.0

    async def _collect_metrics(self) -> SystemMetrics:
        """Collect system metrics"""
        try:
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=0.1)
            disk = psutil.disk_usage('/')
            
            return SystemMetrics(
                memory_used=memory.percent,
                cpu_used=cpu,
                disk_used=disk.percent,
                temperature=self._get_temperature(),
                timestamp=datetime.now()
            )
        except Exception as e:
            self.logger.error(f"Metrics collection failed: {e}")
            raise

    async def stop(self) -> None:
        """Graceful shutdown of monitoring"""
        self._is_running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

    async def cleanup_memory(self) -> None:
        """Clean up unused memory"""
        try:
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Force python garbage collection
            for _ in range(2):
                gc.collect()
                
        except Exception as e:
            self.logger.error(f"Memory cleanup failed: {e}")