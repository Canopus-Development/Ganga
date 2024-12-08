from contextlib import AsyncExitStack
import asyncio
from typing import Optional
from concurrent.futures import ProcessPoolExecutor
import signal
import sys
from config.settings import Settings
from core.vision import VisionSystem
from core.speech import SpeechSystem
from modules.emotion import EmotionProcessor
from modules.biometric import BiometricAuth
from modules.server import ServerManager
from utils.logger import setup_logging
from utils.monitor import SystemMonitor
from utils.cache import ModelCache
from utils.validators import InputValidator
from utils.metrics import MetricsCollector

class GangaFirmware:
    def __init__(self, mode: str = 'local'):
        self.settings = Settings()
        self.settings.create_directories()
        self.loop = asyncio.get_event_loop()
        
        # Configure logging first
        self.logger = setup_logging(
            level=self.settings.LOG_LEVEL,
            log_file=self.settings.DATA_DIR / "ganga.log"
        )
        
        # Initialize components with dependency injection
        self.metrics = MetricsCollector()
        self.monitor = SystemMonitor(self.metrics)
        self.validator = InputValidator()
        self.model_cache = ModelCache(self.settings)
        
        # Initialize core systems
        self.vision = VisionSystem(self.settings, self.model_cache)
        self.speech = SpeechSystem(self.settings, self.model_cache)
        self.emotion = EmotionProcessor(self.settings, self.model_cache)
        self.biometric = BiometricAuth(self.settings)
        
        # Initialize API server if in server mode
        self.server: Optional[ServerManager] = None
        if mode == 'server':
            self.server = ServerManager(self.settings)
        
        # Process pool for CPU-bound tasks
        self.executor = ProcessPoolExecutor(
            max_workers=self.settings.API_WORKERS
        )
        
        # Shutdown flag
        self._shutdown = asyncio.Event()
        
    async def initialize(self):
        """Initialize all systems with proper error handling and cleanup"""
        async with AsyncExitStack() as stack:
            try:
                # Register shutdown handlers
                for sig in (signal.SIGTERM, signal.SIGINT):
                    signal.signal(sig, self._signal_handler)
                
                # Start monitoring and metrics
                await stack.enter_async_context(self.monitor)
                await stack.enter_async_context(self.metrics)
                
                # Initialize core systems with retries
                systems = [self.vision, self.speech, self.emotion, self.biometric]
                for system in systems:
                    await self._init_with_retry(system)
                
                # Preload models
                await self.model_cache.preload_models()
                
                # Start server or local mode
                if self.server:
                    await self.server.start()
                else:
                    await self._run_local_mode()
                    
                # Wait for shutdown signal
                await self._shutdown.wait()
                
            except Exception as e:
                self.logger.critical(f"Critical initialization error: {e}", exc_info=True)
                await self._emergency_shutdown()
                sys.exit(1)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}")
        self._shutdown.set()
    
    async def _emergency_shutdown(self):
        """Handle graceful shutdown in case of critical errors"""
        self.logger.warning("Initiating emergency shutdown")
        tasks = []
        
        # Stop all async systems
        for system in [self.monitor, self.metrics, self.model_cache]:
            if hasattr(system, 'stop'):
                tasks.append(asyncio.create_task(system.stop()))
        
        # Wait for all shutdown tasks with timeout
        try:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=10)
        except asyncio.TimeoutError:
            self.logger.error("Shutdown timed out")
        
        # Shutdown process pool
        self.executor.shutdown(wait=False)

    async def _init_with_retry(self, system, max_retries=3):
        for attempt in range(max_retries):
            try:
                system.start()
                return
            except Exception as e:
                self.logger.error(f"Initialization attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    async def _run_local_mode(self):
        while True:
            try:
                async with self.metrics.measure_time("process_cycle"):
                    # Run synchronous camera capture in process pool
                    frame = await self.vision.get_frame()
                    
                    if not self.validator.validate_frame(frame):
                        continue
                    
                    # Process faces synchronously in process pool
                    faces = await self.vision.detect_faces(frame)
                    
                    for face in faces:
                        await self._process_face(face)
                        
                if not await self.monitor.check_health():
                    await self._handle_health_issues()
                    
            except Exception as e:
                self.logger.error(f"Error in local mode: {str(e)}")
                self.metrics.increment("error_count")
                
            finally:
                await asyncio.sleep(0.1)

    async def _process_face(self, face):
        if not self.validator.validate_face(face):
            return

        async with self.metrics.measure_time("face_processing"):
            # Run sync authentication in executor
            user = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.biometric.authenticate,
                face
            )
            
            if user:
                emotion = await self.emotion.process_batch([face])
                audio = await self._get_audio()
                if audio and emotion:
                    response = await self._generate_response(user, emotion[0], audio)
                    await self._speak_response(response)
                else:   
                    self.metrics.increment("error_count")
            else:
                self.metrics.increment("unauthorized_count")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else 'local'
    firmware = GangaFirmware(mode)
    asyncio.run(firmware.initialize())