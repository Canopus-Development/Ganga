from typing import Optional, Dict, Any, Tuple
import numpy as np
from pathlib import Path
import asyncio
from deepface import DeepFace
import face_recognition
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

class BiometricAuth:
    def __init__(self, settings: "Settings"):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.known_faces: Dict[str, Any] = {}
        self.face_db_path = Path(settings.KNOWN_FACES_DIR)
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._is_running = False

    async def start(self):
        """Start the biometric system"""
        async with self.initialize_context() as _:
            pass

    @asynccontextmanager
    async def initialize_context(self):
        """Context manager for biometric system initialization"""
        try:
            await self.initialize()
            yield self
        finally:
            await self.shutdown()

    async def initialize(self) -> None:
        """Initialize biometric system with async loading"""
        try:
            self.face_db_path.mkdir(parents=True, exist_ok=True)
            await self._load_known_faces()
            self._is_running = True
        except Exception as e:
            self.logger.error(f"Biometric initialization failed: {e}")
            raise

    async def _load_known_faces(self) -> None:
        """Load known face embeddings asynchronously"""
        load_tasks = []
        for face_file in self.face_db_path.glob("*.jpg"):
            load_tasks.append(
                asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    self._load_single_face,
                    face_file
                )
            )
        results = await asyncio.gather(*load_tasks, return_exceptions=True)
        
        # Process results
        for name, embedding in results:
            if isinstance(embedding, Exception):
                self.logger.error(f"Failed to load face {name}: {embedding}")
            else:
                self.known_faces[name] = embedding

    def _load_single_face(self, face_path: Path) -> tuple[str, Any]:
        """Load a single face embedding"""
        name = face_path.stem
        try:
            embedding = DeepFace.represent(
                str(face_path),
                model_name=self.settings.FACE_RECOGNITION_MODEL,
                detector_backend=self.settings.FACE_DETECTOR_BACKEND
            )
            return name, embedding[0]
        except Exception as e:
            return name, e

    def authenticate(self, face_image):
        try:
            # Get embedding for current face
            current_embedding = DeepFace.represent(
                face_image,
                model_name=self.model_name,
                detector_backend=self.detection_model
            )[0]
            
            # Compare with known faces
            for name, known_embedding in self.known_faces.items():
                distance = self._cosine_distance(
                    current_embedding["embedding"],
                    known_embedding["embedding"]
                )
                if distance < self.settings.FACE_RECOGNITION_TOLERANCE:
                    return name
                    
            return None
            
        except Exception as e:
            self.logger.error(f"Authentication error: {str(e)}")
            return None

    def _cosine_distance(self, a, b):
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    async def register_face(self, name: str, face_image: np.ndarray) -> bool:
        """Register a new face with async processing"""
        if not self._is_running:
            raise RuntimeError("Biometric system not initialized")
            
        try:
            face_path = self.face_db_path / f"{name}.jpg"
            
            # Process in thread pool
            embedding = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self._process_new_face,
                face_image,
                face_path
            )
            
            if embedding:
                self.known_faces[name] = embedding
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Face registration error: {e}")
            return False

    async def shutdown(self) -> None:
        """Cleanup resources"""
        self._is_running = False
        self._executor.shutdown(wait=False)