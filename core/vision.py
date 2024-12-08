from typing import List, Optional, Tuple, Any
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import YolosImageProcessor, YolosForObjectDetection
from contextlib import asynccontextmanager
import logging
from pathlib import Path

class VisionSystem:
    def __init__(self, settings: "Settings", model_cache: "ModelCache"):
        self.settings = settings
        self.model_cache = model_cache
        self.logger = logging.getLogger(__name__)
        self.camera: Optional[cv2.VideoCapture] = None
        self.device = torch.device(settings.MODEL_DEVICE)
        self.model: Optional[YolosForObjectDetection] = None
        self.processor: Optional[YolosImageProcessor] = None
        self._is_running = False

    async def start(self):
        """Start the vision system"""
        async with self.initialize() as _:
            pass

    @asynccontextmanager
    async def initialize(self):
        """Context manager for vision system initialization"""
        try:
            await self._setup_camera()
            await self._load_models()
            self._is_running = True
            yield self
        finally:
            await self.shutdown()

    async def _setup_camera(self) -> None:
        """Initialize camera with retries"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.camera = cv2.VideoCapture(self.settings.CAMERA_INDEX)
                if self.camera.isOpened():
                    # Set camera properties for optimal performance
                    self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self.camera.set(cv2.CAP_PROP_FPS, 30)
                    return
            except Exception as e:
                self.logger.error(f"Camera initialization attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise RuntimeError("Failed to initialize camera")
                await asyncio.sleep(1)

    async def _load_models(self) -> None:
        """Load and initialize ML models"""
        try:
            # Try to get models from cache first
            self.model = self.model_cache.get_model('yolos')
            if self.model is None:
                self.model = YolosForObjectDetection.from_pretrained(
                    self.settings.FACE_MODEL_ID
                ).to(self.device)
                self.model.eval()
                
            self.processor = YolosImageProcessor.from_pretrained(
                self.settings.FACE_MODEL_ID
            )
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            raise

    def get_frame(self) -> Optional[np.ndarray]:
        """Synchronous frame capture"""
        if not self._is_running:
            raise RuntimeError("Vision system not initialized")
            
        try:
            if not self.camera.isOpened():
                self._setup_camera_sync()
                
            ret, frame = self.camera.read()
            if not ret:
                raise RuntimeError("Failed to capture frame")
                
            return frame
        except Exception as e:
            self.logger.error(f"Frame capture error: {e}")
            return None

    def _setup_camera_sync(self) -> None:
        """Synchronous camera setup"""
        self.camera = cv2.VideoCapture(self.settings.CAMERA_INDEX)
        if self.camera.isOpened():
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
        else:
            raise RuntimeError("Failed to initialize camera")

    async def get_frame(self) -> Optional[np.ndarray]:
        """Get frame from camera with error handling"""
        if not self._is_running:
            raise RuntimeError("Vision system not initialized")
            
        try:
            if not self.camera.isOpened():
                await self._setup_camera()
                
            ret, frame = self.camera.read()
            if not ret:
                raise RuntimeError("Failed to capture frame")
                
            return frame
        except Exception as e:
            self.logger.error(f"Frame capture error: {e}")
            return None

    async def detect_faces(self, frame: np.ndarray) -> List[np.ndarray]:
        """Detect faces in frame with batched processing"""
        if not self._is_running:
            raise RuntimeError("Vision system not initialized")
            
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            
            # Process image
            with torch.no_grad():
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
            
            # Process detections
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs, 
                threshold=self.settings.FACE_DETECTION_CONFIDENCE,
                target_sizes=target_sizes
            )[0]
            
            # Extract face regions
            faces = []
            for box in results["boxes"]:
                box = [int(i) for i in box.tolist()]
                face_region = frame[box[1]:box[3], box[0]:box[2]]
                if face_region.size > 0:
                    faces.append(face_region)
                    
            return faces
            
        except Exception as e:
            self.logger.error(f"Face detection error: {e}")
            return []

    async def shutdown(self) -> None:
        """Cleanup resources"""
        self._is_running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self):
        """Ensure cleanup on deletion"""
        if self.camera is not None:
            self.camera.release()