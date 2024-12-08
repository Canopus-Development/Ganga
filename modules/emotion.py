from typing import Optional, Dict, List, Any
import torch
import numpy as np
from transformers import pipeline
from PIL import Image
import logging
from contextlib import asynccontextmanager

class EmotionProcessor:
    def __init__(self, settings: "Settings", model_cache: "ModelCache"):
        self.settings = settings
        self.model_cache = model_cache
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(settings.MODEL_DEVICE)
        self.emotion_classifier = None
        self._is_running = False

    async def start(self):
        """Start the emotion processor"""
        async with self.initialize() as _:
            pass

    @asynccontextmanager
    async def initialize(self):
        """Initialize emotion processor"""
        try:
            await self._load_model()
            self._is_running = True
            yield self
        finally:
            await self.shutdown()

    async def _load_model(self) -> None:
        """Load emotion detection model"""
        try:
            # Try to get from cache first
            self.emotion_classifier = self.model_cache.get_model('emotion')
            
            if self.emotion_classifier is None:
                self.emotion_classifier = pipeline(
                    "image-classification",
                    model=self.settings.EMOTION_MODEL_ID,
                    device=self.device
                )
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            raise

    async def process_batch(self, face_images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Process multiple faces in batch"""
        if not self._is_running:
            raise RuntimeError("Emotion processor not initialized")
            
        try:
            # Convert to PIL images
            pil_images = [Image.fromarray(img) for img in face_images]
            
            # Process batch
            with torch.no_grad():
                results = self.emotion_classifier(pil_images, batch_size=self.settings.MODEL_BATCH_SIZE)
            
            return [
                {
                    'emotion': result['label'],
                    'confidence': float(result['score'])
                }
                for result in results
            ]
        except Exception as e:
            self.logger.error(f"Batch emotion processing error: {e}")
            return []

    async def shutdown(self) -> None:
        """Cleanup resources"""
        self._is_running = False
        del self.emotion_classifier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def process_emotion(self, face_image):
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(face_image)
        result = self.emotion_classifier(pil_image)[0]
        return {
            'emotion': result['label'],
            'confidence': float(result['score'])
        }