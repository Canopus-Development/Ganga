from typing import Optional, List, Dict, Any, Union
import torch
import numpy as np
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    SpeechT5Processor, 
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan
)
import soundfile as sf
import numpy as np
from datasets import load_dataset
import speech_recognition as sr
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import pvporcupine
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage

class SpeechSystem:
    def __init__(self, settings: "Settings", model_cache: "ModelCache"):
        self.settings = settings
        self.model_cache = model_cache
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(settings.MODEL_DEVICE)
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._is_running = False
        self._setup_components()

    async def start(self):
        """Start the speech system"""
        async with self.initialize() as _:
            pass

    @asynccontextmanager
    async def initialize(self):
        """Context manager for speech system initialization"""
        try:
            await self._load_models()
            self._is_running = True
            yield self
        finally:
            await self.shutdown()

    def _setup_components(self) -> None:
        """Initialize speech components"""
        self.components = {
            'whisper': WhisperComponent(self.settings, self.model_cache),
            'tts': TextToSpeechComponent(self.settings, self.model_cache),
            'azure': AzureAIComponent(self.settings)
        }

    async def _load_models(self) -> None:
        """Load all required models"""
        try:
            load_tasks = [
                component.initialize() 
                for component in self.components.values()
            ]
            await asyncio.gather(*load_tasks)
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            raise

    async def process_audio(self, audio_data: np.ndarray) -> Optional[str]:
        """Process audio with error handling and retries"""
        if not self._is_running:
            raise RuntimeError("Speech system not initialized")
            
        try:
            # Run audio processing in thread pool
            text = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self.components['whisper'].transcribe,
                audio_data
            )
            
            if text:
                # Analyze sentiment
                sentiment = await self.components['azure'].analyze_sentiment(text)
                return await self._process_with_azure_ai(text, sentiment)
                
            return None
            
        except Exception as e:
            self.logger.error(f"Audio processing error: {e}")
            return None

    async def generate_speech(self, text: str) -> Optional[np.ndarray]:
        """Generate speech from text"""
        if not self._is_running:
            raise RuntimeError("Speech system not initialized")
            
        try:
            return await self.components['tts'].generate(text)
        except Exception as e:
            self.logger.error(f"Speech generation error: {e}")
            return None

    async def shutdown(self) -> None:
        """Cleanup resources"""
        self._is_running = False
        self._executor.shutdown(wait=False)
        
        shutdown_tasks = [
            component.shutdown() 
            for component in self.components.values()
        ]
        await asyncio.gather(*shutdown_tasks)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class WhisperComponent:
    def __init__(self, settings: "Settings", model_cache: "ModelCache"):
        self.settings = settings
        self.model_cache = model_cache
        self.device = torch.device(settings.MODEL_DEVICE)
        self.processor = None
        self.model = None

    async def initialize(self) -> None:
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
        self.model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-medium"
        ).to(self.device)

    def transcribe(self, audio_data: np.ndarray) -> str:
        input_features = self.processor(
            audio_data, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(self.device)
        
        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        return transcription

    async def shutdown(self) -> None:
        del self.processor
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class TextToSpeechComponent:
    def __init__(self, settings: "Settings", model_cache: "ModelCache"):
        self.settings = settings
        self.model_cache = model_cache
        self.device = torch.device(settings.MODEL_DEVICE)
        self.processor = None
        self.model = None
        self.vocoder = None
        self.speaker_embeddings = None

    async def initialize(self) -> None:
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
        
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(self.device)

    async def generate(self, text: str) -> np.ndarray:
        inputs = self.processor(
            text=text,
            return_tensors="pt"
        ).to(self.device)
        
        speech = self.model.generate_speech(
            inputs["input_ids"],
            self.speaker_embeddings,
            vocoder=self.vocoder
        )
        
        return speech.cpu().numpy()

    async def shutdown(self) -> None:
        del self.processor
        del self.model
        del self.vocoder
        del self.speaker_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class AzureAIComponent:
    def __init__(self, settings: "Settings"):
        self.settings = settings
        self.client = None

    async def initialize(self) -> None:
        self.client = TextAnalyticsClient(
            endpoint=self.settings.AZURE_ENDPOINT,
            credential=AzureKeyCredential(self.settings.AZURE_KEY)
        )

    async def analyze_sentiment(self, text: str) -> float:
        response = await self.client.analyze_sentiment(documents=[text])
        sentiment = response[0].sentiment
        return sentiment

    async def shutdown(self) -> None:
        del self.client