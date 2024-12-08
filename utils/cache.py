import torch
from pathlib import Path
import asyncio
from functools import lru_cache
from typing import Optional, Dict, Any
import logging

class ModelCache:
    def __init__(self, settings: "Settings"):
        self.settings = settings
        self.cache_dir = settings.CACHE_DIR
        self.cache_dir.mkdir(exist_ok=True)
        self.loaded_models = {}
        self.logger = logging.getLogger(__name__)

    async def preload_models(self):
        """Preload commonly used models into memory"""
        models_to_load = [
            ('whisper', 'openai/whisper-medium'),
            ('speecht5', 'microsoft/speecht5_tts'),
            ('yolos', 'hustvl/yolos-tiny')
        ]
        
        for name, model_id in models_to_load:
            await self._load_model(name, model_id)

    @lru_cache(maxsize=10)
    async def _load_model(self, name, model_id):
        try:
            cache_path = self.cache_dir / f"{name}.pt"
            
            if cache_path.exists():
                self.loaded_models[name] = torch.load(cache_path)
            else:
                model = await self._download_model(model_id)
                torch.save(model, cache_path)
                self.loaded_models[name] = model
                
        except Exception as e:
            self.logger.error(f"Failed to load model {name}: {str(e)}")
            raise

    async def _download_model(self, model_id: str) -> Any:
        """Download and load model from HuggingFace with CPU optimizations"""
        try:
            import torch
            torch.set_num_threads(self.settings.CPU_THREADS)
            
            if 'whisper' in model_id.lower():
                from transformers import WhisperForConditionalGeneration
                model = WhisperForConditionalGeneration.from_pretrained(
                    model_id,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float32
                )
            elif 'speecht5' in model_id.lower():
                from transformers import SpeechT5ForTextToSpeech
                model = SpeechT5ForTextToSpeech.from_pretrained(
                    model_id,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float32
                )
            elif 'yolos' in model_id.lower():
                from transformers import YolosForObjectDetection
                model = YolosForObjectDetection.from_pretrained(
                    model_id,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float32
                )
            else:
                from transformers import AutoModel
                model = AutoModel.from_pretrained(
                    model_id,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float32
                )
            
            return model.to(self.settings.MODEL_DEVICE)
            
        except Exception as e:
            self.logger.error(f"Failed to download model {model_id}: {str(e)}")
            raise

    def get_model(self, name):
        return self.loaded_models.get(name)