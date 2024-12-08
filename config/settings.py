from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, ConfigDict
from typing import Optional, Any
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='forbid',
        validate_default=True
    )
    
    # Base paths
    BASE_DIR: Path = Field(default=Path(__file__).parent.parent)
    DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")
    MODEL_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "models")
    CACHE_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "cache")
    KNOWN_FACES_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data" / "faces")
    
    # System
    DEBUG: bool = Field(default=False)
    LOG_LEVEL: str = Field(default="INFO")
    ENV: str = Field(default="production")
    
    # Security (updated)
    SECRET_KEY: str = Field(
        default_factory=lambda: os.getenv('SECRET_KEY', os.urandom(32).hex()),
    )
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_EXPIRATION: int = Field(default=3600)  # 1 hour
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 5000
    API_WORKERS: int = 4
    API_TIMEOUT: int = 30
    RATE_LIMIT_PER_MINUTE: int = 100
    
    # Model Configurations
    MODEL_DEVICE: str = "cpu"  # Changed from "cuda" to "cpu"
    MODEL_BATCH_SIZE: int = 1
    INFERENCE_TIMEOUT: int = 60  # Increased timeout for CPU inference
    CPU_THREADS: int = 4  # New setting for CPU threading
    
    # AI Model IDs - Using smaller models better suited for CPU
    FACE_MODEL_ID: str = "hustvl/yolos-tiny"  # Smallest YOLOS model
    EMOTION_MODEL_ID: str = "dima806/emotion_detection_model"
    WHISPER_MODEL_ID: str = "openai/whisper-tiny"  # Changed to tiny model
    SPEECHT5_MODEL_ID: str = "microsoft/speecht5_tts"
    
    # Azure Configuration (updated)
    AZURE_ENDPOINT: str = Field(
        default_factory=lambda: os.getenv('AZURE_ENDPOINT', 'https://models.inference.ai.azure.com'),
    )
    AZURE_KEY: str = Field(
        default_factory=lambda: os.getenv('AZURE_KEY'),
    )
    
    # Thresholds and Limits
    ENABLE_FACE_RECOGNITION: bool = True
    ENABLE_EMOTION_DETECTION: bool = True
    ENABLE_SPEECH_RECOGNITION: bool = True

    # Biometric Settings
    FACE_DETECTION_CONFIDENCE: float = Field(default=0.5)
    FACE_RECOGNITION_TOLERANCE: float = Field(default=0.6)
    FACE_RECOGNITION_MODEL: str = Field(default="vgg-face")
    FACE_DETECTOR_BACKEND: str = Field(default="opencv")
    
    model_config = {
        'env_file': '.env',
        'env_file_encoding': 'utf-8',
        'case_sensitive': True,
        'extra': 'forbid'
    }
    
    @field_validator("SECRET_KEY", "AZURE_ENDPOINT", "AZURE_KEY")
    @classmethod
    def validate_required_fields(cls, v: Optional[str], info: Any) -> str:
        """Validate required fields with development fallback"""
        if not v:
            env = os.getenv('ENV', 'production')
            if env == 'development':
                dummy_values = {
                    'SECRET_KEY': os.getenv('SECRET_KEY', os.urandom(32).hex()),
                    'AZURE_ENDPOINT': os.getenv('AZURE_ENDPOINT', 'https://models.inference.ai.azure.com'),
                    'AZURE_KEY': os.getenv('AZURE_KEY')
                }
                return dummy_values[info.field_name]
            raise ValueError(f"{info.field_name} is required in production mode")
        return v
    
    def create_directories(self):
        """Ensure all required directories exist"""
        for directory in [self.DATA_DIR, self.MODEL_DIR, self.CACHE_DIR, self.KNOWN_FACES_DIR]:
            directory.mkdir(parents=True, exist_ok=True)