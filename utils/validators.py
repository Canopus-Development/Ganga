from typing import Optional, Union, Dict, Any
import numpy as np
import cv2
from dataclasses import dataclass
import logging
import librosa
from PIL import Image

@dataclass
class ValidationResult:
    is_valid: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    validation_type: str = None

class InputValidator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.min_frame_size = (64, 64)
        self.min_face_size = (30, 30)
        self.max_audio_length = 30  # seconds
        self.max_image_size = (1920, 1080)
        self.supported_audio_formats = ['wav', 'mp3', 'ogg']
        self.supported_image_formats = ['jpg', 'jpeg', 'png']
        self.validation_stats = {
            'frame_validations': 0,
            'frame_failures': 0,
            'face_validations': 0,
            'face_failures': 0,
            'audio_validations': 0,
            'audio_failures': 0
        }

    def validate_frame(self, frame: np.ndarray) -> ValidationResult:
        """Validate video frame with detailed error reporting"""
        self.validation_stats['frame_validations'] += 1
        try:
            if frame is None:
                self._log_failure('frame', "Frame is None")
                return ValidationResult(False, "Frame is None", validation_type='frame')

            if not isinstance(frame, np.ndarray):
                self._log_failure('frame', f"Invalid frame type: {type(frame)}")
                return ValidationResult(False, f"Invalid frame type: {type(frame)}", validation_type='frame')

            if len(frame.shape) != 3:
                self._log_failure('frame', "Invalid frame dimensions")
                return ValidationResult(
                    False, 
                    f"Invalid frame dimensions: {frame.shape}",
                    metadata={'shape': frame.shape},
                    validation_type='frame'
                )

            if frame.shape[2] != 3:  # Check channels
                self._log_failure('frame', "Invalid color channels")
                return ValidationResult(
                    False,
                    "Invalid color channels",
                    metadata={'channels': frame.shape[2]},
                    validation_type='frame'
                )

            if frame.shape[0] > self.max_image_size[1] or frame.shape[1] > self.max_image_size[0]:
                self._log_failure('frame', "Frame too large")
                return ValidationResult(
                    False,
                    f"Frame too large: {frame.shape[:2]}",
                    metadata={'size': frame.shape[:2]},
                    validation_type='frame'
                )

            if frame.size == 0 or frame.shape[0] < self.min_frame_size[0] or frame.shape[1] < self.min_frame_size[1]:
                self._log_failure('frame', "Frame too small")
                return ValidationResult(
                    False,
                    f"Frame too small: {frame.shape[:2]}",
                    metadata={'size': frame.shape[:2]},
                    validation_type='frame'
                )

            return ValidationResult(True, metadata={'shape': frame.shape}, validation_type='frame')

        except Exception as e:
            self._log_failure('frame', f"Validation error: {str(e)}")
            return ValidationResult(False, str(e), validation_type='frame')

    def validate_face(self, face: np.ndarray) -> ValidationResult:
        """Validate face image quality and dimensions"""
        self.validation_stats['face_validations'] += 1
        try:
            if face is None:
                self._log_failure('face', "Face is None")
                return ValidationResult(False, "Face is None", validation_type='face')

            # Basic array validation
            if not isinstance(face, np.ndarray):
                self._log_failure('face', "Invalid face type")
                return ValidationResult(False, f"Invalid face type: {type(face)}", validation_type='face')

            # Size validation
            if face.shape[0] < self.min_face_size[0] or face.shape[1] < self.min_face_size[1]:
                self._log_failure('face', "Face too small")
                return ValidationResult(
                    False,
                    f"Face too small: {face.shape[:2]}",
                    metadata={'size': face.shape[:2]},
                    validation_type='face'
                )

            # Quality checks
            quality_score = self._assess_image_quality(face)
            if quality_score < 0.5:
                self._log_failure('face', "Poor face quality")
                return ValidationResult(
                    False,
                    "Poor face quality",
                    metadata={'quality_score': quality_score},
                    validation_type='face'
                )

            return ValidationResult(
                True,
                metadata={
                    'shape': face.shape,
                    'quality_score': quality_score
                },
                validation_type='face'
            )

        except Exception as e:
            self._log_failure('face', f"Validation error: {str(e)}")
            return ValidationResult(False, str(e), validation_type='face')

    def validate_audio(self, audio_data: np.ndarray, sample_rate: int) -> ValidationResult:
        """Validate audio signal quality and parameters"""
        self.validation_stats['audio_validations'] += 1
        try:
            if audio_data is None:
                self._log_failure('audio', "Audio is None")
                return ValidationResult(False, "Audio is None", validation_type='audio')

            # Basic array validation
            if not isinstance(audio_data, np.ndarray):
                self._log_failure('audio', "Invalid audio type")
                return ValidationResult(False, f"Invalid audio type: {type(audio_data)}", validation_type='audio')

            # Duration check
            duration = len(audio_data) / sample_rate
            if duration > self.max_audio_length:
                self._log_failure('audio', "Audio too long")
                return ValidationResult(
                    False,
                    f"Audio too long: {duration}s",
                    metadata={'duration': duration},
                    validation_type='audio'
                )

            # Quality checks
            signal_stats = self._analyze_audio_quality(audio_data, sample_rate)
            if not self._check_audio_quality(signal_stats):
                self._log_failure('audio', "Poor audio quality")
                return ValidationResult(
                    False,
                    "Poor audio quality",
                    metadata=signal_stats,
                    validation_type='audio'
                )

            return ValidationResult(
                True,
                metadata={
                    'duration': duration,
                    'sample_rate': sample_rate,
                    'stats': signal_stats
                },
                validation_type='audio'
            )

        except Exception as e:
            self._log_failure('audio', f"Validation error: {str(e)}")
            return ValidationResult(False, str(e), validation_type='audio')

    def _assess_image_quality(self, image: np.ndarray) -> float:
        """Assess image quality using various metrics"""
        try:
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Calculate metrics
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            contrast = gray.std()
            brightness = gray.mean()

            # Normalize and combine scores
            quality_score = (
                min(blur_score / 500.0, 1.0) * 0.4 +
                min(contrast / 80.0, 1.0) * 0.3 +
                min(abs(brightness - 128) / 128.0, 1.0) * 0.3
            )

            return quality_score

        except Exception as e:
            self.logger.error(f"Quality assessment error: {e}")
            return 0.0

    def _analyze_audio_quality(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Analyze audio signal quality metrics"""
        try:
            return {
                'rms_energy': float(np.sqrt(np.mean(audio_data**2))),
                'zero_crossing_rate': float(librosa.feature.zero_crossing_rate(audio_data)[0].mean()),
                'spectral_centroid': float(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0].mean())
            }
        except Exception as e:
            self.logger.error(f"Audio analysis error: {e}")
            return {}

    def _check_audio_quality(self, stats: Dict[str, float]) -> bool:
        """Check if audio quality metrics meet minimum thresholds"""
        if not stats:
            return False
            
        return (
            stats.get('rms_energy', 0) > 0.01 and
            0.01 < stats.get('zero_crossing_rate', 0) < 0.35 and
            stats.get('spectral_centroid', 0) > 500
        )

    def _log_failure(self, validation_type: str, message: str) -> None:
        """Log validation failures and update stats"""
        self.validation_stats[f'{validation_type}_failures'] += 1
        self.logger.warning(f"Validation failed: {message}")

    def get_stats(self) -> Dict[str, int]:
        """Get validation statistics"""
        return self.validation_stats.copy()

    def reset_stats(self) -> None:
        """Reset validation statistics"""
        for key in self.validation_stats:
            self.validation_stats[key] = 0