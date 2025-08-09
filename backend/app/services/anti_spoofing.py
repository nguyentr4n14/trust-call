"""
Anti-spoofing service for detecting synthetic/deepfake voices
"""

import asyncio
import logging
import numpy as np
import torch
import librosa
from datetime import datetime
from typing import Dict, Any, List, Optional

from app.core.config import settings
from app.schemas.call import AntiSpoofingAnalysis

logger = logging.getLogger(__name__)


class AntiSpoofingService:
    """Service for detecting synthetic/deepfake voices"""
    
    def __init__(self):
        self.model = None
        self.device = None
        self.sample_rate = settings.SAMPLE_RATE
        self.model_version = "1.0.0"
        
        # Feature extractors and models will be loaded here
        self.feature_extractor = None
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the anti-spoofing models"""
        try:
            # Determine device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            # In a real implementation, you would load pre-trained models here
            # For now, we'll simulate with a placeholder
            await self._load_models()
            
            self.is_initialized = True
            logger.info("Anti-spoofing service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize anti-spoofing service: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.model is not None:
            del self.model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        self.is_initialized = False
        logger.info("Anti-spoofing service cleaned up")
    
    async def health_check(self) -> bool:
        """Check if the service is healthy"""
        return self.is_initialized
    
    async def _load_models(self):
        """Load anti-spoofing models"""
        try:
            # In a production environment, you would load actual models like:
            # - AASIST (Anti-spoofing using Automatic Speaker verification Inspired Self-aTtention)
            # - RawNet2
            # - LCNN (Light CNN)
            # - ResNet-based models
            
            # For demonstration, we'll create a simple placeholder model
            self.model = SimpleSpoofingDetector()
            logger.info("Anti-spoofing models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load anti-spoofing models: {e}")
            raise
    
    async def analyze_audio(
        self,
        audio_data: bytes,
        session_id: str = None
    ) -> AntiSpoofingAnalysis:
        """
        Analyze audio for spoofing/synthetic voice detection
        """
        try:
            if not self.is_initialized:
                raise RuntimeError("Anti-spoofing service not initialized")
            
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Normalize audio
            if len(audio_array) > 0:
                audio_float = audio_array.astype(np.float32) / 32767.0
            else:
                # Return safe result for empty audio
                return AntiSpoofingAnalysis(
                    session_id=session_id or "",
                    synthetic_probability=0.0,
                    confidence=1.0,
                    is_synthetic=False,
                    indicators=[],
                    model_version=self.model_version,
                    timestamp=datetime.utcnow()
                )
            
            # Extract features for spoofing detection
            features = await self._extract_spoofing_features(audio_float)
            
            # Run spoofing detection
            synthetic_probability, confidence = await self._detect_spoofing(features)
            
            # Determine if synthetic
            is_synthetic = synthetic_probability > settings.ANTI_SPOOFING_THRESHOLD
            
            # Generate indicators
            indicators = await self._generate_indicators(features, synthetic_probability)
            
            return AntiSpoofingAnalysis(
                session_id=session_id or "",
                synthetic_probability=synthetic_probability,
                confidence=confidence,
                is_synthetic=is_synthetic,
                voice_features=features,
                indicators=indicators,
                model_version=self.model_version,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error in anti-spoofing analysis: {e}")
            return AntiSpoofingAnalysis(
                session_id=session_id or "",
                synthetic_probability=0.0,
                confidence=0.0,
                is_synthetic=False,
                indicators=["analysis_error"],
                model_version=self.model_version,
                timestamp=datetime.utcnow()
            )
    
    async def _extract_spoofing_features(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Extract features relevant for spoofing detection"""
        try:
            if len(audio_data) == 0:
                return {}
            
            features = {}
            
            # Ensure minimum length
            if len(audio_data) < 1024:
                audio_data = np.pad(audio_data, (0, 1024 - len(audio_data)))
            
            # Basic spectral features
            features.update(await self._extract_spectral_features(audio_data))
            
            # Voice quality features
            features.update(await self._extract_voice_quality_features(audio_data))
            
            # Prosodic features
            features.update(await self._extract_prosodic_features(audio_data))
            
            # Frequency domain features
            features.update(await self._extract_frequency_features(audio_data))
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting spoofing features: {e}")
            return {}
    
    async def _extract_spectral_features(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Extract spectral features for spoofing detection"""
        features = {}
        
        try:
            # FFT-based features
            fft = np.fft.fft(audio_data)
            magnitude_spectrum = np.abs(fft)
            
            # Spectral characteristics
            features["spectral_mean"] = float(np.mean(magnitude_spectrum))
            features["spectral_std"] = float(np.std(magnitude_spectrum))
            features["spectral_skewness"] = float(self._calculate_skewness(magnitude_spectrum))
            features["spectral_kurtosis"] = float(self._calculate_kurtosis(magnitude_spectrum))
            
            # Spectral rolloff and centroid
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_data, sr=self.sample_rate
            )[0]
            features["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
            features["spectral_centroid_var"] = float(np.var(spectral_centroids))
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio_data, sr=self.sample_rate
            )[0]
            features["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))
            
            # Spectral flatness (measure of noise-like characteristics)
            spectral_flatness = librosa.feature.spectral_flatness(y=audio_data)[0]
            features["spectral_flatness_mean"] = float(np.mean(spectral_flatness))
            
        except Exception as e:
            logger.warning(f"Error extracting spectral features: {e}")
        
        return features
    
    async def _extract_voice_quality_features(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Extract voice quality features"""
        features = {}
        
        try:
            # Jitter and shimmer (voice quality measures)
            features.update(self._calculate_jitter_shimmer(audio_data))
            
            # Harmonic-to-noise ratio
            hnr = self._calculate_hnr(audio_data)
            features["harmonic_noise_ratio"] = float(hnr)
            
            # Zero crossing rate variability
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            features["zcr_mean"] = float(np.mean(zcr))
            features["zcr_var"] = float(np.var(zcr))
            
            # RMS energy variation
            rms_frames = librosa.feature.rms(y=audio_data)[0]
            features["rms_mean"] = float(np.mean(rms_frames))
            features["rms_var"] = float(np.var(rms_frames))
            
        except Exception as e:
            logger.warning(f"Error extracting voice quality features: {e}")
        
        return features
    
    async def _extract_prosodic_features(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Extract prosodic features (rhythm, stress, intonation)"""
        features = {}
        
        try:
            # Pitch tracking
            pitches, magnitudes = librosa.piptrack(
                y=audio_data, sr=self.sample_rate, threshold=0.1
            )
            
            # Extract fundamental frequency
            f0_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    f0_values.append(pitch)
            
            if f0_values:
                features["f0_mean"] = float(np.mean(f0_values))
                features["f0_std"] = float(np.std(f0_values))
                features["f0_range"] = float(np.max(f0_values) - np.min(f0_values))
                
                # F0 contour characteristics
                f0_diff = np.diff(f0_values)
                features["f0_contour_var"] = float(np.var(f0_diff))
                
        except Exception as e:
            logger.warning(f"Error extracting prosodic features: {e}")
        
        return features
    
    async def _extract_frequency_features(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Extract frequency domain features"""
        features = {}
        
        try:
            # Mel-frequency cepstral coefficients (important for spoofing detection)
            mfccs = librosa.feature.mfcc(
                y=audio_data, sr=self.sample_rate, n_mfcc=13
            )
            
            # MFCC statistics
            for i in range(min(5, mfccs.shape[0])):
                features[f"mfcc_{i}_mean"] = float(np.mean(mfccs[i]))
                features[f"mfcc_{i}_std"] = float(np.std(mfccs[i]))
            
            # Delta and delta-delta MFCC (important for detecting synthesis artifacts)
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            
            features["mfcc_delta_mean"] = float(np.mean(mfcc_delta))
            features["mfcc_delta2_mean"] = float(np.mean(mfcc_delta2))
            
            # Linear prediction coefficients (good for detecting synthesis)
            lpc_coeffs = self._calculate_lpc(audio_data, order=12)
            for i, coeff in enumerate(lpc_coeffs[:5]):
                features[f"lpc_{i}"] = float(coeff)
            
        except Exception as e:
            logger.warning(f"Error extracting frequency features: {e}")
        
        return features
    
    async def _detect_spoofing(self, features: Dict[str, Any]) -> tuple[float, float]:
        """Run spoofing detection on extracted features"""
        try:
            if not features:
                return 0.0, 0.0
            
            # In a real implementation, this would use trained models
            # For demonstration, we'll use heuristic rules
            
            synthetic_score = 0.0
            confidence = 0.8
            
            # Check for suspicious spectral characteristics
            if "spectral_flatness_mean" in features:
                flatness = features["spectral_flatness_mean"]
                if flatness > 0.8:  # Too flat, possibly synthetic
                    synthetic_score += 0.3
            
            # Check for unnatural pitch characteristics
            if "f0_std" in features and "f0_range" in features:
                f0_std = features["f0_std"]
                f0_range = features["f0_range"]
                
                if f0_std < 10:  # Too stable pitch
                    synthetic_score += 0.2
                if f0_range < 50:  # Too narrow pitch range
                    synthetic_score += 0.2
            
            # Check for unusual voice quality
            if "harmonic_noise_ratio" in features:
                hnr = features["harmonic_noise_ratio"]
                if hnr > 20:  # Too clean, possibly synthetic
                    synthetic_score += 0.25
            
            # Check for MFCC irregularities
            mfcc_features = [k for k in features.keys() if k.startswith("mfcc_")]
            if len(mfcc_features) > 5:
                # Check for unusual MFCC patterns that might indicate synthesis
                mfcc_values = [features[k] for k in mfcc_features if "mean" in k]
                if len(mfcc_values) > 0:
                    mfcc_var = np.var(mfcc_values)
                    if mfcc_var < 0.01:  # Too uniform
                        synthetic_score += 0.2
            
            return min(synthetic_score, 1.0), confidence
            
        except Exception as e:
            logger.error(f"Error in spoofing detection: {e}")
            return 0.0, 0.0
    
    async def _generate_indicators(self, features: Dict[str, Any], synthetic_prob: float) -> List[str]:
        """Generate human-readable indicators for the analysis"""
        indicators = []
        
        try:
            if synthetic_prob > 0.8:
                indicators.append("high_synthetic_probability")
            elif synthetic_prob > 0.6:
                indicators.append("moderate_synthetic_probability")
            
            # Specific feature indicators
            if "spectral_flatness_mean" in features and features["spectral_flatness_mean"] > 0.8:
                indicators.append("artificial_spectral_characteristics")
            
            if "f0_std" in features and features["f0_std"] < 10:
                indicators.append("unnatural_pitch_stability")
            
            if "harmonic_noise_ratio" in features and features["harmonic_noise_ratio"] > 20:
                indicators.append("unusually_clean_audio")
            
            if "zcr_var" in features and features["zcr_var"] < 0.001:
                indicators.append("artificial_voice_characteristics")
            
        except Exception as e:
            logger.warning(f"Error generating indicators: {e}")
        
        return indicators
    
    # Helper methods for feature calculation
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        if len(data) == 0:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        if len(data) == 0:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _calculate_jitter_shimmer(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Calculate jitter and shimmer (simplified version)"""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated pitch tracking
        features = {}
        
        try:
            # Simple jitter calculation (pitch period variation)
            # This is a very basic implementation
            autocorr = np.correlate(audio_data, audio_data, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            if len(autocorr) > 100:
                # Find peaks to estimate periods
                peaks = []
                for i in range(1, min(len(autocorr) - 1, 1000)):
                    if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                        if autocorr[i] > 0.1 * np.max(autocorr):
                            peaks.append(i)
                
                if len(peaks) > 2:
                    periods = np.diff(peaks)
                    if len(periods) > 1:
                        features["jitter"] = float(np.std(periods) / np.mean(periods) if np.mean(periods) > 0 else 0)
                        features["shimmer"] = float(np.std(periods) if len(periods) > 0 else 0)
            
        except Exception:
            pass
        
        return features
    
    def _calculate_hnr(self, audio_data: np.ndarray) -> float:
        """Calculate Harmonic-to-Noise Ratio (simplified)"""
        try:
            # This is a very simplified HNR calculation
            # In practice, you'd use more sophisticated methods
            
            # Calculate power spectrum
            fft = np.fft.fft(audio_data)
            power_spectrum = np.abs(fft) ** 2
            
            # Simple harmonic vs noise estimation
            # (This is not a proper HNR calculation, just a placeholder)
            total_power = np.sum(power_spectrum)
            if total_power > 0:
                # Use spectral peaks as proxy for harmonic content
                sorted_power = np.sort(power_spectrum)
                harmonic_power = np.sum(sorted_power[-10:])  # Top 10 components
                noise_power = total_power - harmonic_power
                
                if noise_power > 0:
                    return 10 * np.log10(harmonic_power / noise_power)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_lpc(self, audio_data: np.ndarray, order: int = 12) -> np.ndarray:
        """Calculate Linear Prediction Coefficients (simplified)"""
        try:
            # Simplified LPC calculation using autocorrelation method
            # This is a basic implementation
            
            if len(audio_data) < order + 1:
                return np.zeros(order)
            
            # Calculate autocorrelation
            autocorr = np.correlate(audio_data, audio_data, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr[:order + 1]
            
            # Solve Yule-Walker equations (simplified)
            if len(autocorr) == order + 1 and autocorr[0] != 0:
                # Create Toeplitz matrix
                R = np.array([autocorr[abs(i - j)] for i in range(order) for j in range(order)]).reshape(order, order)
                r = autocorr[1:order + 1]
                
                # Solve for LPC coefficients
                try:
                    lpc_coeffs = np.linalg.solve(R, r)
                    return lpc_coeffs
                except np.linalg.LinAlgError:
                    return np.zeros(order)
            
            return np.zeros(order)
            
        except Exception:
            return np.zeros(order)


class SimpleSpoofingDetector:
    """Simple placeholder for spoofing detection model"""
    
    def __init__(self):
        self.threshold = 0.5
    
    def predict(self, features: Dict[str, Any]) -> float:
        """Predict spoofing probability"""
        # This is a placeholder implementation
        # In practice, this would be a trained neural network
        return 0.1  # Low probability for demo
