"""
Audio processing service for real-time audio analysis
"""

import asyncio
import logging
import numpy as np
import librosa
import webrtcvad
from datetime import datetime
from typing import Optional, Dict, Any

from app.core.config import settings
from app.schemas.call import AudioAnalysis

logger = logging.getLogger(__name__)


class AudioProcessorService:
    """Service for processing real-time audio streams"""
    
    def __init__(self):
        self.vad = None  # Voice Activity Detection
        self.sample_rate = settings.SAMPLE_RATE
        self.chunk_duration_ms = settings.CHUNK_DURATION_MS
        self.frame_duration_ms = 30  # VAD frame duration
        
        # Audio buffers for each session
        self.session_buffers: Dict[str, np.ndarray] = {}
        self.session_metadata: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self):
        """Initialize the audio processor"""
        try:
            # Initialize WebRTC Voice Activity Detection
            self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2 (0-3)
            logger.info("Audio processor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize audio processor: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        self.session_buffers.clear()
        self.session_metadata.clear()
        logger.info("Audio processor cleaned up")
    
    async def health_check(self) -> bool:
        """Check if the service is healthy"""
        return self.vad is not None
    
    async def process_chunk(
        self,
        audio_data: bytes,
        session_id: str,
        speaker: str = "unknown"
    ) -> AudioAnalysis:
        """
        Process an audio chunk and return analysis results
        """
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Normalize audio
            if len(audio_array) > 0:
                audio_float = audio_array.astype(np.float32) / 32767.0
            else:
                audio_float = np.array([])
            
            # Add to session buffer
            if session_id not in self.session_buffers:
                self.session_buffers[session_id] = np.array([])
                self.session_metadata[session_id] = {
                    "total_chunks": 0,
                    "speech_chunks": 0,
                    "last_activity": datetime.utcnow()
                }
            
            # Append to buffer
            self.session_buffers[session_id] = np.concatenate([
                self.session_buffers[session_id],
                audio_float
            ])
            
            # Update metadata
            self.session_metadata[session_id]["total_chunks"] += 1
            self.session_metadata[session_id]["last_activity"] = datetime.utcnow()
            
            # Voice Activity Detection
            has_speech = await self._detect_voice_activity(audio_data)
            
            if has_speech:
                self.session_metadata[session_id]["speech_chunks"] += 1
            
            # Extract audio features
            audio_features = await self._extract_audio_features(audio_float)
            
            # Create analysis result
            analysis = AudioAnalysis(
                session_id=session_id,
                timestamp=datetime.utcnow(),
                has_speech=has_speech,
                speaker=speaker,
                transcription="",  # Will be filled by ASR service
                language="en",     # Will be detected by language detection
                confidence=0.0,    # Will be set by ASR service
                audio_features=audio_features
            )
            
            # Trim buffer if it gets too large
            max_buffer_size = settings.MAX_AUDIO_BUFFER_SIZE * self.sample_rate
            if len(self.session_buffers[session_id]) > max_buffer_size:
                # Keep only the last portion
                self.session_buffers[session_id] = self.session_buffers[session_id][-max_buffer_size:]
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error processing audio chunk for session {session_id}: {e}")
            # Return empty analysis on error
            return AudioAnalysis(
                session_id=session_id,
                timestamp=datetime.utcnow(),
                has_speech=False,
                speaker=speaker,
                transcription="",
                language="en",
                confidence=0.0
            )
    
    async def _detect_voice_activity(self, audio_data: bytes) -> bool:
        """Detect if audio contains speech using WebRTC VAD"""
        try:
            # WebRTC VAD expects specific frame sizes
            frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)
            
            # Process in frames
            speech_frames = 0
            total_frames = 0
            
            for i in range(0, len(audio_data), frame_size * 2):  # 2 bytes per sample
                frame = audio_data[i:i + frame_size * 2]
                
                if len(frame) == frame_size * 2:
                    try:
                        is_speech = self.vad.is_speech(frame, self.sample_rate)
                        if is_speech:
                            speech_frames += 1
                        total_frames += 1
                    except Exception:
                        # Skip problematic frames
                        continue
            
            # Consider it speech if more than 30% of frames contain speech
            if total_frames > 0:
                speech_ratio = speech_frames / total_frames
                return speech_ratio > 0.3
            
            return False
            
        except Exception as e:
            logger.warning(f"VAD detection failed: {e}")
            return False
    
    async def _extract_audio_features(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Extract audio features for analysis"""
        try:
            if len(audio_data) == 0:
                return {}
            
            # Ensure we have enough samples
            if len(audio_data) < 1024:
                # Pad with zeros if too short
                audio_data = np.pad(audio_data, (0, 1024 - len(audio_data)))
            
            features = {}
            
            # Basic audio statistics
            features["rms_energy"] = float(np.sqrt(np.mean(audio_data**2)))
            features["zero_crossing_rate"] = float(
                np.mean(librosa.feature.zero_crossing_rate(audio_data)[0])
            )
            
            # Spectral features
            try:
                # Spectral centroid
                spectral_centroids = librosa.feature.spectral_centroid(
                    y=audio_data, sr=self.sample_rate
                )[0]
                features["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
                features["spectral_centroid_std"] = float(np.std(spectral_centroids))
                
                # Spectral rolloff
                spectral_rolloff = librosa.feature.spectral_rolloff(
                    y=audio_data, sr=self.sample_rate
                )[0]
                features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))
                
                # MFCC features (first few coefficients)
                mfccs = librosa.feature.mfcc(
                    y=audio_data, sr=self.sample_rate, n_mfcc=13
                )
                for i in range(min(5, mfccs.shape[0])):
                    features[f"mfcc_{i}_mean"] = float(np.mean(mfccs[i]))
                    features[f"mfcc_{i}_std"] = float(np.std(mfccs[i]))
                    
            except Exception as e:
                logger.warning(f"Failed to extract spectral features: {e}")
            
            # Pitch-related features
            try:
                # Fundamental frequency estimation
                pitches, magnitudes = librosa.piptrack(
                    y=audio_data, sr=self.sample_rate, threshold=0.1
                )
                
                # Extract pitch values
                pitch_values = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_values.append(pitch)
                
                if pitch_values:
                    features["pitch_mean"] = float(np.mean(pitch_values))
                    features["pitch_std"] = float(np.std(pitch_values))
                    features["pitch_min"] = float(np.min(pitch_values))
                    features["pitch_max"] = float(np.max(pitch_values))
                    
            except Exception as e:
                logger.warning(f"Failed to extract pitch features: {e}")
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            return {}
    
    async def get_session_buffer(self, session_id: str) -> Optional[np.ndarray]:
        """Get the current audio buffer for a session"""
        return self.session_buffers.get(session_id)
    
    async def clear_session_buffer(self, session_id: str):
        """Clear the audio buffer for a session"""
        if session_id in self.session_buffers:
            del self.session_buffers[session_id]
        if session_id in self.session_metadata:
            del self.session_metadata[session_id]
    
    async def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session"""
        if session_id not in self.session_metadata:
            return {}
        
        metadata = self.session_metadata[session_id]
        buffer_length = len(self.session_buffers.get(session_id, []))
        
        return {
            "total_chunks": metadata["total_chunks"],
            "speech_chunks": metadata["speech_chunks"],
            "speech_ratio": (
                metadata["speech_chunks"] / metadata["total_chunks"]
                if metadata["total_chunks"] > 0 else 0
            ),
            "buffer_duration_seconds": buffer_length / self.sample_rate,
            "last_activity": metadata["last_activity"]
        }
