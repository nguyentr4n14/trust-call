"""
Pydantic schemas for call-related data
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class AudioChunk(BaseModel):
    """Audio chunk data for processing"""
    session_id: str
    chunk_data: bytes
    timestamp: datetime
    chunk_index: int
    sample_rate: int = 16000
    duration_ms: int = 1000


class AudioAnalysis(BaseModel):
    """Audio analysis results"""
    session_id: str
    timestamp: datetime
    has_speech: bool
    speaker: str  # user, caller, unknown
    transcription: str = ""
    language: str = "en"
    confidence: float = 0.0
    audio_features: Optional[Dict[str, Any]] = None


class ScamAnalysis(BaseModel):
    """Scam detection analysis results"""
    session_id: str
    text: str
    language: str
    risk_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    indicators: List[str] = []
    reasoning: Optional[str] = None
    detected_patterns: List[str] = []
    timestamp: datetime


class AntiSpoofingAnalysis(BaseModel):
    """Anti-spoofing detection results"""
    session_id: str
    synthetic_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    is_synthetic: bool
    voice_features: Optional[Dict[str, Any]] = None
    indicators: List[str] = []
    model_version: str = "1.0.0"
    timestamp: datetime


class CallAnalysis(BaseModel):
    """Combined call analysis results"""
    session_id: str
    timestamp: datetime
    transcription: str
    language: str
    
    # Scores
    scam_score: float = Field(ge=0.0, le=1.0)
    spoofing_score: float = Field(ge=0.0, le=1.0)
    risk_level: str = Field(regex=r"^(safe|low|medium|high|critical)$")
    confidence: float = Field(ge=0.0, le=1.0)
    
    # Indicators and context
    indicators: List[str] = []
    reasoning: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class CallAlert(BaseModel):
    """Call alert/warning"""
    session_id: str
    alert_type: str
    risk_level: str
    title: Optional[str] = None
    message: str
    language: str = "en"
    severity: str = "medium"
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class CallSessionCreate(BaseModel):
    """Create call session request"""
    caller_number: Optional[str] = None
    caller_name: Optional[str] = None
    language: str = "en"
    metadata: Optional[Dict[str, Any]] = None


class CallSessionUpdate(BaseModel):
    """Update call session request"""
    status: Optional[str] = None
    caller_number: Optional[str] = None
    caller_name: Optional[str] = None
    language: Optional[str] = None
    ended_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class CallSessionResponse(BaseModel):
    """Call session response"""
    id: int
    session_id: str
    user_id: Optional[int] = None
    
    # Call info
    caller_number: Optional[str] = None
    caller_name: Optional[str] = None
    call_duration: Optional[int] = None
    language: str
    
    # Status
    status: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    
    # Analysis results
    overall_risk_score: float
    scam_probability: float
    spoofing_probability: float
    risk_level: str
    
    # Flags
    is_scam: bool
    is_synthetic: bool
    has_alerts: bool
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = None
    
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class CallTranscriptResponse(BaseModel):
    """Call transcript response"""
    session_id: str
    full_transcript: str
    formatted_transcript: str
    primary_language: str
    languages_detected: List[str]
    overall_confidence: float
    total_duration: float
    word_count: int
    speaker_count: int
    key_phrases: Optional[List[str]] = None
    sentiment_score: Optional[float] = None
    topics: Optional[List[str]] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class VoiceProfileCreate(BaseModel):
    """Create voice profile request"""
    profile_name: str
    phone_number: Optional[str] = None
    relationship: str = "unknown"
    voice_sample: bytes  # Audio sample for training


class VoiceProfileResponse(BaseModel):
    """Voice profile response"""
    id: int
    user_id: int
    profile_name: str
    phone_number: Optional[str] = None
    relationship: str
    sample_count: int
    verification_count: int
    successful_verifications: int
    failed_verifications: int
    is_active: bool
    is_trusted: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class CallStatsResponse(BaseModel):
    """Call statistics response"""
    total_calls: int
    scam_calls_detected: int
    synthetic_voices_detected: int
    average_risk_score: float
    languages_detected: Dict[str, int]
    risk_level_distribution: Dict[str, int]
    detection_accuracy: Optional[float] = None
    false_positive_rate: Optional[float] = None
