"""
Configuration settings for Voice Scam Shield Backend
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import validator


class Settings(BaseSettings):
    """Application settings"""
    
    # Basic app configuration
    APP_NAME: str = "Voice Scam Shield"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    SECRET_KEY: str = "your-secret-key-change-in-production"
    
    # Server configuration
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    
    # Database
    DATABASE_URL: str = "postgresql://scam_shield:password@localhost:5432/scam_shield_db"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = None
    ELEVENLABS_API_KEY: Optional[str] = None
    TWILIO_ACCOUNT_SID: Optional[str] = None
    TWILIO_AUTH_TOKEN: Optional[str] = None
    DEEPGRAM_API_KEY: Optional[str] = None
    
    # AI Configuration
    OPENAI_MODEL: str = "gpt-4o-mini"
    WHISPER_MODEL_SIZE: str = "base"
    
    # Detection Thresholds
    SCAM_DETECTION_THRESHOLD: float = 0.7
    ANTI_SPOOFING_THRESHOLD: float = 0.5
    CONFIDENCE_THRESHOLD: float = 0.6
    
    # Audio Processing
    SAMPLE_RATE: int = 16000
    CHUNK_DURATION_MS: int = 1000
    MAX_AUDIO_BUFFER_SIZE: int = 10000
    
    # Supported Languages
    SUPPORTED_LANGUAGES: List[str] = ["en", "es", "fr"]
    DEFAULT_LANGUAGE: str = "en"
    
    # Security
    JWT_SECRET_KEY: str = "jwt-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 100
    MAX_CONCURRENT_CONNECTIONS: int = 50
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Monitoring
    SENTRY_DSN: Optional[str] = None
    PROMETHEUS_PORT: int = 9090
    
    # Feature Flags
    ENABLE_CALLER_VERIFICATION: bool = False
    ENABLE_INCIDENT_REPORTS: bool = True
    ENABLE_ADVANCED_ANALYTICS: bool = False
    ENABLE_VOICE_MATCHING: bool = False
    
    @validator("CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v
    
    @validator("SUPPORTED_LANGUAGES", pre=True)
    def assemble_supported_languages(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v
    
    class Config:
        case_sensitive = True
        env_file = ".env"


# Create settings instance
settings = Settings()
