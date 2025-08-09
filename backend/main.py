"""
Voice Scam Shield - FastAPI Backend
Main application entry point with real-time scam detection capabilities.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes import audio, calls, dashboard, auth, reports
from app.core.config import settings
from app.core.database import get_db, init_db
from app.core.websocket_manager import WebSocketManager
from app.services.audio_processor import AudioProcessorService
from app.services.scam_detector import ScamDetectorService
from app.services.anti_spoofing import AntiSpoofingService
from app.models.call import CallSession
from app.schemas.call import CallAlert, CallAnalysis

# Configure logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

# WebSocket manager for real-time communication
websocket_manager = WebSocketManager()

# AI Services
audio_processor = AudioProcessorService()
scam_detector = ScamDetectorService()
anti_spoofing = AntiSpoofingService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting Voice Scam Shield Backend...")
    await init_db()
    await audio_processor.initialize()
    await scam_detector.initialize()
    await anti_spoofing.initialize()
    logger.info("Backend initialization complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Voice Scam Shield Backend...")
    await audio_processor.cleanup()
    await scam_detector.cleanup()
    await anti_spoofing.cleanup()


# Create FastAPI application
app = FastAPI(
    title="Voice Scam Shield API",
    description="Multilingual AI for Real-Time Call Scam Detection",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if settings.DEBUG else ["localhost", "127.0.0.1"]
)

# Include API routes
app.include_router(auth.router, prefix="/api/v1/auth", tags=["authentication"])
app.include_router(calls.router, prefix="/api/v1/calls", tags=["calls"])
app.include_router(audio.router, prefix="/api/v1/audio", tags=["audio"])
app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["dashboard"])
app.include_router(reports.router, prefix="/api/v1/reports", tags=["reports"])


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "audio_processor": await audio_processor.health_check(),
            "scam_detector": await scam_detector.health_check(),
            "anti_spoofing": await anti_spoofing.health_check(),
        }
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Voice Scam Shield API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "operational"
    }


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    WebSocket endpoint for real-time call monitoring and scam detection
    """
    await websocket_manager.connect(websocket, session_id)
    
    try:
        # Initialize call session
        call_session = CallSession(
            session_id=session_id,
            status="active",
            language="en"  # Default language, will be detected
        )
        
        logger.info(f"WebSocket connected for session: {session_id}")
        
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()
            
            try:
                # Process audio chunk
                audio_analysis = await audio_processor.process_chunk(
                    audio_data=data,
                    session_id=session_id
                )
                
                if audio_analysis.has_speech:
                    # Detect language if not already set
                    if call_session.language == "en":  # Default check
                        detected_lang = await scam_detector.detect_language(
                            audio_analysis.transcription
                        )
                        if detected_lang in settings.SUPPORTED_LANGUAGES:
                            call_session.language = detected_lang
                    
                    # Run scam detection
                    scam_analysis = await scam_detector.analyze_text(
                        text=audio_analysis.transcription,
                        language=call_session.language,
                        session_id=session_id
                    )
                    
                    # Run anti-spoofing detection
                    spoofing_analysis = await anti_spoofing.analyze_audio(
                        audio_data=data,
                        session_id=session_id
                    )
                    
                    # Combine analyses
                    combined_analysis = CallAnalysis(
                        session_id=session_id,
                        timestamp=audio_analysis.timestamp,
                        transcription=audio_analysis.transcription,
                        language=call_session.language,
                        scam_score=scam_analysis.risk_score,
                        spoofing_score=spoofing_analysis.synthetic_probability,
                        risk_level=_calculate_risk_level(
                            scam_analysis.risk_score,
                            spoofing_analysis.synthetic_probability
                        ),
                        confidence=min(
                            scam_analysis.confidence,
                            spoofing_analysis.confidence
                        ),
                        indicators=scam_analysis.indicators + spoofing_analysis.indicators
                    )
                    
                    # Send real-time analysis to frontend
                    await websocket_manager.send_analysis(session_id, combined_analysis)
                    
                    # Check if alert should be triggered
                    if combined_analysis.risk_level in ["high", "critical"]:
                        alert = CallAlert(
                            session_id=session_id,
                            alert_type="scam_detected" if combined_analysis.scam_score > 0.7 else "suspicious_activity",
                            risk_level=combined_analysis.risk_level,
                            message=_generate_alert_message(combined_analysis),
                            timestamp=combined_analysis.timestamp,
                            language=call_session.language
                        )
                        
                        await websocket_manager.send_alert(session_id, alert)
                        
                        # Log high-risk event
                        logger.warning(
                            f"High-risk activity detected in session {session_id}: "
                            f"scam_score={combined_analysis.scam_score}, "
                            f"spoofing_score={combined_analysis.spoofing_score}"
                        )
                
            except Exception as e:
                logger.error(f"Error processing audio chunk for session {session_id}: {e}")
                await websocket_manager.send_error(session_id, str(e))
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(session_id)
        logger.info(f"WebSocket disconnected for session: {session_id}")
        
        # Update call session status
        call_session.status = "completed"
        
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        await websocket_manager.send_error(session_id, str(e))
        websocket_manager.disconnect(session_id)


def _calculate_risk_level(scam_score: float, spoofing_score: float) -> str:
    """Calculate overall risk level based on individual scores"""
    max_score = max(scam_score, spoofing_score)
    
    if max_score >= 0.9:
        return "critical"
    elif max_score >= 0.7:
        return "high"
    elif max_score >= 0.5:
        return "medium"
    elif max_score >= 0.3:
        return "low"
    else:
        return "safe"


def _generate_alert_message(analysis: CallAnalysis) -> str:
    """Generate appropriate alert message based on analysis"""
    if analysis.scam_score > 0.7:
        if analysis.language == "es":
            return "¡ADVERTENCIA! Esta llamada puede ser fraudulenta. No comparta información personal."
        elif analysis.language == "fr":
            return "ATTENTION! Cet appel peut être frauduleux. Ne partagez pas d'informations personnelles."
        else:
            return "WARNING! This call may be fraudulent. Do not share personal information."
    
    elif analysis.spoofing_score > 0.7:
        if analysis.language == "es":
            return "Voz sintética detectada. Tenga cuidado con esta llamada."
        elif analysis.language == "fr":
            return "Voix synthétique détectée. Soyez prudent avec cet appel."
        else:
            return "Synthetic voice detected. Be cautious with this call."
    
    else:
        if analysis.language == "es":
            return "Actividad sospechosa detectada en esta llamada."
        elif analysis.language == "fr":
            return "Activité suspecte détectée dans cet appel."
        else:
            return "Suspicious activity detected in this call."


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
