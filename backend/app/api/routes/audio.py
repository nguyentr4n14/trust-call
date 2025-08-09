"""
Audio processing routes
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.api.routes.auth import get_current_user
from app.models.user import User
from app.services.audio_processor import AudioProcessorService
from app.services.scam_detector import ScamDetectorService
from app.services.anti_spoofing import AntiSpoofingService

router = APIRouter()

# Service instances (these would be injected in a real app)
audio_processor = AudioProcessorService()
scam_detector = ScamDetectorService()
anti_spoofing_service = AntiSpoofingService()


@router.post("/analyze")
async def analyze_audio(
    audio_file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    language: str = Form("en"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Analyze uploaded audio file for scam indicators"""
    try:
        # Read audio data
        audio_data = await audio_file.read()
        
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Process audio
        audio_analysis = await audio_processor.process_chunk(
            audio_data=audio_data,
            session_id=session_id or "upload",
            speaker="caller"
        )
        
        # Run scam detection if there's transcription
        scam_analysis = None
        if audio_analysis.transcription:
            scam_analysis = await scam_detector.analyze_text(
                text=audio_analysis.transcription,
                language=language,
                session_id=session_id or "upload"
            )
        
        # Run anti-spoofing detection
        spoofing_analysis = await anti_spoofing_service.analyze_audio(
            audio_data=audio_data,
            session_id=session_id or "upload"
        )
        
        return {
            "audio_analysis": {
                "has_speech": audio_analysis.has_speech,
                "transcription": audio_analysis.transcription,
                "language": audio_analysis.language,
                "confidence": audio_analysis.confidence,
                "features": audio_analysis.audio_features
            },
            "scam_analysis": {
                "risk_score": scam_analysis.risk_score if scam_analysis else 0.0,
                "confidence": scam_analysis.confidence if scam_analysis else 0.0,
                "indicators": scam_analysis.indicators if scam_analysis else [],
                "reasoning": scam_analysis.reasoning if scam_analysis else ""
            } if scam_analysis else None,
            "spoofing_analysis": {
                "synthetic_probability": spoofing_analysis.synthetic_probability,
                "confidence": spoofing_analysis.confidence,
                "is_synthetic": spoofing_analysis.is_synthetic,
                "indicators": spoofing_analysis.indicators
            },
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/features/{session_id}")
async def get_audio_features(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get extracted audio features for a session"""
    try:
        # Get session statistics
        stats = await audio_processor.get_session_stats(session_id)
        
        return {
            "session_id": session_id,
            "stats": stats,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get features: {str(e)}")


@router.post("/sessions/{session_id}/clear")
async def clear_session_buffer(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """Clear audio buffer for a session"""
    try:
        await audio_processor.clear_session_buffer(session_id)
        
        return {
            "message": f"Session buffer cleared for {session_id}",
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear buffer: {str(e)}")


@router.get("/health")
async def audio_service_health():
    """Check audio service health"""
    try:
        health_status = {
            "audio_processor": await audio_processor.health_check(),
            "scam_detector": await scam_detector.health_check(),
            "anti_spoofing": await anti_spoofing_service.health_check(),
            "timestamp": datetime.utcnow()
        }
        
        overall_healthy = all(health_status.values())
        
        return {
            "status": "healthy" if overall_healthy else "degraded",
            "services": health_status
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow()
        }
