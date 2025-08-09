"""
WebSocket connection manager for real-time communication
"""

import json
import logging
from typing import Dict, List, Optional
from fastapi import WebSocket
from datetime import datetime

from app.schemas.call import CallAnalysis, CallAlert

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections for real-time communication"""
    
    def __init__(self):
        # Active connections: session_id -> WebSocket
        self.active_connections: Dict[str, WebSocket] = {}
        # Connection metadata
        self.connection_info: Dict[str, dict] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.connection_info[session_id] = {
            "connected_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
        }
        logger.info(f"WebSocket connected: {session_id}")
    
    def disconnect(self, session_id: str):
        """Remove a WebSocket connection"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.connection_info:
            del self.connection_info[session_id]
        logger.info(f"WebSocket disconnected: {session_id}")
    
    async def send_message(self, session_id: str, message: dict):
        """Send a message to a specific session"""
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(
                    json.dumps(message, default=str)
                )
                self.connection_info[session_id]["last_activity"] = datetime.utcnow()
            except Exception as e:
                logger.error(f"Error sending message to {session_id}: {e}")
                self.disconnect(session_id)
    
    async def send_analysis(self, session_id: str, analysis: CallAnalysis):
        """Send call analysis results to frontend"""
        message = {
            "type": "analysis",
            "data": analysis.dict()
        }
        await self.send_message(session_id, message)
    
    async def send_alert(self, session_id: str, alert: CallAlert):
        """Send alert to frontend"""
        message = {
            "type": "alert",
            "data": alert.dict()
        }
        await self.send_message(session_id, message)
        logger.warning(f"Alert sent to {session_id}: {alert.alert_type}")
    
    async def send_error(self, session_id: str, error_message: str):
        """Send error message to frontend"""
        message = {
            "type": "error",
            "data": {
                "message": error_message,
                "timestamp": datetime.utcnow()
            }
        }
        await self.send_message(session_id, message)
    
    async def send_status(self, session_id: str, status: str, data: dict = None):
        """Send status update to frontend"""
        message = {
            "type": "status",
            "data": {
                "status": status,
                "timestamp": datetime.utcnow(),
                **(data or {})
            }
        }
        await self.send_message(session_id, message)
    
    async def broadcast(self, message: dict, exclude_sessions: List[str] = None):
        """Broadcast message to all connected sessions"""
        exclude_sessions = exclude_sessions or []
        
        for session_id in list(self.active_connections.keys()):
            if session_id not in exclude_sessions:
                await self.send_message(session_id, message)
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs"""
        return list(self.active_connections.keys())
    
    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)
    
    def get_session_info(self, session_id: str) -> Optional[dict]:
        """Get connection info for a specific session"""
        return self.connection_info.get(session_id)
    
    async def ping_all_connections(self):
        """Send ping to all connections to check if they're alive"""
        message = {
            "type": "ping",
            "data": {
                "timestamp": datetime.utcnow()
            }
        }
        
        disconnected_sessions = []
        for session_id in list(self.active_connections.keys()):
            try:
                await self.send_message(session_id, message)
            except Exception as e:
                logger.warning(f"Session {session_id} appears disconnected: {e}")
                disconnected_sessions.append(session_id)
        
        # Clean up disconnected sessions
        for session_id in disconnected_sessions:
            self.disconnect(session_id)
