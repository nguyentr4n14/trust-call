# Voice Scam Shield – Multilingual AI for Real-Time Call Scam Detection

A comprehensive AI-powered system that detects scam calls and synthetic voices in real-time across multiple languages (English, Spanish, French).

## 🎯 Project Overview

**Track**: VC big bets (Cybersecurity)

### Goal
Develop a multilingual AI agent that works during phone or video calls to:
- Detect scam intent and synthetic voices in real time
- Alert users with discreet on-call feedback
- Support English, Spanish, and French (with extensibility for more languages)

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │  AI Pipeline    │
│   (React)       │◄──►│   (FastAPI)     │◄──►│  (ML Models)    │
│                 │    │                 │    │                 │
│ • Live Dashboard│    │ • WebSocket     │    │ • ASR (Whisper) │
│ • Risk Display │    │ • Audio Ingestion│   │ • Anti-Spoofing │
│ • Alerts UI     │    │ • API Endpoints │    │ • Scam Detection│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                   ┌─────────────────┐
                   │  External APIs  │
                   │                 │
                   │ • Twilio SDK    │
                   │ • WebRTC       │
                   │ • ElevenLabs    │
                   └─────────────────┘
```

## 🚀 Core Features (MVP)

### 1. Real-Time Call Audio Monitoring
- Integration with Twilio, WebRTC, or Zoom SDK
- Voice activity detection and speaker diarization
- Live audio stream processing

### 2. Multilingual Scam & Deepfake Detection
- Streaming transcription using Whisper/Deepgram
- Scam pattern detection with LLM prompts
- Anti-spoofing detection (AASIST/RawNet2)
- Support for English, Spanish, and French

### 3. On-Call User Alerts
- Visual dashboard with live risk scoring
- Discreet TTS alerts via ElevenLabs
- Non-intrusive notification system

## 🎯 Stretch Goals

- **Caller Verification**: Voice matching against safe lists
- **Incident Reports**: Post-call analysis and recommendations
- **Advanced Analytics**: Pattern recognition and learning

## 📊 Evaluation Criteria

- **Coverage**: Phone and video call support across EN/ES/FR
- **Detection Accuracy**: ≥80% scam vs safe classification
- **Anti-Spoofing**: ≤10% Equal Error Rate on synthetic voice detection
- **Latency**: Alerts within 2 seconds of suspicious speech
- **User Experience**: Clear, discreet alerts without call disruption

## 🛠️ Technology Stack

### Frontend
- **Framework**: React/Next.js
- **Real-time**: WebSocket connection
- **UI Components**: Material-UI or Tailwind CSS
- **State Management**: Redux Toolkit

### Backend
- **Framework**: FastAPI
- **Real-time**: WebSocket support
- **Database**: PostgreSQL with SQLAlchemy
- **Authentication**: JWT tokens

### AI/ML Pipeline
- **ASR**: OpenAI Whisper (multilingual)
- **Anti-Spoofing**: AASIST, RawNet2
- **Scam Detection**: GPT-4o-mini or fine-tuned transformers
- **TTS Alerts**: ElevenLabs API

### Infrastructure
- **Containerization**: Docker & Docker Compose
- **API Integration**: Twilio, WebRTC
- **Monitoring**: Prometheus + Grafana

## 📁 Project Structure

```
trust-call/
├── frontend/                 # React application
├── backend/                  # FastAPI server
├── ai-pipeline/             # ML models and processing
├── shared/                  # Shared utilities and types
├── docker/                  # Docker configurations
├── docs/                    # Documentation
├── tests/                   # Test suites
└── scripts/                 # Utility scripts
```

## 🚀 Quick Start

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd trust-call
   ```

2. **Environment Setup**
   ```bash
   cp .env.example .env
   # Configure your API keys and settings
   ```

3. **Run with Docker**
   ```bash
   docker-compose up --build
   ```

4. **Development Mode**
   ```bash
   # Backend
   cd backend && python -m uvicorn main:app --reload
   
   # Frontend
   cd frontend && npm start
   
   # AI Pipeline
   cd ai-pipeline && python main.py
   ```

## 🔧 Configuration

### Required API Keys
- OpenAI API key (for Whisper and GPT)
- ElevenLabs API key (for TTS alerts)
- Twilio credentials (for call integration)
- Deepgram API key (alternative ASR)

### Environment Variables
```bash
OPENAI_API_KEY=your_openai_key
ELEVENLABS_API_KEY=your_elevenlabs_key
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
DATABASE_URL=postgresql://user:pass@localhost/scam_shield
```

## 📚 Documentation

- [API Documentation](./docs/api.md)
- [Frontend Guide](./docs/frontend.md)
- [AI Pipeline Documentation](./docs/ai-pipeline.md)
- [Deployment Guide](./docs/deployment.md)

## 🧪 Testing

```bash
# Run all tests
docker-compose -f docker-compose.test.yml up

# Backend tests
cd backend && pytest

# Frontend tests
cd frontend && npm test

# AI Pipeline tests
cd ai-pipeline && python -m pytest
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support, email support@scamshield.ai or join our Slack channel.

---

**Voice Scam Shield** - Protecting people from AI-driven scam calls in real-time. 🛡️
