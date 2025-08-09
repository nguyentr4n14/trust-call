"""
Scam detection service using LLM and pattern matching
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

import openai
from app.core.config import settings
from app.schemas.call import ScamAnalysis

logger = logging.getLogger(__name__)


class ScamDetectorService:
    """Service for detecting scam patterns in conversation text"""
    
    def __init__(self):
        self.openai_client = None
        self.scam_patterns = {}
        self.language_models = {}
        
        # Multilingual scam indicators
        self.scam_keywords = {
            "en": [
                "urgent", "immediate action", "verify account", "suspended account",
                "click here", "confirm identity", "social security", "bank account",
                "credit card", "wire transfer", "gift card", "refund", "tax refund",
                "irs", "government agency", "police", "arrest", "legal action",
                "microsoft", "apple", "amazon", "paypal", "virus", "infected",
                "tech support", "remote access", "computer problem", "security breach",
                "limited time", "act now", "congratulations", "winner", "lottery",
                "inheritance", "prince", "lawyer", "died", "beneficiary"
            ],
            "es": [
                "urgente", "acción inmediata", "verificar cuenta", "cuenta suspendida",
                "haga clic aquí", "confirmar identidad", "seguro social", "cuenta bancaria",
                "tarjeta de crédito", "transferencia", "tarjeta de regalo", "reembolso",
                "hacienda", "agencia gubernamental", "policía", "arresto", "acción legal",
                "soporte técnico", "acceso remoto", "problema computadora", "virus",
                "tiempo limitado", "actúe ahora", "felicidades", "ganador", "lotería",
                "herencia", "príncipe", "abogado", "murió", "beneficiario"
            ],
            "fr": [
                "urgent", "action immédiate", "vérifier compte", "compte suspendu",
                "cliquez ici", "confirmer identité", "sécurité sociale", "compte bancaire",
                "carte de crédit", "virement", "carte cadeau", "remboursement",
                "impôts", "agence gouvernementale", "police", "arrestation", "action légale",
                "support technique", "accès à distance", "problème ordinateur", "virus",
                "temps limité", "agissez maintenant", "félicitations", "gagnant", "loterie",
                "héritage", "prince", "avocat", "décédé", "bénéficiaire"
            ]
        }
        
        # Scam pattern templates
        self.scam_patterns_templates = {
            "urgency": [
                r"(urgent|immediate|act now|limited time|expires)",
                r"(urgente|inmediato|actúe ahora|tiempo limitado)",
                r"(urgent|immédiat|agissez maintenant|temps limité)"
            ],
            "verification": [
                r"(verify|confirm|update).*(account|identity|information)",
                r"(verificar|confirmar|actualizar).*(cuenta|identidad|información)",
                r"(vérifier|confirmer|mettre à jour).*(compte|identité|informations)"
            ],
            "threats": [
                r"(suspend|close|terminate|arrest|legal action)",
                r"(suspender|cerrar|terminar|arrestar|acción legal)",
                r"(suspendre|fermer|terminer|arrêter|action légale)"
            ],
            "financial": [
                r"(bank account|credit card|social security|wire transfer)",
                r"(cuenta bancaria|tarjeta de crédito|seguro social|transferencia)",
                r"(compte bancaire|carte de crédit|sécurité sociale|virement)"
            ]
        }
    
    async def initialize(self):
        """Initialize the scam detector"""
        try:
            if settings.OPENAI_API_KEY:
                openai.api_key = settings.OPENAI_API_KEY
                self.openai_client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info("OpenAI client initialized")
            else:
                logger.warning("OpenAI API key not provided, using pattern matching only")
            
            # Compile regex patterns
            self._compile_patterns()
            logger.info("Scam detector initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize scam detector: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        self.openai_client = None
        logger.info("Scam detector cleaned up")
    
    async def health_check(self) -> bool:
        """Check if the service is healthy"""
        return True  # Pattern matching always works, OpenAI is optional
    
    def _compile_patterns(self):
        """Compile regex patterns for each language"""
        for pattern_type, patterns in self.scam_patterns_templates.items():
            self.scam_patterns[pattern_type] = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                for pattern in patterns
            ]
    
    async def detect_language(self, text: str) -> str:
        """Detect the language of the text"""
        if not text or len(text.strip()) < 10:
            return settings.DEFAULT_LANGUAGE
        
        # Simple language detection based on keywords
        text_lower = text.lower()
        
        # Count language-specific words
        lang_scores = {}
        for lang, keywords in self.scam_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            lang_scores[lang] = score
        
        # Return language with highest score, default to English
        if lang_scores:
            detected_lang = max(lang_scores, key=lang_scores.get)
            if lang_scores[detected_lang] > 0:
                return detected_lang
        
        return settings.DEFAULT_LANGUAGE
    
    async def analyze_text(
        self,
        text: str,
        language: str = "en",
        session_id: str = None
    ) -> ScamAnalysis:
        """
        Analyze text for scam indicators
        """
        try:
            if not text or len(text.strip()) < 5:
                return ScamAnalysis(
                    session_id=session_id or "",
                    text=text,
                    language=language,
                    risk_score=0.0,
                    confidence=1.0,
                    indicators=[],
                    timestamp=datetime.utcnow()
                )
            
            # Pattern-based detection
            pattern_score, pattern_indicators = await self._pattern_based_detection(text, language)
            
            # LLM-based detection (if available)
            llm_score, llm_reasoning, llm_indicators = await self._llm_based_detection(text, language)
            
            # Combine scores
            combined_score = max(pattern_score, llm_score)
            
            # Determine confidence
            confidence = self._calculate_confidence(pattern_score, llm_score, text)
            
            # Combine indicators
            all_indicators = list(set(pattern_indicators + llm_indicators))
            
            return ScamAnalysis(
                session_id=session_id or "",
                text=text,
                language=language,
                risk_score=min(combined_score, 1.0),
                confidence=confidence,
                indicators=all_indicators,
                reasoning=llm_reasoning,
                detected_patterns=pattern_indicators,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return ScamAnalysis(
                session_id=session_id or "",
                text=text,
                language=language,
                risk_score=0.0,
                confidence=0.0,
                indicators=["analysis_error"],
                timestamp=datetime.utcnow()
            )
    
    async def _pattern_based_detection(self, text: str, language: str) -> tuple[float, List[str]]:
        """Detect scam patterns using regex and keyword matching"""
        indicators = []
        total_score = 0.0
        
        # Check for scam keywords
        text_lower = text.lower()
        keywords = self.scam_keywords.get(language, self.scam_keywords["en"])
        
        keyword_matches = [kw for kw in keywords if kw in text_lower]
        if keyword_matches:
            indicators.extend([f"keyword_{kw}" for kw in keyword_matches[:5]])
            total_score += min(len(keyword_matches) * 0.1, 0.5)
        
        # Check regex patterns
        for pattern_type, patterns in self.scam_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    indicators.append(f"pattern_{pattern_type}")
                    total_score += 0.2
                    break  # Only count each pattern type once
        
        # Additional heuristics
        
        # Check for phone numbers and suspicious formatting
        phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
        if phone_pattern.search(text):
            indicators.append("phone_number_request")
            total_score += 0.15
        
        # Check for URLs or suspicious links
        url_pattern = re.compile(r'http[s]?://|www\.|\.com|\.org|click here|cliquez ici|haga clic')
        if url_pattern.search(text_lower):
            indicators.append("suspicious_link")
            total_score += 0.2
        
        # Check for money/financial mentions
        money_patterns = [
            r'\$\d+', r'money', r'payment', r'transfer', r'account number',
            r'dinero', r'pago', r'transferencia', r'número de cuenta',
            r'argent', r'paiement', r'transfert', r'numéro de compte'
        ]
        
        for pattern in money_patterns:
            if re.search(pattern, text_lower):
                indicators.append("financial_request")
                total_score += 0.25
                break
        
        # Check for identity theft indicators
        identity_patterns = [
            r'social security', r'ssn', r'date of birth', r'mother\'s maiden',
            r'seguro social', r'fecha de nacimiento', r'nombre de soltera',
            r'sécurité sociale', r'date de naissance', r'nom de jeune fille'
        ]
        
        for pattern in identity_patterns:
            if re.search(pattern, text_lower):
                indicators.append("identity_theft")
                total_score += 0.3
                break
        
        return min(total_score, 1.0), indicators
    
    async def _llm_based_detection(self, text: str, language: str) -> tuple[float, str, List[str]]:
        """Use LLM for sophisticated scam detection"""
        if not self.openai_client:
            return 0.0, "", []
        
        try:
            # Create language-appropriate prompt
            prompts = {
                "en": f"""
                Analyze the following phone call transcript for scam indicators. 
                Rate the scam likelihood from 0.0 (definitely not a scam) to 1.0 (definitely a scam).
                
                Look for:
                - Urgency tactics
                - Requests for personal information
                - Threats or pressure
                - Too-good-to-be-true offers
                - Impersonation of authorities/companies
                - Financial manipulation
                
                Transcript: "{text}"
                
                Respond in JSON format:
                {{
                    "scam_score": 0.0-1.0,
                    "reasoning": "explanation",
                    "indicators": ["list", "of", "specific", "indicators"]
                }}
                """,
                "es": f"""
                Analiza la siguiente transcripción de llamada telefónica en busca de indicadores de estafa.
                Califica la probabilidad de estafa de 0.0 (definitivamente no es estafa) a 1.0 (definitivamente es estafa).
                
                Busca:
                - Tácticas de urgencia
                - Solicitudes de información personal
                - Amenazas o presión
                - Ofertas demasiado buenas para ser verdad
                - Suplantación de autoridades/empresas
                - Manipulación financiera
                
                Transcripción: "{text}"
                
                Responde en formato JSON:
                {{
                    "scam_score": 0.0-1.0,
                    "reasoning": "explicación",
                    "indicators": ["lista", "de", "indicadores", "específicos"]
                }}
                """,
                "fr": f"""
                Analysez la transcription d'appel téléphonique suivante pour les indicateurs d'escroquerie.
                Évaluez la probabilité d'escroquerie de 0.0 (définitivement pas une escroquerie) à 1.0 (définitivement une escroquerie).
                
                Recherchez:
                - Tactiques d'urgence
                - Demandes d'informations personnelles
                - Menaces ou pression
                - Offres trop belles pour être vraies
                - Usurpation d'identité d'autorités/entreprises
                - Manipulation financière
                
                Transcription: "{text}"
                
                Répondez au format JSON:
                {{
                    "scam_score": 0.0-1.0,
                    "reasoning": "explication",
                    "indicators": ["liste", "des", "indicateurs", "spécifiques"]
                }}
                """
            }
            
            prompt = prompts.get(language, prompts["en"])
            
            response = await self.openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert scam detection AI."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content
            
            # Parse JSON response
            import json
            try:
                result = json.loads(result_text)
                return (
                    float(result.get("scam_score", 0.0)),
                    result.get("reasoning", ""),
                    result.get("indicators", [])
                )
            except json.JSONDecodeError:
                # Fallback parsing
                score_match = re.search(r'"scam_score":\s*([0-9.]+)', result_text)
                score = float(score_match.group(1)) if score_match else 0.0
                return score, result_text, []
            
        except Exception as e:
            logger.error(f"LLM-based detection failed: {e}")
            return 0.0, "", []
    
    def _calculate_confidence(self, pattern_score: float, llm_score: float, text: str) -> float:
        """Calculate confidence in the analysis"""
        # Base confidence on text length and analysis agreement
        text_length_factor = min(len(text) / 100, 1.0)  # Longer text = higher confidence
        
        if pattern_score == 0 and llm_score == 0:
            # Both agree it's not a scam
            return 0.9 * text_length_factor
        
        if abs(pattern_score - llm_score) < 0.2:
            # Scores are close, high confidence
            return 0.9 * text_length_factor
        elif abs(pattern_score - llm_score) < 0.4:
            # Moderate agreement
            return 0.7 * text_length_factor
        else:
            # Disagreement, lower confidence
            return 0.5 * text_length_factor
    
    async def analyze_conversation_context(
        self,
        conversation_history: List[str],
        language: str = "en"
    ) -> ScamAnalysis:
        """Analyze entire conversation context for scam patterns"""
        # Combine all conversation text
        full_text = " ".join(conversation_history)
        
        # Analyze combined text
        analysis = await self.analyze_text(full_text, language)
        
        # Additional context-specific analysis
        if len(conversation_history) > 1:
            # Check for escalation patterns
            if self._detect_escalation(conversation_history):
                analysis.indicators.append("escalation_pattern")
                analysis.risk_score = min(analysis.risk_score + 0.2, 1.0)
        
        return analysis
    
    def _detect_escalation(self, conversation_history: List[str]) -> bool:
        """Detect if conversation shows escalation patterns typical of scams"""
        if len(conversation_history) < 2:
            return False
        
        # Simple escalation detection based on urgency keywords
        urgency_keywords = ["urgent", "now", "immediate", "quickly", "hurry"]
        
        early_urgency = sum(1 for kw in urgency_keywords 
                           if kw in conversation_history[0].lower())
        late_urgency = sum(1 for kw in urgency_keywords 
                          if kw in conversation_history[-1].lower())
        
        return late_urgency > early_urgency
