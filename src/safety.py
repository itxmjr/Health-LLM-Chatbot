# src/health_chatbot/safety.py
"""
Safety filters and content moderation for health chatbot.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Risk levels for health queries."""
    LOW = "low"           # General health info
    MEDIUM = "medium"     # Needs disclaimer
    HIGH = "high"         # Needs strong warning
    EMERGENCY = "emergency"  # Needs immediate action


class ContentFlag(str, Enum):
    """Types of flagged content."""
    EMERGENCY = "emergency"
    MEDICATION_REQUEST = "medication_request"
    DIAGNOSIS_REQUEST = "diagnosis_request"
    MENTAL_HEALTH_CRISIS = "mental_health_crisis"
    HARMFUL_INTENT = "harmful_intent"
    CHILD_SAFETY = "child_safety"


@dataclass
class SafetyCheck:
    """Result of a safety check."""
    is_safe: bool
    risk_level: RiskLevel
    flags: list[ContentFlag]
    message: Optional[str] = None
    modified_response: Optional[str] = None


class SafetyFilter:
    """Filters and checks content for safety."""
    
    # Emergency keywords that need immediate action
    EMERGENCY_KEYWORDS = [
        r"\b(chest\s*pain)\b",
        r"\b(can'?t\s*breathe|difficulty\s*breathing|shortness\s*of\s*breath)\b",
        r"\b(stroke|heart\s*attack)\b",
        r"\b(unconscious|passed\s*out|fainted)\b",
        r"\b(severe\s*bleeding|won'?t\s*stop\s*bleeding)\b",
        r"\b(overdose|poison)\b",
        r"\b(suicid|kill\s*(my)?self|end\s*(my)?\s*life)\b",
        r"\b(anaphyla|allergic\s*shock)\b",
        r"\b(seizure|convulsion)\b",
    ]
    
    # Keywords suggesting medication requests
    MEDICATION_KEYWORDS = [
        r"\b(prescribe|prescription)\b",
        r"\b(what\s*(medication|drug|medicine)\s*should\s*i\s*take)\b",
        r"\b(dosage|how\s*much\s*should\s*i\s*take)\b",
        r"\b(can\s*i\s*take|is\s*it\s*safe\s*to\s*take)\b.*\b(mg|milligram)\b",
    ]
    
    # Keywords suggesting diagnosis requests
    DIAGNOSIS_KEYWORDS = [
        r"\b(do\s*i\s*have|what\s*do\s*i\s*have)\b",
        r"\b(diagnose|diagnosis)\b",
        r"\b(is\s*this|is\s*it)\s*(cancer|serious|dangerous)\b",
        r"\b(what'?s\s*wrong\s*with\s*me)\b",
    ]
    
    # Mental health crisis keywords
    MENTAL_HEALTH_CRISIS_KEYWORDS = [
        r"\b(want\s*to\s*die|want\s*to\s*end\s*it)\b",
        r"\b(self[- ]?harm|cut(ting)?\s*myself)\b",
        r"\b(hopeless|no\s*point\s*(in\s*living)?)\b",
        r"\b(nobody\s*cares|better\s*off\s*dead)\b",
    ]
    
    # Potentially harmful requests
    HARMFUL_KEYWORDS = [
        r"\b(how\s*to\s*(hurt|harm|poison))\b",
        r"\b(dangerous\s*combination)\b",
        r"\b(make\s*(myself|me)\s*sick)\b",
    ]
    
    def __init__(self, enabled: bool = True):
        """Initialize the safety filter."""
        self.enabled = enabled
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        self._emergency_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.EMERGENCY_KEYWORDS
        ]
        self._medication_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.MEDICATION_KEYWORDS
        ]
        self._diagnosis_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.DIAGNOSIS_KEYWORDS
        ]
        self._mental_health_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.MENTAL_HEALTH_CRISIS_KEYWORDS
        ]
        self._harmful_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.HARMFUL_KEYWORDS
        ]
    
    def check_input(self, text: str) -> SafetyCheck:
        """Check user input for safety concerns."""
        if not self.enabled:
            return SafetyCheck(
                is_safe=True,
                risk_level=RiskLevel.LOW,
                flags=[]
            )
        
        flags = []
        risk_level = RiskLevel.LOW
        message = None
        
        # Check for emergencies (highest priority)
        if self._check_patterns(text, self._emergency_patterns):
            flags.append(ContentFlag.EMERGENCY)
            risk_level = RiskLevel.EMERGENCY
            logger.warning(f"Emergency keywords detected in input")
        
        # Check for mental health crisis
        if self._check_patterns(text, self._mental_health_patterns):
            flags.append(ContentFlag.MENTAL_HEALTH_CRISIS)
            if risk_level != RiskLevel.EMERGENCY:
                risk_level = RiskLevel.EMERGENCY
            logger.warning(f"Mental health crisis keywords detected")
        
        # Check for harmful intent
        if self._check_patterns(text, self._harmful_patterns):
            flags.append(ContentFlag.HARMFUL_INTENT)
            risk_level = max(risk_level, RiskLevel.HIGH, key=lambda x: list(RiskLevel).index(x))
            logger.warning(f"Harmful intent keywords detected")
        
        # Check for medication requests
        if self._check_patterns(text, self._medication_patterns):
            flags.append(ContentFlag.MEDICATION_REQUEST)
            if risk_level == RiskLevel.LOW:
                risk_level = RiskLevel.MEDIUM
        
        # Check for diagnosis requests
        if self._check_patterns(text, self._diagnosis_patterns):
            flags.append(ContentFlag.DIAGNOSIS_REQUEST)
            if risk_level == RiskLevel.LOW:
                risk_level = RiskLevel.MEDIUM
        
        return SafetyCheck(
            is_safe=risk_level not in [RiskLevel.EMERGENCY, RiskLevel.HIGH],
            risk_level=risk_level,
            flags=flags,
            message=message
        )
    
    def _check_patterns(self, text: str, patterns: list) -> bool:
        """Check if any pattern matches the text."""
        return any(pattern.search(text) for pattern in patterns)
    
    def check_output(self, response: str) -> SafetyCheck:
        """Check LLM output for safety issues."""
        if not self.enabled:
            return SafetyCheck(is_safe=True, risk_level=RiskLevel.LOW, flags=[])
        
        flags = []
        risk_level = RiskLevel.LOW
        
        # Check for dosage recommendations
        dosage_pattern = re.compile(
            r'\b(take|use)\s+\d+\s*(mg|ml|tablet|pill|capsule)',
            re.IGNORECASE
        )
        if dosage_pattern.search(response):
            flags.append(ContentFlag.MEDICATION_REQUEST)
            risk_level = RiskLevel.MEDIUM
            logger.warning("Response contains dosage recommendation")
        
        # Check for definitive diagnoses
        diagnosis_pattern = re.compile(
            r'\b(you\s+(have|definitely\s+have|probably\s+have))\s+\w+',
            re.IGNORECASE
        )
        if diagnosis_pattern.search(response):
            flags.append(ContentFlag.DIAGNOSIS_REQUEST)
            risk_level = RiskLevel.MEDIUM
            logger.warning("Response contains diagnostic statement")
        
        return SafetyCheck(
            is_safe=len(flags) == 0,
            risk_level=risk_level,
            flags=flags
        )
    
    def get_emergency_response(self, flags: list[ContentFlag]) -> str:
        """Get appropriate emergency response based on flags."""
        if ContentFlag.MENTAL_HEALTH_CRISIS in flags:
            return """🆘 **I'm concerned about you.**

If you're having thoughts of suicide or self-harm, please reach out for help right now:

**National Suicide Prevention Lifeline:** 988 (US)
**Crisis Text Line:** Text HOME to 741741
**International Association for Suicide Prevention:** https://www.iasp.info/resources/Crisis_Centres/

You're not alone, and there are people who want to help. Please talk to someone. 💙"""
        
        if ContentFlag.EMERGENCY in flags:
            return """🚨 **This sounds like it could be a medical emergency.**

Please **call 911** (or your local emergency number) or **go to the nearest emergency room immediately**.

If someone is with you, have them help you get emergency care.

Do not wait to see if symptoms improve. Time is critical in medical emergencies.

**Emergency contacts:**
- 🇺🇸 US: 911
- 🇬🇧 UK: 999
- 🇪🇺 EU: 112
- 🇦🇺 AU: 000"""
        
        return ""
    
    def add_disclaimer(self, response: str, risk_level: RiskLevel) -> str:
        """Add appropriate disclaimer to response based on risk level."""
        disclaimers = {
            RiskLevel.LOW: "",
            RiskLevel.MEDIUM: "\n\n---\n📋 *Remember: This is general information only. Please consult a healthcare provider for personalized medical advice.*",
            RiskLevel.HIGH: "\n\n---\n⚠️ **Important:** This information is not a substitute for professional medical advice. Please consult a doctor or healthcare provider, especially if your symptoms are severe or concerning.",
            RiskLevel.EMERGENCY: ""  # Emergency responses have their own format
        }
        
        disclaimer = disclaimers.get(risk_level, "")
        if disclaimer and disclaimer not in response:
            return response + disclaimer
        return response


def create_safety_filter(enabled: bool = True) -> SafetyFilter:
    """Factory function to create a configured safety filter."""
    return SafetyFilter(enabled=enabled)