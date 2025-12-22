# src/health_chatbot/utils.py
"""
Utility functions for the health chatbot.
"""

import logging
import re
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Configure standard logging for the application."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    logger.info(f"Logging configured at {log_level} level")


def sanitize_input(text: str) -> str:
    """Clean and sanitize user input."""
    if not text:
        return ""
    
    # Remove control characters (except newlines and tabs)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Limit length to prevent abuse
    max_length = 1000
    if len(text) > max_length:
        text = text[:max_length] + "..."
        logger.warning(f"Input truncated to {max_length} characters")
    
    return text.strip()


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """Format datetime for display."""
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to max length, adding suffix if truncated."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def count_tokens_approximate(text: str) -> int:
    """Approximate token count for a text string."""
    # Rough approximation: 1 token ≈ 4 characters
    return len(text) // 4


def create_error_response(error_type: str, message: str) -> dict:
    """Create a standardized error response."""
    return {
        "success": False,
        "error": {
            "type": error_type,
            "message": message
        },
        "timestamp": format_timestamp()
    }