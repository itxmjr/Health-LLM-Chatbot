# src/health_chatbot/config.py
"""
Configuration settings for the application.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # App Settings
    app_name: str = "Health LLM Chatbot"
    debug: bool = False
    log_level: str = "INFO"
    environment: str = "development"
    
    # API Keys
    gemini_api_key: Optional[str] = Field(default=None, validation_alias='GEMINI_API_KEY')
    
    # Model Configuration
    llm_provider: str = "gemini"
    model_name: str = "gemini-2.5-flash"
    max_tokens: int = 500
    temperature: float = 0.7
    
    # Safety Configuration
    enable_safety_filter: bool = True
    max_response_length: int = 1000
    
    def validate_api_keys(self) -> bool:
        """Check if required API keys are present."""
        return bool(self.gemini_api_key)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()