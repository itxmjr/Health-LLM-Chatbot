"""
LLM Client implementation for Google Gemini.
"""

import logging
from typing import Generator

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import settings
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    """Base exception for LLM client errors."""
    pass


class LLMClient:
    """Client for Google Gemini API."""
    
    def __init__(self):
        """Initialize the Gemini client."""
        self.client = genai.Client(api_key=settings.gemini_api_key)
    
    def is_available(self) -> bool:
        """Check if the client is configured and available."""
        return bool(self.client)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """Generate a response from the LLM."""
        if not self.is_available():
            raise LLMClientError("Gemini API key not configured")
        
        try:
            # Gemini expects a prompt string or a list of Content objects
            # For simplicity, we'll concatenate the history into a chat session
            
            # Construct the prompt from messages
            
            # Construct the prompt from messages
            # Note: Gemini python lib handles chat history differently. 
            # We'll adapt our message list to a simple prompt string for now 
            # or use the proper chat history structure if we wanted to be more robust.
            # Given the format, let's just send the last user message with context.
            
            # Simple prompt construction
            prompt = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    prompt += f"System: {content}\n\n"
                elif role == "user":
                    prompt += f"User: {content}\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n"
            
            prompt += "Assistant: "
            
            response = self.client.models.generate_content(
                model=settings.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise LLMClientError(f"Failed to generate response: {e}")
    
    def generate_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Generator[str, None, None]:
        """Stream a response from the LLM."""
        if not self.is_available():
            raise LLMClientError("Gemini API key not configured")
        
        try:
            # Same prompt construction
            prompt = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    prompt += f"System: {content}\n\n"
                elif role == "user":
                    prompt += f"User: {content}\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n"
            
            prompt += "Assistant: "
            
            for chunk in self.client.models.generate_content_stream(
                model=settings.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            ):
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            raise LLMClientError(f"Streaming error: {e}")


def get_llm_client(use_mock: bool = False) -> LLMClient:
    """Factory function to get an LLM client."""
    logger.info(f"Using Gemini client with model: {settings.model_name}")
    return LLMClient()