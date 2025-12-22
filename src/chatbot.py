# src/health_chatbot/chatbot.py
"""
Main chatbot orchestration module.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Generator, Optional
from uuid import uuid4

from .config import settings
from .llm_client import LLMClient, get_llm_client, LLMClientError
from .prompts import PromptManager, ResponseTone, get_prompt_manager
from .safety import SafetyFilter, ContentFlag, RiskLevel, create_safety_filter
from .utils import sanitize_input, setup_logging

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a chat message."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = field(default_factory=lambda: str(uuid4())[:8])
    metadata: dict = field(default_factory=dict)


@dataclass
class ChatResponse:
    """Response from the chatbot."""
    content: str
    success: bool
    risk_level: RiskLevel
    flags: list[ContentFlag]
    was_filtered: bool = False
    error_message: Optional[str] = None


class HealthChatbot:
    """
    Main chatbot class that orchestrates all components.
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        prompt_manager: Optional[PromptManager] = None,
        safety_filter: Optional[SafetyFilter] = None,
        tone: ResponseTone = ResponseTone.FRIENDLY
    ):
        """Initialize the health chatbot."""
        self.llm_client = llm_client or get_llm_client()
        self.prompt_manager = prompt_manager or get_prompt_manager(tone)
        self.safety_filter = safety_filter or create_safety_filter(
            enabled=settings.enable_safety_filter
        )
        
        # Conversation history
        self.history: list[Message] = []
        self.max_history = 10  # Keep last N exchanges
        
        # Setup logging
        setup_logging(settings.log_level.value)
        
        logger.info(f"HealthChatbot initialized with {type(self.llm_client).__name__}")
    
    def chat(self, user_input: str) -> ChatResponse:
        """Process a user message and return a response."""
        try:
            # Step 1: Sanitize input
            clean_input = sanitize_input(user_input)
            if not clean_input:
                return ChatResponse(
                    content="I didn't catch that. Could you please rephrase your question?",
                    success=False,
                    risk_level=RiskLevel.LOW,
                    flags=[]
                )
            
            logger.info(f"Processing query: {clean_input[:50]}...")
            
            # Step 2: Safety check on input
            input_check = self.safety_filter.check_input(clean_input)
            
            # Step 3: Handle emergencies immediately
            if input_check.risk_level == RiskLevel.EMERGENCY:
                emergency_response = self.safety_filter.get_emergency_response(
                    input_check.flags
                )
                self._add_to_history("user", clean_input)
                self._add_to_history("assistant", emergency_response)
                
                return ChatResponse(
                    content=emergency_response,
                    success=True,
                    risk_level=RiskLevel.EMERGENCY,
                    flags=input_check.flags,
                    was_filtered=True
                )
            
            # Step 4: Format messages for LLM
            history_messages = self._get_history_messages()
            messages = self.prompt_manager.format_conversation(
                clean_input,
                history=history_messages
            )
            
            # Step 5: Generate response from LLM
            try:
                response_text = self.llm_client.generate(
                    messages=messages,
                    temperature=settings.temperature,
                    max_tokens=settings.max_tokens
                )
            except LLMClientError as e:
                logger.error(f"LLM error: {e}")
                return ChatResponse(
                    content="I'm having trouble connecting to my brain right now. "
                           "Please try again in a moment.",
                    success=False,
                    risk_level=RiskLevel.LOW,
                    flags=[],
                    error_message=str(e)
                )
            
            # Step 6: Safety check on output
            output_check = self.safety_filter.check_output(response_text)
            
            # Step 7: Add appropriate disclaimers
            final_response = self.safety_filter.add_disclaimer(
                response_text,
                max(input_check.risk_level, output_check.risk_level)
            )
            
            # Step 8: Update conversation history
            self._add_to_history("user", clean_input)
            self._add_to_history("assistant", final_response)
            
            return ChatResponse(
                content=final_response,
                success=True,
                risk_level=max(input_check.risk_level, output_check.risk_level),
                flags=list(set(input_check.flags + output_check.flags)),
                was_filtered=not output_check.is_safe
            )
            
        except Exception as e:
            logger.exception(f"Unexpected error in chat: {e}")
            return ChatResponse(
                content="I encountered an unexpected error. Please try again.",
                success=False,
                risk_level=RiskLevel.LOW,
                flags=[],
                error_message=str(e)
            )
    
    def chat_stream(self, user_input: str) -> Generator[str, None, ChatResponse]:
        """Stream a response to the user."""
        # Safety check first
        clean_input = sanitize_input(user_input)
        input_check = self.safety_filter.check_input(clean_input)
        
        if input_check.risk_level == RiskLevel.EMERGENCY:
            emergency_response = self.safety_filter.get_emergency_response(
                input_check.flags
            )
            yield emergency_response
            return ChatResponse(
                content=emergency_response,
                success=True,
                risk_level=RiskLevel.EMERGENCY,
                flags=input_check.flags,
                was_filtered=True
            )
        
        # Format and stream
        history_messages = self._get_history_messages()
        messages = self.prompt_manager.format_conversation(
            clean_input,
            history=history_messages
        )
        
        full_response = ""
        try:
            for chunk in self.llm_client.generate_stream(
                messages=messages,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens
            ):
                full_response += chunk
                yield chunk
            
            # Add disclaimer at the end
            disclaimer = self.safety_filter.add_disclaimer("", input_check.risk_level)
            if disclaimer:
                yield disclaimer
                full_response += disclaimer
            
            # Update history
            self._add_to_history("user", clean_input)
            self._add_to_history("assistant", full_response)
            
            return ChatResponse(
                content=full_response,
                success=True,
                risk_level=input_check.risk_level,
                flags=input_check.flags
            )
            
        except Exception as e:
            error_msg = f"\n\n*Error occurred: {e}*"
            yield error_msg
            return ChatResponse(
                content=full_response + error_msg,
                success=False,
                risk_level=RiskLevel.LOW,
                flags=[],
                error_message=str(e)
            )
    
    def _add_to_history(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self.history.append(Message(role=role, content=content))
        
        # Trim history if too long
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]
    
    def _get_history_messages(self) -> list[dict]:
        """Get history formatted for LLM API."""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.history[-self.max_history * 2:]
        ]
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history = []
        logger.info("Conversation history cleared")
    
    def get_history(self) -> list[Message]:
        """Get the conversation history."""
        return self.history.copy()


def main():
    """
    CLI interface for the chatbot.
    
    Run with: uv run health-chat
    """
    import sys
    
    print("=" * 50)
    print("🏥 Health Assistant Chatbot")
    print("=" * 50)
    print("Type your health questions, or 'quit' to exit.")
    print("Type 'clear' to reset the conversation.")
    print("-" * 50)
    
    chatbot = HealthChatbot()
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 Take care! Remember to consult healthcare professionals for medical advice.")
                break
            
            if user_input.lower() == "clear":
                chatbot.clear_history()
                print("🔄 Conversation cleared.")
                continue
            
            print("\n🤖 Assistant: ", end="", flush=True)
            
            # Use streaming for better UX
            for chunk in chatbot.chat_stream(user_input):
                print(chunk, end="", flush=True)
            
            print()  # Newline after response
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    return 0


if __name__ == "__main__":
    main()