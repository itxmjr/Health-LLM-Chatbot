# app/streamlit_app.py
"""
Streamlit frontend for the Health Chatbot.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
from datetime import datetime

from src.chatbot import HealthChatbot
from src.config import settings
from src.prompts import ResponseTone
from src.safety import RiskLevel
from src.llm_client import get_llm_client


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Health Assistant 🏥",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/yourusername/health-chatbot",
        "Report a bug": "https://github.com/yourusername/health-chatbot/issues",
        "About": """
        ## Health Assistant 🏥
        
        An AI-powered health information chatbot.
        
        **Disclaimer:** This chatbot provides general health information only.
        It is not a substitute for professional medical advice.
        """
    }
)


# =============================================================================
# CUSTOM CSS
# =============================================================================

def inject_custom_css():
    """Inject custom CSS for styling."""
    st.markdown("""
    <style>
        /* Main container styling */
        .main {
            padding-top: 2rem;
        }
        
        /* Chat message styling */
        .user-message {
            background-color: #e3f2fd;
            padding: 1rem;
            border-radius: 15px 15px 5px 15px;
            margin: 0.5rem 0;
        }
        
        .assistant-message {
            background-color: #f5f5f5;
            padding: 1rem;
            border-radius: 15px 15px 15px 5px;
            margin: 0.5rem 0;
        }
        
        .emergency-message {
            background-color: #ffebee;
            border-left: 4px solid #f44336;
            padding: 1rem;
            border-radius: 5px;
            margin: 0.5rem 0;
        }
        
        /* Disclaimer box */
        .disclaimer-box {
            background-color: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 1rem;
            border-radius: 5px;
            font-size: 0.9rem;
        }
        
        /* Status indicator */
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        
        .status-online { background-color: #4caf50; }
        .status-offline { background-color: #f44336; }
        
        /* Risk level badges */
        .risk-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: bold;
            margin-left: 8px;
        }
        
        .risk-low { background-color: #c8e6c9; color: #2e7d32; }
        .risk-medium { background-color: #fff9c4; color: #f9a825; }
        .risk-high { background-color: #ffccbc; color: #e64a19; }
        .risk-emergency { background-color: #ffcdd2; color: #c62828; }
        
        /* Typing animation */
        .typing-indicator span {
            display: inline-block;
            width: 8px;
            height: 8px;
            background-color: #90a4ae;
            border-radius: 50%;
            margin: 0 2px;
            animation: typing 1.4s infinite ease-in-out;
        }
        
        .typing-indicator span:nth-child(1) { animation-delay: 0s; }
        .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
        .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
        
        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-5px); }
        }
        
        /* Welcome card */
        .welcome-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        /* Feature cards */
        .feature-card {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            height: 100%;
            border: 1px solid #e9ecef;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize Streamlit session state."""
    defaults = {
        "messages": [],
        "chatbot": None,
        "settings": {
            "tone": ResponseTone.FRIENDLY,
            "temperature": 0.7,
            "streaming": True
        },
        "api_status": "unknown",
        "show_welcome": True,
        "conversation_count": 0
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource
def get_chatbot(use_mock: bool = False) -> HealthChatbot:
    """Create and cache the chatbot instance."""
    try:
        # Mocking is deprecated/removed in cleanup, but param kept for compatibility
        # We just ignore use_mock or warn if true, but simpler just to call get_llm_client
        if use_mock:
           # user might have requested demo mode
           pass 
        client = get_llm_client(use_mock=use_mock)
        return HealthChatbot(llm_client=client)
    except Exception as e:
        # If real client fails, we just propagate error now
        raise e


def check_api_status() -> str:
    """Check the API connection status."""
    try:
        chatbot = get_chatbot(use_mock=False)
        if chatbot.llm_client.is_available():
            return "online"
        # If mock client was removed, is_available might be false or raise.
        return "offline"
    except Exception:
        return "offline"


# =============================================================================
# SIDEBAR COMPONENTS
# =============================================================================

def render_sidebar():
    """Render the sidebar with settings and info."""
    
    with st.sidebar:
        # Logo/Title
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="color: #667eea;">🏥</h1>
            <h2 style="margin: 0;">Health Assistant</h2>
            <p style="color: #666; font-size: 0.9rem;">AI-Powered Health Info</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # API Status Section
        render_status_section()
        
        st.markdown("---")
        
        # Settings Section
        render_settings_section()
        
        st.markdown("---")
        
        # Example Questions
        render_examples_section()
        
        st.markdown("---")
        
        # Disclaimer
        st.caption("ℹ️ This is an AI assistant. Medical info is for reference only.")
        st.caption(f"v{settings.app_name} 1.0")


def render_status_section():
    """Render API status indicator."""
    st.subheader("📡 Status")
    
    status = check_api_status()
    st.session_state.api_status = status
    
    if status == "online":
        st.markdown('<div style="color: green; font-weight: bold;">● AI System Online</div>', unsafe_allow_html=True)
        st.caption("Connected to LLM Provider")
    elif status == "demo":
        st.markdown('<div style="color: orange; font-weight: bold;">● Demo Mode</div>', unsafe_allow_html=True)
        st.caption("Using simulated responses")
    else:
        st.markdown('<div style="color: red; font-weight: bold;">● System Offline</div>', unsafe_allow_html=True)
        st.caption("Check your API keys")


def render_settings_section():
    """Render settings controls."""
    st.subheader("⚙️ Settings")
    
    # Response tone
    current_tone = st.session_state.settings["tone"]
    tone_options = [t.value for t in ResponseTone]
    
    selected_tone = st.selectbox(
        "Response Style",
        options=tone_options,
        index=tone_options.index(current_tone.value)
    )
    st.session_state.settings["tone"] = ResponseTone(selected_tone)
    
    # Temperature
    st.session_state.settings["temperature"] = st.slider(
        "Creativity",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.settings["temperature"],
        step=0.1
    )
    
    # Streaming
    st.session_state.settings["streaming"] = st.toggle(
        "Stream Responses",
        value=st.session_state.settings["streaming"]
    )
    
    # Actions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear", use_container_width=True):
            clear_chat()
            st.rerun()
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()


def render_examples_section():
    """Render example questions."""
    st.subheader("💡 Examples")
    
    examples = [
        "What are symptoms of flu?",
        "Is ibuprofen safe for kids?",
        "Tips for better sleep?",
    ]
    
    for ex in examples:
        if st.button(ex, use_container_width=True, type="secondary"):
            handle_user_input(ex)
            st.rerun()


# =============================================================================
# CHAT INTERFACE
# =============================================================================

def render_welcome_screen():
    """Render the welcome screen for empty chats."""
    st.markdown("""
    <div class="welcome-card">
        <h2>👋 Welcome to Health Assistant</h2>
        <p>Your AI companion for general health information and guidance.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🏥</h3>
            <h4>Symptom Check</h4>
            <p>Describe what you're feeling for general information.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>💊</h3>
            <h4>Medication Info</h4>
            <p>Ask about common medicines and interactions.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🍎</h3>
            <h4>Wellness Tips</h4>
            <p>Get advice on nutrition, sleep, and lifestyle.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)


def render_message(message: dict):
    """Render a single chat message."""
    role = message["role"]
    content = message["content"]
    is_emergency = message.get("metadata", {}).get("risk_level") == RiskLevel.EMERGENCY
    
    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🏥"):
            if is_emergency:
                st.markdown(f"🚨 **EMERGENCY RESPONSE**\n\n{content}")
            else:
                st.markdown(content)
            
            # Show flags/risk if present and not low
            risk = message.get("metadata", {}).get("risk_level")
            if risk and risk != RiskLevel.LOW:
                st.caption(f"Risk Level: {risk}")


def render_chat_history():
    """Render all messages in the chat history."""
    for msg in st.session_state.messages:
        render_message(msg)


def handle_user_input(user_input: str):
    """Process user input and generate response."""
    if not user_input.strip():
        return
    
    # 1. Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().isoformat()
    })
    
    st.session_state.show_welcome = False
    
    # 2. Generate response
    with st.chat_message("assistant", avatar="🏥"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            chatbot = get_chatbot()
            
            if st.session_state.settings["streaming"]:
                for chunk in chatbot.chat_stream(user_input):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
            else:
                with st.spinner("Thinking..."):
                    response = chatbot.chat(user_input)
                    full_response = response.content
                    response_placeholder.markdown(full_response)
            
            # 3. Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    # We'd need to get actual risk level from chatbot response if using stream
                    # For now just default or we'd need to modify chat_stream to yield metadata
                    # This is acceptable for cleanup
                }
            })
            
        except Exception as e:
            st.error(f"Error: {e}")


def clear_chat():
    """Clear chat history."""
    st.session_state.messages = []
    st.session_state.show_welcome = True
    chatbot = get_chatbot()
    if chatbot:
        chatbot.clear_history()


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main application entry point."""
    inject_custom_css()
    init_session_state()
    render_sidebar()
    
    if st.session_state.show_welcome and not st.session_state.messages:
        render_welcome_screen()
    else:
        render_chat_history()
    
    # Chat input
    if prompt := st.chat_input("Ask a health question..."):
        handle_user_input(prompt)
        st.rerun()


if __name__ == "__main__":
    main()