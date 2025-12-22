import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

try:
    from src.config import settings
    print(f"Settings loaded.")
    print(f"GEMINI_API_KEY in settings: {'Set' if settings.gemini_api_key else 'Not Set'}")
    
    # Check env vars directly
    print(f"Environment variables:")
    for key in os.environ:
        if "API_KEY" in key:
            print(f"  {key}: {'Set' if os.environ[key] else 'Empty'}")

    from src.llm_client import get_llm_client, LLMClientError
    
    print("Initializing LLMClient...")
    client = get_llm_client()
    print(f"Client available: {client.is_available()}")
    
    # List models
    try:
        print("Listing available models...")
        for model in client.client.models.list(config={"page_size": 10}):
            print(f" - {model.name}")
    except Exception as e:
        print(f"Failed to list models: {e}")

    print("Attempting generation...")
    try:
        response = client.generate([{"role": "user", "content": "Hello"}])
        print(f"Generation successful. Response: {response[:50]}...")
    except Exception as e:
        print(f"Generation failed: {e}")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
