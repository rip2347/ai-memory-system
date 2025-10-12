"""
LLM Abstraction Layer
Provides a unified interface to interact with different LLM providers.
Currently supports Ollama, but can be extended to OpenAI, Anthropic, etc.
"""

import requests
import json
from typing import Optional, Dict, Any
from config import LLM_PROVIDER, MODEL_NAME, OLLAMA_URL, TEMPERATURE, MAX_TOKENS


def ask_llm(prompt: str, system_prompt: Optional[str] = None, 
            temperature: Optional[float] = None, 
            max_tokens: Optional[int] = None,
            model_override: Optional[str] = None) -> str:
    """
    Send a prompt to the LLM and get a response.
    
    Args:
        prompt: The user prompt/question
        system_prompt: Optional system instructions for the LLM
        temperature: Optional temperature override (0.0-2.0, lower = more focused)
        max_tokens: Optional max tokens override
        model_override: Optional model name override (use different model)
    
    Returns:
        str: The LLM's response text
    
    Raises:
        Exception: If the LLM call fails
    """
    
    # Use config defaults if not specified
    temp = temperature if temperature is not None else TEMPERATURE
    max_tok = max_tokens if max_tokens is not None else MAX_TOKENS
    model = model_override if model_override is not None else MODEL_NAME
    
    if LLM_PROVIDER == "ollama":
        return _ask_ollama(prompt, system_prompt, temp, max_tok, model)
    elif LLM_PROVIDER == "openai":
        # TODO: Implement OpenAI
        raise NotImplementedError("OpenAI provider not yet implemented")
    elif LLM_PROVIDER == "anthropic":
        # TODO: Implement Anthropic
        raise NotImplementedError("Anthropic provider not yet implemented")
    else:
        raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}")


def _ask_ollama(prompt: str, system_prompt: Optional[str], 
                temperature: float, max_tokens: int, model: str) -> str:
    """
    Call Ollama API.
    
    Ollama API docs: https://github.com/ollama/ollama/blob/main/docs/api.md
    """
    
    url = f"{OLLAMA_URL}/api/generate"
    
    # Build the full prompt with system instructions if provided
    full_prompt = prompt
    if system_prompt:
        full_prompt = f"{system_prompt}\n\n{prompt}"
    
    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,  # Get complete response at once
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "").strip()
        
    except requests.exceptions.ConnectionError:
        raise Exception(
            f"Could not connect to Ollama at {OLLAMA_URL}. "
            "Make sure Ollama is running (systemctl status ollama)"
        )
    except requests.exceptions.Timeout:
        raise Exception("Ollama request timed out. The model might be processing a complex prompt.")
    except requests.exceptions.HTTPError as e:
        raise Exception(f"Ollama API error: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error calling Ollama: {e}")


def test_connection() -> bool:
    """
    Test if we can connect to the LLM.
    
    Returns:
        bool: True if connection successful
    """
    try:
        if LLM_PROVIDER == "ollama":
            response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            response.raise_for_status()
            print(f"✓ Connected to Ollama at {OLLAMA_URL}")
            
            # List available models
            models = response.json().get("models", [])
            print(f"✓ Available models: {[m['name'] for m in models]}")
            
            # Check if our configured models are available
            model_names = [m['name'] for m in models]
            
            # Check chat model
            if MODEL_NAME in model_names:
                print(f"✓ Chat model '{MODEL_NAME}' is ready")
            else:
                print(f"⚠ Chat model '{MODEL_NAME}' not found. Run: ollama pull {MODEL_NAME}")
                return False
                        
            return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False


if __name__ == "__main__":
    """
    Test the LLM connection and basic functionality.
    Run: python llm.py
    """
    print("Testing LLM connection...")
    print("-" * 50)
    
    if test_connection():
        print("\nTesting basic query...")
        print("-" * 50)
        
        response = ask_llm("What is 2+2? Answer in one sentence.")
        print(f"Question: What is 2+2?")
        print(f"Response: {response}")
        
        print("\n✓ LLM is working!")
    else:
        print("\n✗ LLM connection failed. Check the errors above.")