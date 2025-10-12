"""
Together API Configuration for LanguageAgentTreeSearch

This module provides centralized configuration for using Together API
instead of OpenAI API in the LATS implementation.
"""

import os

# Together API Configuration
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
TOGETHER_API_BASE = "https://api.together.xyz/v1"

# Fallback to OpenAI if Together key not available
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "")

# Model mapping from OpenAI model names to Together API models
TOGETHER_MODELS = {
    # OpenAI models -> Together models
    "gpt-3.5-turbo": "meta-llama/Llama-2-7b-chat-hf",
    "gpt-4": "meta-llama/Llama-2-70b-chat-hf", 
    "gpt-3.5-turbo-16k": "meta-llama/Llama-2-7b-chat-hf",
    
    # Direct Together model names (for direct usage)
    "llama-2-7b": "meta-llama/Llama-2-7b-chat-hf",
    "llama-2-13b": "meta-llama/Llama-2-13b-chat-hf", 
    "llama-2-70b": "meta-llama/Llama-2-70b-chat-hf",
    "codellama-7b": "codellama/CodeLlama-7b-Instruct-hf",
    "codellama-13b": "codellama/CodeLlama-13b-Instruct-hf",
    "codellama-34b": "codellama/CodeLlama-34b-Instruct-hf",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.1",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "qwen-7b": "Qwen/Qwen-7B-Chat",
    "qwen-14b": "Qwen/Qwen-14B-Chat",
    "qwen-32b": "Qwen/Qwen-32B-Chat",
    "claude-3": "anthropic/claude-3-sonnet"
}

# Together API pricing (per 1M tokens)
TOGETHER_PRICING = {
    "meta-llama/Llama-2-7b-chat-hf": {"input": 0.0002, "output": 0.0002},
    "meta-llama/Llama-2-13b-chat-hf": {"input": 0.0002, "output": 0.0002}, 
    "meta-llama/Llama-2-70b-chat-hf": {"input": 0.0007, "output": 0.0007},
    "codellama/CodeLlama-7b-Instruct-hf": {"input": 0.0002, "output": 0.0002},
    "codellama/CodeLlama-13b-Instruct-hf": {"input": 0.0002, "output": 0.0002},
    "codellama/CodeLlama-34b-Instruct-hf": {"input": 0.0007, "output": 0.0007},
    "mistralai/Mistral-7B-Instruct-v0.1": {"input": 0.0002, "output": 0.0002},
    "mistralai/Mixtral-8x7B-Instruct-v0.1": {"input": 0.00027, "output": 0.00027},
    "Qwen/Qwen-7B-Chat": {"input": 0.0002, "output": 0.0002},
    "Qwen/Qwen-14B-Chat": {"input": 0.0002, "output": 0.0002},
    "Qwen/Qwen-32B-Chat": {"input": 0.0007, "output": 0.0007}
}

def get_api_config():
    """Get the appropriate API configuration based on available keys."""
    if TOGETHER_API_KEY:
        return {
            "api_key": TOGETHER_API_KEY,
            "api_base": TOGETHER_API_BASE,
            "provider": "together"
        }
    elif OPENAI_API_KEY:
        return {
            "api_key": OPENAI_API_KEY,
            "api_base": OPENAI_API_BASE or "https://api.openai.com/v1",
            "provider": "openai"
        }
    else:
        raise ValueError("Neither TOGETHER_API_KEY nor OPENAI_API_KEY is set")

def get_model_name(openai_model_name):
    """Map OpenAI model name to Together API model name."""
    return TOGETHER_MODELS.get(openai_model_name, openai_model_name)

def calculate_cost(model_name, completion_tokens, prompt_tokens):
    """Calculate cost based on Together API pricing."""
    together_model = get_model_name(model_name)
    
    if together_model in TOGETHER_PRICING:
        pricing = TOGETHER_PRICING[together_model]
        cost = (completion_tokens / 1000000 * pricing["output"] + 
                prompt_tokens / 1000000 * pricing["input"])
        return cost
    else:
        # Fallback to OpenAI pricing
        if model_name == "gpt-4":
            return completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
        elif model_name == "gpt-3.5-turbo":
            return completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
        elif model_name == "gpt-3.5-turbo-16k":
            return completion_tokens / 1000 * 0.004 + prompt_tokens / 1000 * 0.003
        else:
            return 0.0
