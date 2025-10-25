import os
from typing import List, Dict, Union
from litellm import completion
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

def generate_from_litellm_completion(
    prompt: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    system_prompt: str = None,
    stop_sequences: List[str] | None = None,
) -> str:
    """
    Generate text completion using LiteLLM with retry logic.
    
    Args:
        prompt (str): The input prompt to send to the model.
        model (str): The model name to use for completion.
        temperature (float): The temperature for text generation.
        max_tokens (int): Maximum number of tokens to generate.
        system_prompt (str): Optional system prompt to prepend.
        stop_sequences (List[str]): Optional list of stop sequences.
        
    Returns:
        str: The generated text response.
        
    Raises:
        ValueError: If TOGETHER_API_KEY environment variable is not set.
    """
    if not os.getenv("TOGETHER_API_KEY"):
        raise ValueError("TOGETHER_API_KEY environment variable must be set.")
    
    messages = [{"content": prompt, "role": "user"}]
    if system_prompt:
        messages.insert(0, {"content": system_prompt, "role": "system"})
    
    # Call completion directly like the working LiteLLMModel does
    try:
        response = completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_sequences,
        )
    except Exception as e:
        print(f"Error: {e}")
        return None


    # Together AI returns text directly in choices[0].text
    return response["choices"][0]["message"]["content"]



