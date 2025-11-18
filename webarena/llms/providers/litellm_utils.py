import os
from typing import List, Dict, Any
import json
from litellm import completion
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

import numpy as np

# import litellm
# litellm._turn_on_debug()

def entropy_from_top_logprobs(top_logprobs):
    """
    top_logprobs: list[TopLogprob] for a single generated token.
    """
    logps = np.array([t.logprob for t in top_logprobs])
    # Convert logprobs → probs safely
    probs = np.exp(logps - np.max(logps))  # numerical stability
    probs /= probs.sum()
    # Compute entropy (nats)
    return -np.sum(probs * np.log(probs + 1e-9))


def get_mean_and_action_entropies(logprobs_data):
    action_decision_entropy = None
    entropies = []

    # TODO: Confirm that tokens indeed look like the following actions when we're on the action generation (e.g. go_back is not 2 tokens)
    actions = [
        'click',
        'type',
        'hover',
        'press',
        'scroll',
        'new_tab',
        'tab_focus',
        'close_tab',
        'goto',
        'go_back',
        'go_forward',
        'stop'
    ]

    for token_info in logprobs_data:
        h = entropy_from_top_logprobs(token_info.top_logprobs)
        entropies.append(h)

        if token_info.token in actions:
            action_decision_entropy = h

    mean_entropy = np.mean(entropies)

    return mean_entropy, action_decision_entropy

def generate_from_litellm_completion(
    prompt: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    system_prompt: str = None,
    stop_sequences: List[str] | None = None,
) -> Dict[str, Any]:
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
            # logprobs=True,
			# top_logprobs=5
        )
    except Exception as e:
        print(f"Error: {e}")
        return None

    answer = response["choices"][0]["message"]["content"]
    reasoning = response["choices"][0]["message"].get("reasoning_content")

    return {
        "answer": answer,
        "reasoning": reasoning
    }



