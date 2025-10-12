import os
import openai
import backoff 
from transformers import GPT2Tokenizer

completion_tokens = prompt_tokens = 0
MAX_TOKENS = 15000
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')

# Together API configuration
api_key = os.getenv("TOGETHER_API_KEY", "")
if api_key != "":
    openai.api_key = api_key
    openai.api_base = "https://api.together.xyz/v1"
else:
    print("Warning: TOGETHER_API_KEY is not set")
    
# Fallback to OpenAI if Together key not available
if api_key == "":
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key != "":
        openai.api_key = api_key
        print("Using OpenAI API as fallback")
    else:
        print("Warning: Neither TOGETHER_API_KEY nor OPENAI_API_KEY is set")
    
api_base = os.getenv("OPENAI_API_BASE", "")
if api_base != "":
    print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
    openai.api_base = api_base

@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

# Together API model mapping
TOGETHER_MODELS = {
    "gpt-3.5-turbo": "meta-llama/Llama-2-7b-chat-hf",
    "gpt-4": "meta-llama/Llama-2-70b-chat-hf", 
    "gpt-3.5-turbo-16k": "meta-llama/Llama-2-7b-chat-hf",
    "claude-3": "anthropic/claude-3-sonnet",
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
    "qwen-32b": "Qwen/Qwen-32B-Chat"
}

def gpt3(prompt, model="text-davinci-002", temperature=1.0, max_tokens=100, n=1, stop=None) -> list:
    outputs = []
    for _ in range(n):
        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            stop=stop
        )
        outputs.append(response.choices[0].text.strip())
    return outputs

def gpt(prompt, model="gpt-3.5-turbo-16k", temperature=1.0, max_tokens=100, n=1, stop=None) -> list:
    if model == "test-davinci-002":
        return gpt3(prompt, model, temperature, max_tokens, n, stop)
    else:
        # Map OpenAI model names to Together API models
        together_model = TOGETHER_MODELS.get(model, model)
        messages = [{"role": "user", "content": prompt}]
        return chatgpt(messages, model=together_model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

def gpt4(prompt, model="gpt-4", temperature=0.2, max_tokens=100, n=1, stop=None) -> list:
    if model == "test-davinci-002":
        return gpt3(prompt, model, temperature, max_tokens, n, stop)
    else:
        # Map OpenAI model names to Together API models
        together_model = TOGETHER_MODELS.get(model, model)
        messages = [{"role": "user", "content": prompt}]
        return chatgpt(messages, model=together_model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    
def chatgpt(messages, model="gpt-3.5-turbo-16k", temperature=1.0, max_tokens=100, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        outputs.extend([choice["message"]["content"] for choice in res["choices"]])
        # log completion tokens
        completion_tokens += res["usage"]["completion_tokens"]
        prompt_tokens += res["usage"]["prompt_tokens"]
    return outputs
    
def gpt_usage(backend="gpt-4"):
    global completion_tokens, prompt_tokens
    # Together API pricing (per 1M tokens)
    together_pricing = {
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
    
    # Map backend to Together model
    together_model = TOGETHER_MODELS.get(backend, backend)
    
    if together_model in together_pricing:
        pricing = together_pricing[together_model]
        cost = (completion_tokens / 1000000 * pricing["output"] + 
                prompt_tokens / 1000000 * pricing["input"])
    else:
        # Fallback to OpenAI pricing
        if backend == "gpt-4":
            cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
        elif backend == "gpt-3.5-turbo":
            cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
        elif backend == "gpt-3.5-turbo-16k":
            cost = completion_tokens / 1000 * 0.004 + prompt_tokens / 1000 * 0.003
        else:
            cost = 0.0
    
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}
