# Together API Integration for LanguageAgentTreeSearch

This document explains how to use the Together API instead of OpenAI API in the LanguageAgentTreeSearch (LATS) implementation.

## Overview

The codebase has been adapted to support Together API, which provides access to various open-source language models at lower costs than OpenAI. The integration maintains compatibility with the existing codebase while offering more model options and cost savings.

## Setup

### 1. Get Together API Key

1. Sign up at [Together AI](https://together.ai/)
2. Get your API key from the dashboard
3. Set the environment variable:

```bash
export TOGETHER_API_KEY="your_together_api_key_here"
```

### 2. Fallback to OpenAI (Optional)

If you don't have a Together API key, the system will fall back to OpenAI:

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

## Model Mapping

The system automatically maps OpenAI model names to Together API models:

| OpenAI Model | Together Model | Use Case |
|-------------|----------------|----------|
| `gpt-3.5-turbo` | `meta-llama/Llama-2-7b-chat-hf` | General reasoning |
| `gpt-4` | `meta-llama/Llama-2-70b-chat-hf` | Complex reasoning |
| `gpt-3.5-turbo-16k` | `meta-llama/Llama-2-7b-chat-hf` | Long context |

### Available Together Models

You can also use these models directly:

- **Llama 2**: `llama-2-7b`, `llama-2-13b`, `llama-2-70b`
- **CodeLlama**: `codellama-7b`, `codellama-13b`, `codellama-34b`
- **Mistral**: `mistral-7b`, `mixtral-8x7b`
- **Qwen**: `qwen-7b`, `qwen-14b`, `qwen-32b`

## Usage

### Running LATS with Together API

The integration is automatic. Just run the existing scripts:

```bash
# HotPotQA domain
cd hotpot
python run.py --algorithm lats --backend gpt-3.5-turbo

# Programming domain  
cd programming
python main.py --strategy mcts --model gpt-3.5-turbo

# WebShop domain
cd webshop
python run.py --algorithm lats --backend gpt-3.5-turbo
```

### Using Specific Together Models

To use specific Together models, modify the model parameter:

```bash
# Use CodeLlama for programming tasks
python main.py --strategy mcts --model codellama-7b

# Use Mistral for reasoning tasks
python run.py --algorithm lats --backend mistral-7b
```

## Cost Comparison

Together API offers significant cost savings:

| Model | OpenAI Cost (per 1M tokens) | Together Cost (per 1M tokens) | Savings |
|-------|----------------------------|-------------------------------|---------|
| GPT-3.5-turbo | $2.00 | $0.20 | 90% |
| GPT-4 | $60.00 | $0.70 | 98.8% |
| GPT-3.5-turbo-16k | $4.00 | $0.20 | 95% |

## Testing

Run the test script to verify the integration:

```bash
python test_together_api.py
```

This will:
1. Test API connectivity
2. Verify model mapping
3. Calculate costs
4. Show usage statistics

## Configuration

### Environment Variables

- `TOGETHER_API_KEY`: Your Together API key
- `OPENAI_API_KEY`: Fallback OpenAI API key
- `OPENAI_API_BASE`: Custom OpenAI endpoint (optional)

### Model Configuration

Edit `together_config.py` to:
- Add new model mappings
- Update pricing information
- Configure custom models

## Troubleshooting

### Common Issues

1. **API Key Not Set**
   ```
   Warning: TOGETHER_API_KEY is not set
   ```
   Solution: Set your Together API key

2. **Model Not Found**
   ```
   Error: Model 'custom-model' not found
   ```
   Solution: Add the model to `TOGETHER_MODELS` in `together_config.py`

3. **Rate Limiting**
   ```
   Error: Rate limit exceeded
   ```
   Solution: The system includes automatic retry with exponential backoff

### Debug Mode

Enable debug logging to see API calls:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Considerations

### Model Selection

- **Llama-2-7b**: Fast, good for simple tasks
- **Llama-2-70b**: Slower, better for complex reasoning
- **CodeLlama**: Optimized for programming tasks
- **Mistral**: Good balance of speed and quality

### Batch Processing

The system automatically batches requests to optimize API usage and reduce costs.

## Migration from OpenAI

The integration is designed to be drop-in compatible. No code changes are required in your existing LATS scripts.

### Before (OpenAI)
```bash
export OPENAI_API_KEY="sk-..."
python run.py --backend gpt-4
```

### After (Together)
```bash
export TOGETHER_API_KEY="your_key"
python run.py --backend gpt-4  # Automatically uses Llama-2-70b
```

## Support

For issues with Together API integration:

1. Check the test script output
2. Verify API key and permissions
3. Review Together AI documentation
4. Check model availability and pricing

## License

This integration maintains the same license as the original LanguageAgentTreeSearch project.
