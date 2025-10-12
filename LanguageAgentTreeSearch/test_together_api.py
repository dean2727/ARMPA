#!/usr/bin/env python3
"""
Test script for Together API integration with LanguageAgentTreeSearch

This script tests the Together API integration by making a simple
API call and verifying the response.
"""

import os
import sys
import openai
from together_config import get_api_config, get_model_name, calculate_cost

def test_together_api():
    """Test the Together API integration."""
    print("Testing Together API integration...")
    
    try:
        # Get API configuration
        config = get_api_config()
        print(f"Using {config['provider']} API")
        
        # Set up OpenAI client
        openai.api_key = config["api_key"]
        openai.api_base = config["api_base"]
        
        # Test with a simple prompt
        test_prompt = "Hello! Please respond with a short greeting."
        model_name = "gpt-3.5-turbo"  # This will be mapped to Llama-2-7b
        together_model = get_model_name(model_name)
        
        print(f"Testing with model: {model_name} -> {together_model}")
        
        # Make API call
        response = openai.ChatCompletion.create(
            model=together_model,
            messages=[{"role": "user", "content": test_prompt}],
            max_tokens=50,
            temperature=0.7
        )
        
        # Extract response
        content = response.choices[0].message.content
        usage = response.usage
        
        print(f"Response: {content}")
        print(f"Usage: {usage.completion_tokens} completion tokens, {usage.prompt_tokens} prompt tokens")
        
        # Calculate cost
        cost = calculate_cost(model_name, usage.completion_tokens, usage.prompt_tokens)
        print(f"Estimated cost: ${cost:.6f}")
        
        print("✅ Together API integration test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Together API integration test failed: {e}")
        return False

def test_model_mapping():
    """Test the model mapping functionality."""
    print("\nTesting model mapping...")
    
    test_cases = [
        ("gpt-3.5-turbo", "meta-llama/Llama-2-7b-chat-hf"),
        ("gpt-4", "meta-llama/Llama-2-70b-chat-hf"),
        ("llama-2-7b", "meta-llama/Llama-2-7b-chat-hf"),
        ("codellama-7b", "codellama/CodeLlama-7b-Instruct-hf")
    ]
    
    for openai_model, expected_together in test_cases:
        actual_together = get_model_name(openai_model)
        if actual_together == expected_together:
            print(f"✅ {openai_model} -> {actual_together}")
        else:
            print(f"❌ {openai_model} -> {actual_together} (expected {expected_together})")
            return False
    
    return True

def main():
    """Main test function."""
    print("LanguageAgentTreeSearch - Together API Integration Test")
    print("=" * 60)
    
    # Check environment variables
    together_key = os.getenv("TOGETHER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not together_key and not openai_key:
        print("❌ No API keys found. Please set TOGETHER_API_KEY or OPENAI_API_KEY")
        return False
    
    if together_key:
        print("✅ TOGETHER_API_KEY found")
    if openai_key:
        print("✅ OPENAI_API_KEY found (fallback)")
    
    # Test model mapping
    if not test_model_mapping():
        return False
    
    # Test API call
    if not test_together_api():
        return False
    
    print("\n🎉 All tests passed! Together API integration is working.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
