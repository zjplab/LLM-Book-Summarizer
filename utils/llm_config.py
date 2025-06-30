import os
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic

def setup_llm(provider, api_key, model_name):
    """
    Setup and return the appropriate LLM based on provider and model selection.
    
    Args:
        provider (str): Either "OpenAI" or "Anthropic"
        api_key (str): API key for the selected provider
        model_name (str): Model name to use
    
    Returns:
        LLM instance configured for the selected provider
    """
    if provider == "OpenAI":
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        return OpenAI(
            api_key=api_key,
            model=model_name,
            temperature=0.1,  # Lower temperature for more consistent summaries
            max_tokens=4000   # Adequate tokens for detailed summaries
        )
    
    elif provider == "Anthropic":
        # The newest Anthropic model is "claude-sonnet-4-20250514", not "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022" nor "claude-3-sonnet-20240229". 
        # If the user doesn't specify a model, always prefer using "claude-sonnet-4-20250514" as it is the latest model. However, if the user has already selected "claude-3-7-sonnet-20250219", keep that selection unless they explicitly request a change.
        return Anthropic(
            api_key=api_key,
            model=model_name,
            temperature=0.1,  # Lower temperature for more consistent summaries
            max_tokens=4000   # Adequate tokens for detailed summaries
        )
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def validate_api_key(provider, api_key):
    """
    Validate if the provided API key is properly formatted.
    
    Args:
        provider (str): Either "OpenAI" or "Anthropic"
        api_key (str): API key to validate
    
    Returns:
        bool: True if key appears valid, False otherwise
    """
    if not api_key or len(api_key.strip()) == 0:
        return False
    
    if provider == "OpenAI":
        return api_key.startswith("sk-") and len(api_key) > 20
    elif provider == "Anthropic":
        return api_key.startswith("sk-ant-") and len(api_key) > 30
    
    return False
