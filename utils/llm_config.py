import os
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic

def setup_llm(provider, api_key, model_name, temperature=0.1, max_tokens=4000, api_base=None, custom_model_name=None):
    """
    Setup and return the appropriate LLM based on provider and model selection.
    
    Args:
        provider (str): "OpenAI", "Anthropic", or "Custom AI Vendor"
        api_key (str): API key for the selected provider
        model_name (str): Model name to use
        temperature (float): Temperature for response generation
        max_tokens (int): Maximum tokens for response
        api_base (str): Custom API base URL for custom providers
        custom_model_name (str): Custom model name for custom providers
    
    Returns:
        LLM instance configured for the selected provider
    """
    if provider == "OpenAI":
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        return OpenAI(
            api_key=api_key,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    elif provider == "Anthropic":
        # The newest Anthropic model is "claude-sonnet-4-20250514", not "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022" nor "claude-3-sonnet-20240229". 
        # If the user doesn't specify a model, always prefer using "claude-sonnet-4-20250514" as it is the latest model. However, if the user has already selected "claude-3-7-sonnet-20250219", keep that selection unless they explicitly request a change.
        return Anthropic(
            api_key=api_key,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    elif provider == "Custom AI Vendor":
        # For custom providers, set environment variable to disable validation
        actual_model = custom_model_name or model_name
        
        # Temporarily disable OpenAI model validation
        import os
        original_env = os.environ.get('OPENAI_API_BASE')
        os.environ['OPENAI_API_BASE'] = api_base or 'https://api.openai.com/v1'
        
        try:
            # Create OpenAI instance with custom settings
            llm = OpenAI(
                api_key=api_key,
                model=actual_model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_base=api_base
            )
            return llm
        except Exception as e:
            # If model validation still fails, create a minimal custom LLM wrapper
            from llama_index.core.llms.llm import LLM
            from llama_index.core.llms.types import CompletionResponse, ChatMessage, MessageRole
            import requests
            
            class CustomLLM(LLM):
                def __init__(self, api_key, model, api_base, temperature=0.1, max_tokens=4000):
                    super().__init__()
                    self.api_key = api_key
                    self.model = model
                    self.api_base = api_base.rstrip('/')
                    self.temperature = temperature
                    self.max_tokens = max_tokens
                
                @property
                def metadata(self):
                    return {"model_name": self.model}
                
                def complete(self, prompt: str, **kwargs) -> CompletionResponse:
                    # Convert to chat format for better compatibility
                    messages = [{"role": "user", "content": prompt}]
                    return self._call_api(messages, **kwargs)
                
                def chat(self, messages, **kwargs) -> CompletionResponse:
                    chat_messages = []
                    for msg in messages:
                        if hasattr(msg, 'role') and hasattr(msg, 'content'):
                            chat_messages.append({"role": msg.role.value, "content": msg.content})
                        elif isinstance(msg, dict):
                            chat_messages.append(msg)
                    return self._call_api(chat_messages, **kwargs)
                
                def _call_api(self, messages, **kwargs):
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    data = {
                        "model": self.model,
                        "messages": messages,
                        "temperature": kwargs.get('temperature', self.temperature),
                        "max_tokens": kwargs.get('max_tokens', self.max_tokens)
                    }
                    
                    response = requests.post(
                        f"{self.api_base}/chat/completions",
                        headers=headers,
                        json=data
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        content = result['choices'][0]['message']['content']
                        return CompletionResponse(text=content)
                    else:
                        raise Exception(f"API call failed: {response.status_code} - {response.text}")
            
            return CustomLLM(api_key, actual_model, api_base, temperature, max_tokens)
        
        finally:
            # Restore original environment
            if original_env:
                os.environ['OPENAI_API_BASE'] = original_env
            elif 'OPENAI_API_BASE' in os.environ:
                del os.environ['OPENAI_API_BASE']
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def setup_local_embedding():
    """
    Setup a local embedding model to avoid OpenAI API dependency.
    
    Returns:
        str: Embedding model setting for local use
    """
    return "local"

def validate_api_key(provider, api_key):
    """
    Validate if the provided API key is properly formatted.
    
    Args:
        provider (str): "OpenAI", "Anthropic", or "Custom AI Vendor"
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
    elif provider == "Custom AI Vendor":
        # For custom providers, just check that it's not empty
        return len(api_key.strip()) > 5
    
    return False

def get_default_models(provider):
    """
    Get default model options for each provider.
    
    Args:
        provider (str): Provider name
    
    Returns:
        list: List of default model names
    """
    if provider == "OpenAI":
        return ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]
    elif provider == "Anthropic":
        return ["claude-sonnet-4-20250514", "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022"]
    elif provider == "Custom AI Vendor":
        return ["custom-model", "deepseek/deepseek-r1-0528:free", "gpt-4o-mini", "claude-3-haiku"]
    else:
        return []
