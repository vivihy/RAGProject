import os
from abc import ABC, abstractmethod
from zhipuai import ZhipuAI

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        pass

class ZhipuLLM(BaseLLM):
    def __init__(self, api_key: str = None, model: str = "glm-4-flash"):
        api_key = api_key or os.environ.get("ZHIPU_API_KEY")
        if not api_key:
            raise ValueError("ZHIPU_API_KEY not set. Please set environment variable.")
        self.client = ZhipuAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content

class OpenAIClient(BaseLLM):
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        from openai import OpenAI
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content

def get_llm(provider: str, model: str = None, api_key: str = None) -> BaseLLM:
    if provider == "zhipu-glm4-flash":
        return ZhipuLLM(api_key=api_key, model="glm-4-flash")
    elif provider == "gpt-class":
        return OpenAIClient(api_key=api_key, model=model or "gpt-3.5-turbo")
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")