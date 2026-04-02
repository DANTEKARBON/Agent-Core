from abc import ABC, abstractmethod

class LLMAdapter(ABC):
    @abstractmethod
    def generate(self, prompt, **kwargs):
        pass
