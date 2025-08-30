from abc import ABC, abstractmethod

class LlmWrapper(ABC):
    _system_instruction = "Evaluate the statement below and respond with either 'true' or 'false'."

    @abstractmethod
    def vibe_eval_statement(self, statement: str) -> bool:
        pass