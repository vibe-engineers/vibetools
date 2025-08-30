from typing import Any

from google import genai

from dependencies.llm_wrapper import LlmWrapper
from exceptions import VibeResponseTypeException


class GeminiWrapper(LlmWrapper):

    def __init__(self, client: genai.Client, model: str, num_tries: int):
        self.client = client
        self.model = model
        self.num_tries = num_tries

    def vibe_eval_statement(self, statement: str) -> bool:
        for _ in range(0, self.num_tries):
            response = self.client.models.generate_content(
                model=self.model, 
                contents=statement,
                config=genai.types.GenerateContentConfig(
                    system_instruction=self._eval_statement_instruction
                )
            )

            output_text = response.text.lower().strip()

            if "true" in output_text:
                return True
            elif "false" in output_text:
                return False

        raise VibeResponseTypeException

    def vibe_call_function(self, func_signature: str, docstring: str, *args, **kwargs) -> Any:
        prompt = f"""
        Function Signature: {func_signature}
        Docstring: {docstring}
        Arguments: {args}, {kwargs}
        """

        response = self.client.models.generate_content(
            model=self.model, 
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=self._call_function_instruction
            )
        )

        return response.text.strip()

