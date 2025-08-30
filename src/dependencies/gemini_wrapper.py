from google import genai
from dependencies.llm_wrapper import LlmWrapper
from exceptions import VibeResponseException

class GeminiWrapper(LlmWrapper):

    def __init__(self, client: genai.Client, model: str, num_tries: int):
        self.client = client
        self.model = model
        self.num_tries = num_tries

    def vibe_eval_statement(self, statement: str) -> bool:
        for i in range(0, self.num_tries):
            response = self.client.models.generate_content(
                model=self.model, 
                contents=statement,
                config=genai.types.GenerateContentConfig(
                    system_instruction=self._system_instruction
                )
            )

            output_text = response.text.lower().strip()

            if "true" in output_text:
                return True
            elif "false" in output_text:
                return False

        raise VibeResponseException

