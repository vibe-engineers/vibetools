from openai import OpenAI
from dependencies.llm_wrapper import LlmWrapper
from exceptions import VibeResponseException

class OpenAiWrapper(LlmWrapper):

    def __init__(self, client: OpenAI, model: str, num_tries: str):
        self.client = client
        self.model = model
        self.num_tries = num_tries

    def vibe_eval_statement(self, statement: str) -> bool:
        for i in range(0, self.num_tries):
            response = self.client.responses.create(
                model=self.model,
                instructions=self._system_instruction,
                input=statement,
            )

            output_text = response.output_text.lower().strip()

            if "true" in output_text:
                return True
            elif "false" in output_text:
                return False

        raise VibeResponseException
