class GeminiWrapper:

    def __init__(self, client: genai.client, model: str, num_tries: int):
        self.client = client
        self.model = model
        self.num_tries = num_tries

    def vibe_eval_statement(self, statement: str, fallback: bool = False) -> bool:
        for i in range(0, self.num_tries):
            response = self.client.models.generate_content(
                model=self.model, 
                contents=statement,
                config=types.GenerateContentConfig(
                    # TODO : consider abstracting this into the llmWrapper class to reduce duplication across wrappers
                    system_instruction="Evaluate the statement below and respond with either 'true' or 'false'."
                )
            )

            # print(response.text)

            # TODO: check if response.output_text is true or false and convert to boolean to return
            # if not, keep trying again until num tries is exceeded

        return fallback
