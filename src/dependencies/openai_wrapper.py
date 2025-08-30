class OpenAiWrapper:

    def __init__(self, client: OpenAI, model: str, num_tries: str):
        self.client = client
        self.model = model
        self.num_tries = num_tries

    def vibe_eval_statement(self, statement: str, fallback: bool = False) -> bool:
        for i in range(0, self.num_tries):
            response = self.client.responses.create(
                model=self.model,
                instructions="Evaluate the statement below and respond with either 'true' or 'false'.",
                input=statement,
            )

            # TODO: check if response.output_text is true or false and convert to boolean to return
            # if not, keep trying again until num tries is exceeded

        return fallback
