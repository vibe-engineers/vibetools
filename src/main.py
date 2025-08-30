class Vibechecks:

    def __init__(self, client: OpenAI | genai.Client, model: str, num_tries: int):
        self.__load_llm(client, model)

    def __load_llm(self, client: OpenAI | genai.Client, model: str, num_tries: int):
        if isinstance(client, OpenAI):
            self.llm = OpenAiWrapper(client, model, num_tries)
        elif isinstance(client, genai.Client):
            self.llm = GeminiWrapper(client, model, num_tries)
        else: 
            raise InvalidClientException # TODO: implement custom exceptions

    def __call__(self, statement):
        # TODO: use self.llm to determine if statement is true 
        # use the gemini wrapper and openai wrapper here
        self.llm.vibe_eval_statement(statement)

