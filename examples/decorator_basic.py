from dotenv import load_dotenv

load_dotenv()

import os

from google import genai

from vibechecks import VibeCheck

# create a google gemini client (an openai client works as well)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# create a vibecheck instance using the above client and specify a model
# model variants for gemini: https://ai.google.dev/gemini-api/docs/models#model-variations
# model variants for openai: https://platform.openai.com/docs/models
vc = VibeCheck(client, model="gemini-2.0-flash-lite")

# the example below vibe evaluates the result of the add_number function
# tip #1: be descriptive in your docstring and declare your parameter/response types
# the llm responds with more relevant information
@vc
def add_number(num1: int, num2: int) -> int:
    """
    Adds two numbers.

    Args:
        num1 (int): The first number.
        num2 (int): The second number.

    Returns:
        int: The sum of the two numbers.

    """
    pass

print(add_number(3, 5))  # should print 8