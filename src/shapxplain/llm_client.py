"""
LLM Client for interacting with supported Large Language Models.
"""
import openai

def query_llm(prompt, api_key):
    """
    Query an LLM with a given prompt and return the response.
    """
    openai.api_key = api_key
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=200
    )
    return response["choices"][0]["text"].strip()
