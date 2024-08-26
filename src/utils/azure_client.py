import os

from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2023-05-15",
    api_key=os.getenv("AZURE_OPENAI_KEY"),
)
