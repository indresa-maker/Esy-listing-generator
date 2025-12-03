import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

async def call_openai(*args, **kwargs):
    return client.chat.completions.create(*args, **kwargs)
