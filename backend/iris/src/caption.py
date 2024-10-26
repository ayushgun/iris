from typing import Type
from dotenv import load_dotenv
from inference import ClaudeBackend, InferenceBackend

# Load environment variables from the .env file
load_dotenv()

# Retrieve the prompt from the current directory
with open("prompt.txt", "r") as file:
    prompt = file.read()

if not prompt.strip():
    raise ValueError("Unable to locate prompt in prompt.txt")


async def describe_frame(
    frame: bytes, backend: Type[InferenceBackend] = ClaudeBackend
) -> str:
    return await backend().describe_frame(frame, prompt)
