import json

from typing import Type
from dotenv import load_dotenv
from inference import ClaudeBackend, InferenceBackend

# Load environment variables from the .env file
load_dotenv()

# Retrieve the prompt from the current directory
with open("prompt.json", "r") as file:
    prompts = json.load(file)

# Check if describe_prompt and hazard_prompt are present and not empty
describe_prompt = prompts.get("describe_prompt", "").strip()
hazard_prompt = prompts.get("hazard_prompt", "").strip()

if not describe_prompt:
    raise ValueError("Unable to locate describe_prompt in prompt.json")

if not hazard_prompt:
    raise ValueError("Unable to locate hazard_prompt in prompt.json")


async def describe_frame(
    frame: bytes, backend: Type[InferenceBackend] = ClaudeBackend
) -> str:
    return await backend().describe_frame(frame, describe_prompt)


async def is_hazardous_frame(
    frame: bytes, backend: Type[InferenceBackend] = ClaudeBackend
) -> bool:
    return await backend().is_hazard(frame, hazard_prompt)
