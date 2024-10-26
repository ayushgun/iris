import base64
import httpx
import json
import os

from abc import ABC, abstractmethod


class InferenceBackend(ABC):
    @abstractmethod
    async def describe_frame(self, frame: bytes, prompt: str) -> str:
        raise NotImplementedError("This is an abstract method that must be implemented")

    @abstractmethod
    async def is_hazard(self, frame: bytes, prompt: str) -> bool:
        raise NotImplementedError("This is an abstract method that must be implemented")


# Implementation of inference using Anthropic Claude as the inference model
class ClaudeBackend(InferenceBackend):
    def __init__(self) -> None:
        self.api_key = os.getenv("ANTHROPIC_KEY")

        if not self.api_key:
            raise ValueError("ANTHROPIC_KEY is missing in the .env file")

        self.model = "claude-3-haiku-20240307"

    async def describe_frame(self, frame: bytes, prompt: str) -> str:
        encoded_frame = base64.b64encode(frame).decode("utf-8")

        message_list = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": encoded_frame,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        data = {"model": self.model, "max_tokens": 50, "messages": message_list}

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages", headers=headers, json=data
            )

        if response.status_code != 200:
            raise ValueError(f"Request failed with status code {response.status_code}")

        response_data = response.json()
        return response_data["content"][0]["text"]

    async def is_hazard(self, frame: bytes, prompt: str) -> bool:
        encoded_frame = base64.b64encode(frame).decode("utf-8")

        message_list = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": encoded_frame,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        data = {"model": self.model, "max_tokens": 1, "messages": message_list}

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages", headers=headers, json=data
            )

        if response.status_code != 200:
            raise ValueError(f"Request failed with status code {response.status_code}")

        response_data = response.json()
        result = response_data["content"][0]["text"].strip().lower()

        if result == "yes":
            return True
        elif result == "no":
            return False
        else:
            raise ValueError("Unexpected response from model: '{result}'")


# Implementation of inference using Deepmind Gemini as the inference model
class GeminiBackend(InferenceBackend):
    def __init__(self) -> None:
        self.api_key = os.getenv("DEEPMIND_KEY")

        if not self.api_key:
            raise ValueError("DEEPMIND_KEY is missing in the .env file")

        self.upload_url = (
            "https://generativelanguage.googleapis.com/upload/v1beta/files"
        )
        self.generate_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

    async def upload_image(self, frame: bytes) -> str:
        mime_type = "image/jpeg"

        headers = {
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(len(frame)),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json",
        }

        payload = json.dumps({"file": {"display_name": "Uploaded Image"}})

        async with httpx.AsyncClient() as client:
            start_response = await client.post(
                f"{self.upload_url}?key={self.api_key}", headers=headers, data=payload
            )

            if start_response.status_code != 200:
                raise ValueError(
                    f"Initial upload request failed with status code {start_response.status_code}"
                )

            upload_url = start_response.headers.get("X-Goog-Upload-URL")

            if not upload_url:
                raise ValueError("No upload URL received in response headers.")

        headers = {
            "Content-Length": str(len(frame)),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        }

        async with httpx.AsyncClient() as client:
            upload_response = await client.post(
                upload_url, headers=headers, content=frame
            )

            if upload_response.status_code != 200:
                raise ValueError(
                    f"Image upload failed with status code {upload_response.status_code}"
                )

            file_info = upload_response.json()
            file_uri = file_info.get("file", {}).get("uri")

            if not file_uri:
                raise ValueError("No file URI returned in upload response.")

        return file_uri

    async def describe_frame(self, frame: bytes, prompt: str) -> str:
        file_uri = await self.upload_image(frame)

        headers = {
            "Content-Type": "application/json",
        }

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "file_data": {
                                "mime_type": "image/jpeg",
                                "file_uri": file_uri,
                            }
                        },
                    ]
                }
            ]
        }

        async with httpx.AsyncClient() as client:
            generate_response = await client.post(
                f"{self.generate_url}?key={self.api_key}", headers=headers, json=payload
            )

            if generate_response.status_code != 200:
                raise ValueError(
                    f"Content generation failed with status code {generate_response.status_code}"
                )

            response_data = generate_response.json()
            text_content = response_data["candidates"][0]["content"]["parts"][0]["text"]

        return text_content

    async def is_hazard(self, frame: bytes, prompt: str) -> bool:
        file_uri = await self.upload_image(frame)

        headers = {
            "Content-Type": "application/json",
        }

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "file_data": {
                                "mime_type": "image/jpeg",
                                "file_uri": file_uri,
                            }
                        },
                    ]
                }
            ],
            "max_tokens": 1,
        }

        async with httpx.AsyncClient() as client:
            generate_response = await client.post(
                f"{self.generate_url}?key={self.api_key}", headers=headers, json=payload
            )

            if generate_response.status_code != 200:
                raise ValueError(
                    f"Content generation failed with status code {generate_response.status_code}"
                )

            response_data = generate_response.json()
            result = (
                response_data["candidates"][0]["content"]["parts"][0]["text"]
                .strip()
                .lower()
            )

        if result == "yes":
            return True
        elif result == "no":
            return False
        else:
            raise ValueError("Unexpected response from model: '{result}'")
