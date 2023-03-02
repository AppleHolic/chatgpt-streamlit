import openai
import json
from io import BytesIO
from typing import Dict, Union, Any, List


CHATGPT_API_NAME: str = "gpt-3.5-turbo"
WHISPER_API_NAME: str = "whisper-1"


def call_chatgpt_api(prompt: str, role: str = "user") -> str:
    result: Dict[str, Union[List[Dict[str, Dict[str, str]]], Any]] = openai.ChatCompletion.create(
        model=CHATGPT_API_NAME,
        messages=[
            {"role": role, "content": prompt},
        ]
    )
    return result["choices"][0]["message"]["content"]


def call_whisper_api(audio_file_binary: bytes) -> str:
    # wrap audio file in BytesIO
    buf = BytesIO(audio_file_binary)
    buf.name = "audio.wav"

    # call Whisper API
    result = openai.Audio.transcribe(WHISPER_API_NAME, buf)

    # parse and return transcription
    return json.loads(str(result))["text"]
