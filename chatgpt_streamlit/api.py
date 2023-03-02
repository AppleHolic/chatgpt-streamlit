import openai
import json
import streamlit as st
from io import BytesIO
from typing import Dict, Union, Any, List, Callable
try:
    import torch
    import soundfile as sf
except ImportError:
    torch = None


CHATGPT_API_NAME: str = "gpt-3.5-turbo"
WHISPER_API_NAME: str = "whisper-1"


# make a singletone instance for TTS
@st.cache_resource 
def text_to_speech() -> Union[Callable, None]:
    if torch is None:
        return None
    
    model, symbols, sample_rate, _, apply_tts = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_tts',
        language='en',
        speaker='lj_16khz'
    )
    model = model.to('cpu')  # gpu or cpu

    # define forward function
    def forward(text: str) -> bytes:
        with torch.inference_mode():
            audio = apply_tts(
                texts=[sentence + '.' for sentence in text.split('.') if sentence],
                model=model,
                sample_rate=sample_rate,
                symbols=symbols,
                device='cpu'
            )

        # mel to audio
        audio = torch.cat(audio, dim=-1).squeeze()
        audio_numpy = audio.data.cpu().numpy()
        audio_bytes = audio_numpy.tobytes()

        # make a buffer
        buf = BytesIO(audio_bytes)
        
        # make a binary data of wav file from numpy array
        sf.write(buf, audio_numpy, 16000, format="WAV", subtype="PCM_16")

        # return binary of wav file
        return buf.getvalue()

    return forward


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
