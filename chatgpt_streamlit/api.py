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
    # load tacotron2 model from torch hub
    print("Loading Tacotron2 model from torch hub...")
    tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')
    tacotron2 = tacotron2.to('cuda')
    tacotron2.eval()

    # load waveglow model from torch hub
    print("Loading Waveglow model from torch hub...")
    waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow = waveglow.to('cuda')

    # load preprocessing utils
    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')

    # define forward function
    def forward(text: str) -> bytes:
        # preprocess text
        sequences, lengths = utils.prepare_input_sequence([text])

        with torch.inference_mode():
            # text to mel
            mel, _, _ = tacotron2.infer(sequences, lengths)
            audio = waveglow.infer(mel)

        # mel to audio
        audio_numpy = audio[0].data.cpu().numpy()
        audio_bytes = audio_numpy.tobytes()

        # make a buffer
        buf = BytesIO(audio_bytes)
        
        # make a binary data of wav file from numpy array
        sf.write(buf, audio_numpy, 22050, format="WAV", subtype="PCM_16")

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
