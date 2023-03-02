import streamlit as st
from audio_recorder_streamlit import audio_recorder


def main() -> None:
    # get center column
    center = st.columns(3)[1]

    # with center column
    with center:
        audio_bytes = audio_recorder()
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")


if __name__ == "__main__":
    main()
