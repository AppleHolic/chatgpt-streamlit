import streamlit as st
from audio_recorder_streamlit import audio_recorder
from chatgpt_streamlit.api import call_chatgpt_api, call_whisper_api


def main() -> None:
    st.title("ChatGPT Streamlit")

    audio_bytes = audio_recorder()

    prompt = ''

    if audio_bytes:
        st.text("Recorded audio")
        st.audio(audio_bytes, format="audio/wav")

        if st.button("Call API"):
            with st.spinner("Calling Whisper API..."):
                prompt = call_whisper_api(audio_bytes)

                st.text("Whisper API response:")
                st.text(prompt)

                if not prompt:
                    st.error("Please enter a prompt or record audio.")
                else:
                    with st.spinner("Calling ChatGPT API..."):
                        answer = call_chatgpt_api(prompt)
                        print(answer)
                        st.text("ChatGPT API response:")
                        st.text_area('', value=answer)


if __name__ == "__main__":
    main()
