import streamlit as st
from audio_recorder_streamlit import audio_recorder
from chatgpt_streamlit.api import call_chatgpt_api, call_whisper_api, text_to_speech
from typing import Union


def main() -> None:
    # title
    st.title("ChatGPT Streamlit")    

    # record audio
    audio_bytes = audio_recorder()

    if audio_bytes:
        # add separator
        st.markdown("---")

        # display audio
        st.text("Recorded audio")
        st.audio(audio_bytes, format="audio/wav")

        # add separator
        st.markdown("---")

        # call APIs
        if st.button("Call API - without TTS"):
            setup_calling_api_seciton(audio_bytes)
            st.balloons()

        if st.button("Call API - with TTS"):
            answer = setup_calling_api_seciton(audio_bytes)
            st.balloons()
            spoken_answer = text_to_speech()(answer)
            st.audio(spoken_answer, format="audio/wav")


def setup_calling_api_seciton(audio_bytes: bytes) -> Union[str, None]:
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
                return answer
            
        st.error("Failed to call API.")
    return ''


if __name__ == "__main__":
    main()
