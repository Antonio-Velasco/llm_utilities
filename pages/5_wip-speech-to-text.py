from pathlib import Path
from urllib.parse import unquote, urlparse
import requests
import streamlit as st
from st_audiorec import st_audiorec

import openai
import base64

from modules.state import read_url_param_values
import azure.cognitiveservices.speech as speechsdk

import os
import io
import json
from Home import APP_TITLE, APP_ICON


st.set_page_config(
    page_title=f"{APP_TITLE} - FIT & LOT extraction",
    page_icon=APP_ICON
)


def configuration():
    # Config
    config = read_url_param_values()
    api_key = config["openai_api_key"]
    openai.api_key = api_key
    os.environ["OPENAI_API_KEY"] = api_key
    speech_key = config["speech_key"]
    os.environ["SPEECH_KEY"] = speech_key
    speech_region = config["speech_region"]
    os.environ["SPEECH_REGION"] = speech_region


class BinaryFileReaderCallback(speechsdk.audio.PullAudioInputStreamCallback):
    def __init__(self, file):
        super().__init__()
        self._file_h = file

    def read(self, buffer: memoryview) -> int:
        try:
            size = buffer.nbytes
            frames = self._file_h.read(size)

            buffer[:len(frames)] = frames

            return len(frames)
        except Exception as ex:
            print('Exception in `read`: {}'.format(ex))
            raise

    def close(self) -> None:
        print('closing file')
        try:
            self._file_h.close()
        except Exception as ex:
            print('Exception in `close`: {}'.format(ex))
            raise

configuration()

speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))
speech_config.speech_recognition_language="en-US"


"""
### Please provide either a file or a url
"""

wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    st.audio(wav_audio_data, format='audio/wav')


if st.button("Process", type="primary"):
    with st.spinner("Processing ..."):
        
        st.write("asd")

        from scipy.io.wavfile import write
        import numpy as np
     
        # convert the wav bytes to a numpy array of int16
        wav_array = np.frombuffer(wav_audio_data, dtype=np.int16)

        # create a bytesio object to store the wav data
        wav_bytesio = io.BytesIO()

        # write the wav data to the bytesio object
        write(wav_bytesio, 16000, wav_array)

        # get the bytes from the bytesio object
        wav_bytesio.seek(0) # go back to the beginning of the buffer
        wav_bytes = wav_bytesio.read()


        audio_data = base64.b64decode(wav_audio_data)
        # Create an instance of an audio data stream from the audio data.
        audio_data_format = speechsdk.audio.AudioStreamFormat()
        callback = BinaryFileReaderCallback(wav_bytesio)
        stream = speechsdk.audio.PullAudioInputStream(stream_format=audio_data_format, pull_stream_callback=callback)

        # Create an AudioConfig object with the stream and format parameters
        audio_config = speechsdk.audio.AudioConfig(stream=stream)

        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        result = speech_recognizer.recognize_once()

        st.write(result.text)
        st.write("asddos")


        