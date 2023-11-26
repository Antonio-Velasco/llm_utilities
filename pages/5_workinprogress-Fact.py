from pathlib import Path
from urllib.parse import unquote, urlparse
import requests
import streamlit as st
from st_audiorec import st_audiorec

import openai
import base64

from modules.state import read_url_param_values

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


configuration()
json_template = {}


"""
### Please provide either a file or a url
"""

wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    st.audio(wav_audio_data, format='audio/wav')


if st.button("Process", type="primary"):
    with st.spinner("Extracting fields ..."):
        st.write("hello world")