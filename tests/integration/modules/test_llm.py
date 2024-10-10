import openai
import os
from modules.llm import extract_text
from langchain.schema.document import Document
from modules.state import read_url_param_values


def configuration():
    # Config
    config = read_url_param_values()
    api_key = config["openai_api_key"]
    openai.api_key = api_key
    os.environ["OPENAI_API_KEY"] = api_key


configuration()


def test_extract_text(pdf_example):
    t = extract_text(pdf_example)
    assert len(t) == 14
    assert type(t) is list
    assert type(t[0]) is Document
