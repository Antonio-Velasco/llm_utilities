import openai
import os

from code.modules.llm import summarize_text
from code.modules.state import read_url_param_values


def configuration():
    # Config
    config = read_url_param_values()
    api_key = config["openai_api_key"]
    openai.api_key = api_key
    os.environ["OPENAI_API_KEY"] = api_key


configuration()


def test_summarize_text(glacier_wiki_text):
    t = summarize_text(glacier_wiki_text["text"], 1)
    print(t)
    assert type(t) is str
