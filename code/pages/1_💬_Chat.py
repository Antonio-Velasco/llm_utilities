import streamlit as st

from streamlit_chat import message
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI

from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents.openai_assistant import OpenAIAssistantRunnable
from langchain.agents import AgentExecutor

import openai
import os

from modules.state import read_url_param_values
from Home import APP_TITLE, APP_ICON


st.set_page_config(
    page_title=f"{APP_TITLE} - OpenAI Chat",
    page_icon=APP_ICON
)


def configuration():
    # Config
    config = read_url_param_values()
    api_key = config["openai_api_key"]
    openai.api_key = api_key
    os.environ["OPENAI_API_KEY"] = api_key


configuration()


def setup_chat_session():
    # Storing the chat
    if "chat" not in st.session_state:
        st.session_state["chat"] = []


def generate_chat_response(prompt):
    tools = [DuckDuckGoSearchRun()]

    agent = OpenAIAssistantRunnable.create_assistant(
        name="langchain assistant tool",
        instructions="You are a personal assistant. You can search the internet to answer user questions.",
        tools=tools,
        model=config["model"],
        as_agent=True,
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools)
    response = agent_executor.invoke({"content": prompt})
    return response["output"]


def memory_buffer():
    memory = ConversationBufferMemory(return_messages=True)

    for i, msg in list(enumerate(st.session_state.messages)):
        if msg["role"] == "user":
            memory.chat_memory.add_user_message(msg["content"])
        else:
            memory.chat_memory.add_ai_message(msg["content"])
    return memory


config = read_url_param_values()


DEFAULT_CONFIG = {
    "temperature": config["temperature"],
    "max_tokens": config["max_tokens"],
    "top_p": config["top_p"],
}


st.title("Chat")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

import streamlit as st

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = generate_chat_response(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})