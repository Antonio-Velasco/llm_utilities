import streamlit as st

APP_TITLE = "FIT & LOT data types extraction"
APP_ICON = "⚙️"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
)

st.write("# Welcome to LLMs utility tools! 👋")

st.sidebar.success("Select a tool above.")

st.markdown(
    """
    **👈 Select a tool from the sidebar to start**

    # Overview

    **LLM utility tools** is a set of simple techniques and applications
    powered by a LLM (Large Language Model). They provide quality of life
    and/or improve efficiency in comon day to day office tasks.

    I started developing some of these as either learning exercises,
    for my own personal use or as demo examples of what can be done
    for the industry.

    The version published here are updated for simplicity and easiness of use.
    Conceived as a showcase and how-to example.

    ## Use Instructions

    ### Installation
    - Git clone in your local machine.
    - Create a '.env' file with: SECRET_OPENAI_API_KEY=[YOUR_KEY_HERE]
        - Alternatively, skip this step and use streamlit's config page. 
    - Create a virtual environment (pipenv files available) and install
    the libraries in 'requirements.txt'
    - In the terminal run the command (either):
        - run streamlit run code/Home.py
        - pipenv run streamlit run code/Home.py

    ## Tools description
    #### 🗨️ Chat agent
    The chat agent is a simple bot interface agent between the LLM and you.
    It has access to a toolset. Currently the toolset includes a web search
    capability powered by **DuckDuckGO**.
    This makes it able to reply with reliable and updated data.

    #### Future updates
    * *Summary Tool* - (To be updated & documented)
    * *Document Query* - (To be documented)
    * *Unstrusctured Data Extraction* - (To be updated & documented)
    * *Table Agent* - (To be updated & documented)
    * *Text Composer with memory* - (WIP)

    ## Powered by Streamlit.
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    ##### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
"""
)
