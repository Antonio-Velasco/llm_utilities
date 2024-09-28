# Overview

**LLM utility tools** is a set of simple techniques and applications
powered by a LLM (Large Language Model). They provide quality of life
and/or improve efficiency in common day to day office tasks.

I started developing some of these as either learning exercises, 
for my own personal use or as demo examples of what can be done for the industry.

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

![](/data/media/chat_agent.gif)

#### 📃 Document Summarizer

The Document Summarizer accepts a drag-and-drop PDF or a URL. It extracts
the text from the source and then sends it to a LLM chain that uses Map-Reduce
to synthetise the content down to 3 bullet points per number of paragraphs 
instructed.  

![](/data/media/doc_summarizer.gif)

#### Future updates
* *Document Query* - (To be documented)
* *Unstructured Data Extraction* - (To be updated & documented)
* *Table Agent* - (To be updated & documented)
* *Text Composer with memory* - (WIP)


