from pathlib import Path
from urllib.parse import unquote, urlparse
import requests
from bs4 import BeautifulSoup
import streamlit as st

import pdfplumber

import openai
import base64

from azure.core.credentials import AzureKeyCredential  # noqa E402
from azure.ai.formrecognizer import DocumentAnalysisClient  # noqa E402

from PyPDF4 import PdfFileReader, PdfFileWriter

from modules.llm import document_queries
from modules.state import read_url_param_values
from langchain.schema.document import Document

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

def display_pdf(file, source_pages):

    inputpdf = PdfFileReader(file)
    output = PdfFileWriter()

    with pdfplumber.open(file) as pdf:
        for page in source_pages:
            if page[0] != "null":
                output.addPage(inputpdf.pages[int(*page)-1])

    output_bytesio = io.BytesIO()
    output.write(output_bytesio)

    # Encode the data into base64
    base64_pdf = base64.b64encode(output_bytesio.getvalue()).decode("utf-8")

    # Create the HTML tag
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'  # noqa E501
    # Render the tag in Streamlit
    st.markdown(pdf_display, unsafe_allow_html=True)


def base_form_recogniser(pdf_bytes: io.BytesIO) -> dict:
    # OCR from base form recogniser
    config = read_url_param_values()
    credential = AzureKeyCredential(config["form_key"])
    document_analysis_client = DocumentAnalysisClient(config["form_endpoint"],
                                                      credential)
    document = pdf_bytes.getvalue()

    # Start the document analysis
    poller = document_analysis_client.begin_analyze_document(
        "prebuilt-document", document, polling_interval=5)

    # Get the result
    result = poller.result()
    return result


def extract_text(file):
    result = base_form_recogniser(file)
    
    pages = []
    for page in result.pages:
        doc_with_metadata = Document(page_content=("\n".join([line.content for line in page.lines])),
                                     metadata={"page": f"{page.page_number}"})
        pages.append(doc_with_metadata)

    return pages


def is_pdf(file):
    return file is not None and file.type == "application/pdf"


@st.cache_data()
def process_pdf(uploaded_file):
    return extract_text(uploaded_file)


@st.cache_data()
def process_url(url):
    headers = {'User-Agent':
               'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}  # noqa: E501
    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()
    return text


@st.cache_data()
def query_from_document(_pages, query):
    return document_queries(_pages, query)


def get_text_and_file(uploaded_file, url):
    if url:
        file = unquote(Path(urlparse(url).path).name)
        text = process_url(url)
        pdf_pages = []
    elif is_pdf(uploaded_file):
        file = uploaded_file.name
        text = ""
        pdf_pages = process_pdf(uploaded_file)
    else:
        file = uploaded_file.name
        text = uploaded_file.getvalue().decode()
        pdf_pages = []
    return file, text, pdf_pages


"""
### Please provide either a file or a url
"""


# Pdf input
uploaded_file = st.file_uploader(
    "Upload a document to extract unstructured data!",
    type=['txt', 'pdf']
    )

# Url input
url = st.text_input("Or paste a Url here:")

# Validate Pdf and url input
if uploaded_file and url:
    st.error("Please, provide only file or url")
    st.stop()

# Pdf extra configuration
page_range = None
clean_document = False
if is_pdf(uploaded_file):
    with st.expander("How should this file be processed?"):
        if is_pdf(uploaded_file):
            clean_document = st.checkbox(
                "Try to remove common footers and headers"
                 )


"""
### Query
"""

query = st.text_input("Ask anything about the document", "")

if query:
    if st.button("Ask document", type="primary"):
        with st.spinner("Thinking ..."):

            summary_file, text, pdf_pages = get_text_and_file(uploaded_file,
                                                                url
                                                                )

            response = query_from_document(pdf_pages, query)
                
            """
            ### Answer
            """
            result = json.loads(response)["answer"]            
            st.write(result)


            if json.loads(response)["sources"]:
                """
                ### Source pages
                """
                source_pages = json.loads(response)["sources"][0]
                st.write(f"Answer provided based on information found in pages: {source_pages}")
                display_pdf(uploaded_file, source_pages)
