from pathlib import Path
from urllib.parse import unquote, urlparse
import requests
from bs4 import BeautifulSoup
import streamlit as st

import pdfplumber
from PyPDF4 import PdfFileWriter, PdfFileReader

import openai
import base64

from azure.core.credentials import AzureKeyCredential  # noqa E402
from azure.ai.formrecognizer import DocumentAnalysisClient  # noqa E402

from modules.llm import extract_unstructured
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

def display_pdf(file):
    # Encode the data into base64
    base64_pdf = base64.b64encode(file.getvalue()).decode("utf-8")

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
    data = result.to_dict()
    return data


def extract_text_ocr(file):
    inputpdf = PdfFileReader(file)
    output = PdfFileWriter()

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            # if page.search("FIT") and page.search("sg") and page.search("EMW"):
            #     output.addPage(inputpdf.pages[page.page_number-1])
            output.addPage(inputpdf.pages[page.page_number-1])

    output_bytesio = io.BytesIO()
    output.write(output_bytesio)

    extracted_text = base_form_recogniser(output_bytesio)

    return extracted_text, output_bytesio


def is_pdf(file):
    return file is not None and file.type == "application/pdf"


@st.cache_data()
def process_pdf(uploaded_file):
    return extract_text_ocr(uploaded_file)


@st.cache_data()
def process_url(url):
    headers = {'User-Agent':
               'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}  # noqa: E501
    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()
    return text


@st.cache_data()
def extract_unstructured_from_document(text, system, json_template):
    return extract_unstructured(text, system, json_template)


def get_text_and_file(uploaded_file, url):
    if url:
        file = unquote(Path(urlparse(url).path).name)
        text = process_url(url)
        pdf_pages = []
    elif is_pdf(uploaded_file):
        file = uploaded_file.name
        text, pdf_pages = process_pdf(uploaded_file)
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


    cols = st.columns((0.7,1,2))
    with cols[0]: 
        number_inputs = st.number_input("Fields Number:", step=1, min_value=1)
    st.write("Number of fields", number_inputs)

    field_names = []
    field_descriptions = []
    for i in range (number_inputs):
        with cols[1]: field_names.append(st.text_input(f'Target field {i+1}', "", key=f"field_input_{i}"))
        with cols[2]: field_descriptions.append("<" + st.text_input(f'Field Description {i+1} (Optional)', "", key=f"desc_input_{i}") + ">")


    json_template = json.dumps({"Results": [dict(zip(field_names, field_descriptions))]})


system = f"""
You are an assistant that given a text extracted using OCR from a document will extract user provided data fields.
Fields can have multiple formats.
Write your output as a JSON with an entry with the format {json_template} per each test you find.
If there is a field that you can not find, set it a null.
If there is any additional information of feedback from the infromation extraction, add a {{"notes": "<additional-information>"}}
"""   # noqa E501

import pandas as pd

if st.button("Process", type="primary"):
    with st.spinner("Extracting fields ..."):

        summary_file, text, pdf_pages = get_text_and_file(uploaded_file,
                                                               url
                                                               )


        """
        ### Extraction result
        """


        response = extract_unstructured_from_document(text, system, json_template)
        
        # Extract the dict from the results key
        results = json.loads(response)["Results"][0]

        # Create a list of tuples from the dict items
        tuples = list(results.items())

        # Create a dataframe from the list of tuples
        df = pd.DataFrame(tuples, columns=["Field", "Content"])

        st.dataframe(df)

        st.json(response)


        """
        ### Relevant pages
        """

        display_pdf(pdf_pages)
