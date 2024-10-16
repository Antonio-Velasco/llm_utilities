from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain  # noqa E501
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import create_qa_with_sources_chain, RetrievalQA
from langchain_core.documents.base import Document
import itertools as it
from langchain_community.callbacks import get_openai_callback
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

import openai

from azure.core.credentials import AzureKeyCredential  # noqa E402
from azure.ai.formrecognizer import DocumentAnalysisClient  # noqa E402
from modules.state import read_url_param_values

import os
import io


def configuration():
    # Config
    config = read_url_param_values()
    api_key = config["openai_api_key"]
    openai.api_key = api_key
    os.environ["OPENAI_API_KEY"] = api_key


configuration()


#######################
# Document Summarizer #
#######################


map_prompt = """
Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:
"""
map_prompt_template = PromptTemplate(template=map_prompt,
                                     input_variables=["text"])


reduce_prompt = """
Write a concise summary of the following
text delimited by triple backquotes.
Return your response in 3 bullet points
which covers the key ideas of the text.
```{text}```
BULLET POINT SUMMARY:
"""
reduce_prompt_template = PromptTemplate(template=reduce_prompt,
                                        input_variables=["text"])


def summarization_chain(
        verbose: bool = False
        ) -> MapReduceDocumentsChain:

    llm = OpenAI(temperature=0, max_tokens=512)

    map_chain = LLMChain(llm=llm, prompt=map_prompt_template, verbose=verbose)
    reduce_chain = LLMChain(llm=llm,
                            prompt=reduce_prompt_template,
                            verbose=verbose)

    combine_document_chain = StuffDocumentsChain(
        llm_chain=reduce_chain,
        document_variable_name="text",
        verbose=verbose,
    )

    mapreduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        combine_document_chain=combine_document_chain,
        document_variable_name=combine_document_chain.document_variable_name,
        verbose=verbose
    )

    return mapreduce_chain


def split_text(
        text: str,
        separators: list[str] = ["\n\n", "\n", " "],
        chunk_size: int = 3000,
        chunk_overlap: int = 500
        ) -> list[Document]:

    text_splitter = RecursiveCharacterTextSplitter(
        separators=separators + [""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)
    docs = text_splitter.create_documents([text])
    return docs


def split_in_chuncks(
        docs: list[Document],
        number_of_chunks: int
        ) -> list[list[Document]]:

    chunk_size = (len(docs) // number_of_chunks)

    groups = [
        [group for _, group in enumerated_group]
        for _, enumerated_group in it.groupby(
                enumerate(docs),
                key=lambda e: e[0] // chunk_size
                )
            ]
    return groups


def summarize_text(
        text: str,
        number_of_chunks: int,
        include_costs: bool = False
        ) -> str:

    with get_openai_callback() as cb:
        chain = summarization_chain()
        docs = split_text(text)
        groups = split_in_chuncks(docs, number_of_chunks)
        summaries = [chain.run(group) for group in groups]
        joined_summary = "\n\n ---- \n".join(summaries)

        if include_costs:
            return joined_summary, cb
        return joined_summary


##################
# Document Query #
##################


def document_queries(pages, query):

    system = f"""
    You are a helpfull assitant tasked with answering user queries
    about a provided document.
    Try to be concise and provide all the information available.
    When adding the source page, answer with just the number.
    User query:
    ###
    {query}
    ###
    """

    # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2")
    # load it into Chroma
    db = Chroma.from_documents(pages, embedding_function)

    llm_src = ChatOpenAI(temperature=0, model="gpt-4")

    qa_chain = create_qa_with_sources_chain(llm_src)

    doc_prompt = PromptTemplate(
        template="Content: {page_content}\n Source page: {page}",
        input_variables=["page_content", "page"],
    )

    final_qa_chain = StuffDocumentsChain(
        llm_chain=qa_chain,
        document_variable_name='context',
        document_prompt=doc_prompt,
    )
    retrieval_qa = RetrievalQA(
        retriever=db.as_retriever(),
        combine_documents_chain=final_qa_chain
    )
    response = retrieval_qa.run(system)

    return response


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


def extract_text(file: io.BytesIO) -> list[Document]:
    result = base_form_recogniser(file)
    pages = []
    for page in result.pages:
        doc_with_metadata = Document(
            page_content=("\n".join([line.content for line in page.lines])),
            metadata={"page": f"{page.page_number}"})
        pages.append(doc_with_metadata)

    return pages


def is_pdf(file: io.BytesIO) -> bool:
    return file is not None and file.type == "application/pdf"
