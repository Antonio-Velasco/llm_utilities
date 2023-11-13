from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain  # noqa E501
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import create_qa_with_sources_chain, RetrievalQA

import itertools as it
from langchain.callbacks import get_openai_callback


from llama_index import VectorStoreIndex, ServiceContext
from llama_index.node_parser import SimpleNodeParser
from llama_index import Document
from llama_index import LLMPredictor
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

import pandas as pd
import json
import re


######################################
#### Unstructured Data Extraction ####
######################################


# def extract_unstructured(pages, system, json_template):
#     # create the open-source embedding function
#     embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#     # load it into Chroma
#     db = Chroma.from_documents(pages, embedding_function)

#     llm_src = ChatOpenAI(temperature=0, model="gpt-4")

#     qa_chain = create_qa_with_sources_chain(llm_src)

#     doc_prompt = PromptTemplate(
#         template="Content: {page_content}\n source: {source}", # look at the prompt does have page#
#         input_variables=["page_content", "source"],
#     )

#     final_qa_chain = StuffDocumentsChain(
#         llm_chain=qa_chain, 
#         document_variable_name='context',
#         document_prompt=doc_prompt,
#     )
#     retrieval_qa = RetrievalQA(
#         retriever=db.as_retriever(),
#         combine_documents_chain=final_qa_chain
#     )
#     response = retrieval_qa.run(system)

#     return response

def extract_unstructured(extracted_text, system, json_template):

    documents = [Document(text=extracted_text["content"])]

    node_parser = SimpleNodeParser.from_defaults(chunk_size=4096,
                                                 chunk_overlap=200)
                                                 
    llm = ChatOpenAI(temperature=0, max_tokens=512)
    llm_predictor = LLMPredictor(llm=llm)
    service_context = ServiceContext.from_defaults(node_parser=node_parser,
                                                   llm_predictor=llm_predictor)

    index = VectorStoreIndex.from_documents(documents,
                                            service_context=service_context)
    query_engine = index.as_query_engine()
    response = query_engine.query(system)

    return response.response


######################################
######## Document Summarizer #########
######################################


map_prompt = """
Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:
"""
map_prompt_template = PromptTemplate(template=map_prompt,
                                     input_variables=["text"])


reduce_prompt = """
Write a concise summary of the following text delimited by triple backquotes.
Return your response in bullet points which covers the key points of the text.
```{text}```
BULLET POINT SUMMARY:
"""
reduce_prompt_template = PromptTemplate(template=reduce_prompt,
                                        input_variables=["text"])


def summarization_chain(verbose=False):
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


def split_text(text,
               separators=["\n\n", "\n", " "],
               chunk_size=3000,
               chunk_overlap=500):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=separators + [""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)
    docs = text_splitter.create_documents([text])
    return docs


def split_in_chuncks(docs, number_of_chunks):

    chunk_size = (len(docs) // number_of_chunks)

    groups = [
        [group for _, group in enumerated_group]
        for _, enumerated_group in it.groupby(
                enumerate(docs),
                key=lambda e: e[0] // chunk_size
                )
            ]
    return groups


def summarize_text(text, number_of_chunks, include_costs=False):
    with get_openai_callback() as cb:
        chain = summarization_chain()
        docs = split_text(text)
        groups = split_in_chuncks(docs, number_of_chunks)
        summaries = [chain.run(group) for group in groups]
        joined_summary = "\n\n".join(summaries)

        if include_costs:
            return joined_summary, cb
        return joined_summary
