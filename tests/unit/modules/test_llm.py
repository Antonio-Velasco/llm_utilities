import pytest
from code.modules.llm import (
    split_text,
    split_in_chuncks,
    summarize_text,
    extract_text,
    is_pdf
    )

from langchain.schema.document import Document


@pytest.fixture()
def mock_mapreduce_chain(mocker, chain_response_text):
    mocker.patch(
        "langchain.chains.combine_documents.map_reduce.MapReduceDocumentsChain.run",  # noqa: E501
        return_value=chain_response_text["text"]
    )


def test_split_text(glacier_wiki_text):
    t = split_text(glacier_wiki_text["text"])
    assert "glacier" in t[0].page_content
    assert len(t[0].page_content) == 2997
    assert len(t[1].page_content) == 869
    assert len(t) == 2


def test_split_in_chunks(glacier_wiki_text):
    d = split_text(glacier_wiki_text["text"])
    t = split_in_chuncks(d, 1)
    u = split_in_chuncks(d, 2)
    assert len(t) == 1
    assert len(u) == 2


@pytest.mark.usefixtures("mock_mapreduce_chain")
def test_summarize_text(glacier_wiki_text):
    t = summarize_text(glacier_wiki_text["text"], 1)
    assert len(t) == 265


def test_extract_text(pdf_example):
    t = extract_text(pdf_example)
    assert len(t) == 14
    assert type(t) is list
    assert type(t[0]) is Document


def test_is_pdf(pdf_example):
    t = is_pdf(pdf_example)
    u = is_pdf("string")
    v = is_pdf()
    assert t
    assert not u
    assert not v
