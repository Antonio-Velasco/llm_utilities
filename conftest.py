# -*- coding: utf-8 -*-

import io
import json
import pytest


@pytest.fixture()
def glacier_wiki_text():
    response = json.load(open("data/fixtures/glacier_wiki_fixture.json", "r"))
    yield response


@pytest.fixture()
def chain_response_text():
    response = json.load(open("data/fixtures/chain_response_text.json", "r"))
    yield response


@pytest.fixture()
def pdf_example():
    with open('data/fixtures/fmars-10-1221701.pdf', 'rb') as file:
        pdf_data = file.read()
    yield io.BytesIO(pdf_data)