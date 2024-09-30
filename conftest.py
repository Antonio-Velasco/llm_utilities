# -*- coding: utf-8 -*-

'''
unit test fixtures
'''

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
