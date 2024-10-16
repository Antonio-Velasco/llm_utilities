from modules.state import (
    read_environ_params,
    read_url_param_values
    )


def test_read_environ_params():
    t = read_environ_params()
    assert "openai_api_key" in t


def test_read_url_param_values():
    t = read_url_param_values()
    assert "openai_api_key" in t
