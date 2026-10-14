"""Tests for coding_eval/llm_coder.py: the frozen prompt and reading Llama's replies. No Ollama needed.
Run all tests with:  pytest"""
from coding_eval.config import CONFIG
from coding_eval.data import llm_path
from coding_eval.llm_coder import answer_schema, load_codebook, parse_reply, prompt_version, system_prompt


def test_frozen_prompt_has_not_changed():
    # D-013: prompt B was frozen before the test run. Changing a codebook or the prompt text breaks this test.
    for scheme, version in CONFIG["llm"]["frozen_versions"].items():
        assert prompt_version(system_prompt(load_codebook(scheme))) == version


def test_prompt_A_versions():
    # The prompt A versions recorded in results/prompt_comparison_dev.json (D-023)
    for scheme, version in {"12": "affd148b", "50": "06a61780"}.items():
        assert prompt_version(system_prompt(load_codebook(scheme), with_examples=False)) == version


def test_prompt_A_is_prompt_B_without_examples():
    codebook = load_codebook("12")
    a, b = system_prompt(codebook, with_examples=False), system_prompt(codebook)
    assert "Examples:" in b and "Examples:" not in a
    assert "- Follow the definitions and examples." in b and "- Follow the definitions." in a
    assert all(c["definition"] in a for c in codebook)        # A keeps every definition


def test_prompt_A_never_overwrites_prompt_B_files():
    assert llm_path("dev", "12").name == "llm_dev_12.csv"
    assert llm_path("dev", "12", prompt="A").name == "llm_dev_12_A.csv"


def test_schema_only_allows_codebook_names():
    codebook = load_codebook("12")
    schema = answer_schema(codebook)
    assert schema["properties"]["category"]["enum"] == [c["name"] for c in codebook]
    assert "Uncoded" in schema["properties"]["category"]["enum"]
    assert schema["required"] == ["category", "unsure"]


def reply(content):
    """What Ollama sends back, with only the part we read."""
    return {"message": {"content": content}}


def test_parse_a_good_reply():
    assert parse_reply(reply('{"category": "Health", "unsure": false}')) == ("Health", False)
    assert parse_reply(reply('{"category": "Uncoded", "unsure": true}')) == ("Uncoded", True)


def test_parse_bad_replies():
    assert parse_reply(reply("Health")) == ("INVALID", "")                 # not JSON
    assert parse_reply(reply('{"category": "Health"')) == ("INVALID", "")  # cut off half way
    assert parse_reply(reply('{"unsure": false}')) == ("INVALID", "")      # no category
    assert parse_reply(reply('["Health"]')) == ("INVALID", "")             # JSON, but a list
    assert parse_reply(reply(None)) == ("INVALID", "")                     # empty message
    assert parse_reply({"error": "model crashed"}) == ("INVALID", "")      # an error instead of a message
