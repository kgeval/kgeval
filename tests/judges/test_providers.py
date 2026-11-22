"""Test creating judges from ``provider:model`` names."""

import pytest

from kgeval.errors import JudgeError
from kgeval.judges.providers import PROVIDERS, create_judge


@pytest.fixture(autouse=True)
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for provider in PROVIDERS.values():
        monkeypatch.setenv(provider.api_key_variable, "test-key")
        monkeypatch.delenv(provider.base_url_variable, raising=False)
    monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "https://kg.openai.azure.com/openai/v1/")


def base_url(model: str, base_url: str | None = None) -> str:
    return str(create_judge(model, base_url=base_url)._client.base_url)


def test_a_model_without_a_provider_goes_to_openai() -> None:
    judge = create_judge("gpt-5-mini")

    assert judge.name == "openai:gpt-5-mini"
    assert base_url("gpt-5-mini") == "https://api.openai.com/v1/"


@pytest.mark.parametrize(
    ("model", "name", "url"),
    [
        (
            "groq:llama-3.3-70b-versatile",
            "groq:llama-3.3-70b-versatile",
            "https://api.groq.com/openai/v1/",
        ),
        (
            "gemini:gemini-2.5-flash",
            "gemini:gemini-2.5-flash",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
        ("ollama:llama3.1:8b", "ollama:llama3.1:8b", "http://localhost:11434/v1/"),
        ("azure:my-deployment", "azure:my-deployment", "https://kg.openai.azure.com/openai/v1/"),
    ],
)
def test_a_provider_prefix_selects_the_provider(model: str, name: str, url: str) -> None:
    assert create_judge(model).name == name
    assert base_url(model) == url


def test_only_known_providers_count_as_prefixes() -> None:
    judge = create_judge("ft:gpt-4o-mini:acme::abc123")

    assert judge.name == "openai:ft:gpt-4o-mini:acme::abc123"


def test_the_base_url_variable_replaces_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_BASE_URL", "http://gpu-box:8000/v1")

    assert base_url("vllm:my-model") == "http://gpu-box:8000/v1/"
    assert base_url("vllm:my-model", base_url="http://other:8000/v1") == "http://other:8000/v1/"


def test_every_provider_reads_prefixed_variables() -> None:
    assert (PROVIDERS["lmstudio"].base_url_variable, PROVIDERS["lmstudio"].api_key_variable) == (
        "LMSTUDIO_BASE_URL",
        "LMSTUDIO_API_KEY",
    )


def test_missing_keys_and_endpoints_are_named(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY")
    monkeypatch.delenv("AZURE_OPENAI_BASE_URL")
    monkeypatch.delenv("AZURE_OPENAI_API_KEY")

    with pytest.raises(JudgeError, match=r"^missing environment variable: GROQ_API_KEY$"):
        create_judge("groq:llama-3.3-70b-versatile")
    with pytest.raises(
        JudgeError,
        match=r"^missing environment variables: AZURE_OPENAI_BASE_URL, AZURE_OPENAI_API_KEY$",
    ):
        create_judge("azure:my-deployment")


def test_local_servers_need_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_API_KEY")

    assert create_judge("vllm:my-model").name == "vllm:my-model"
    assert create_judge("lmstudio:my-model").name == "lmstudio:my-model"


def test_providers_without_json_schema_support_are_marked() -> None:
    assert not PROVIDERS["deepseek"].json_schema
    assert all(PROVIDERS[name].json_schema for name in ("openai", "groq", "gemini", "ollama"))
