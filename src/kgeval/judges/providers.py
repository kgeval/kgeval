"""Name the common LLM providers, and create judges from ``provider:model`` names."""

from __future__ import annotations

import os
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from kgeval.errors import JudgeError
from kgeval.judges.openai import OpenAIJudge

__all__ = ["PROVIDERS", "Provider", "create_judge"]


class Provider(BaseModel):
    """Describe how to reach one provider through the OpenAI API.

    The environment variables ``{prefix}_BASE_URL`` and ``{prefix}_API_KEY`` hold the
    provider's base URL and API key.

    Attributes:
        prefix: The prefix of the provider's environment variables, such as ``OPENAI``.
        base_url: The base URL used when ``{prefix}_BASE_URL`` is unset; ``None`` when
            the variable must be set, as for an Azure resource.
        requires_api_key: Whether requests fail without a key; local servers need none.
        json_schema: Whether the provider accepts JSON schema response formats, rather
            than only JSON objects.
    """

    model_config = ConfigDict(frozen=True)

    prefix: str
    base_url: str | None
    requires_api_key: bool = True
    json_schema: bool = True

    @property
    def base_url_variable(self) -> str:
        """Return the environment variable that holds the base URL."""
        return f"{self.prefix}_BASE_URL"

    @property
    def api_key_variable(self) -> str:
        """Return the environment variable that holds the API key."""
        return f"{self.prefix}_API_KEY"

    def resolve_base_url(self) -> str:
        """Return the provider's base URL from the environment, or its default.

        Returns:
            The base URL.

        Raises:
            JudgeError: If the base URL variable is unset and there is no default.
        """
        base_url = os.environ.get(self.base_url_variable) or self.base_url
        if base_url is None:
            raise _missing([self.base_url_variable])
        return base_url

    def resolve_api_key(self) -> str:
        """Return the provider's API key from the environment.

        Returns:
            The key, or a placeholder for a server that needs none.

        Raises:
            JudgeError: If a required key is not set.
        """
        if key := os.environ.get(self.api_key_variable):
            return key
        if self.requires_api_key:
            raise _missing([self.api_key_variable])
        # Pass a placeholder: the client insists on a key even for servers that need none.
        return "unused"


def _missing(variables: list[str]) -> JudgeError:
    noun = "variable" if len(variables) == 1 else "variables"
    return JudgeError(f"missing environment {noun}: {', '.join(variables)}")


PROVIDERS: dict[str, Provider] = {
    "openai": Provider(prefix="OPENAI", base_url="https://api.openai.com/v1"),
    "azure": Provider(prefix="AZURE_OPENAI", base_url=None),
    "gemini": Provider(
        prefix="GEMINI", base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    ),
    "mistral": Provider(prefix="MISTRAL", base_url="https://api.mistral.ai/v1"),
    "xai": Provider(prefix="XAI", base_url="https://api.x.ai/v1"),
    "deepseek": Provider(prefix="DEEPSEEK", base_url="https://api.deepseek.com", json_schema=False),
    "groq": Provider(prefix="GROQ", base_url="https://api.groq.com/openai/v1"),
    "together": Provider(prefix="TOGETHER", base_url="https://api.together.xyz/v1"),
    "fireworks": Provider(prefix="FIREWORKS", base_url="https://api.fireworks.ai/inference/v1"),
    "openrouter": Provider(prefix="OPENROUTER", base_url="https://openrouter.ai/api/v1"),
    "ollama": Provider(
        prefix="OLLAMA", base_url="http://localhost:11434/v1", requires_api_key=False
    ),
    "lmstudio": Provider(
        prefix="LMSTUDIO", base_url="http://localhost:1234/v1", requires_api_key=False
    ),
    "vllm": Provider(prefix="VLLM", base_url="http://localhost:8000/v1", requires_api_key=False),
}
"""The providers kgeval knows by name, with their OpenAI API settings."""


def create_judge(
    model: str,
    *,
    base_url: str | None = None,
    request_body: Mapping[str, object] | None = None,
) -> OpenAIJudge:
    """Create a judge from a model ID with an optional provider prefix.

    The text before the first colon selects a provider when it is a key of
    ``PROVIDERS``; otherwise all of ``model`` is an OpenAI model ID. The base URL and
    API key come from the provider's environment variables.

    Args:
        model: The model ID, optionally prefixed with a provider name and a colon.
        base_url: A base URL that replaces the provider's.
        request_body: Fields added to every request, such as ``{"temperature": 0}``.

    Returns:
        The judge, named ``provider:model`` in metric signatures, followed by the fields
        of ``request_body``.

    Raises:
        JudgeError: If the provider's API key or base URL is required but not set, or the
            client rejects the settings.
    """
    prefix, separator, rest = model.partition(":")
    if separator and rest and prefix in PROVIDERS:
        provider_name, model_id = prefix, rest
    else:
        provider_name, model_id = "openai", model
    provider = PROVIDERS[provider_name]
    # Check every variable before resolving any, so one error names all that are missing.
    missing = [
        variable
        for variable, required in (
            (provider.base_url_variable, base_url is None and provider.base_url is None),
            (provider.api_key_variable, provider.requires_api_key),
        )
        if required and not os.environ.get(variable)
    ]
    if missing:
        raise _missing(missing)
    return OpenAIJudge(
        model_id,
        base_url=base_url or provider.resolve_base_url(),
        api_key=provider.resolve_api_key(),
        json_schema=provider.json_schema,
        name=f"{provider_name}:{model_id}",
        request_body=request_body,
    )
