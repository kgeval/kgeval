"""Implement the model that metrics ask when rules cannot decide."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import closing, suppress
from pathlib import Path
from typing import Any, TypeVar

import openai
import tenacity
from pydantic import BaseModel, ValidationError

from kgeval.errors import ModelError, VerdictError

__all__ = ["Model"]

_OutputT = TypeVar("_OutputT", bound=BaseModel)


class Model:
    """Get verdicts on evaluation prompts from any model that LiteLLM supports.

    Each request asks for the verdict's JSON schema as structured output. Requests that
    hit a rate limit or a server error are retried for up to about a minute.

    Args:
        model: The model as LiteLLM names it, ``provider/model``, such as
            ``openai/gpt-5-mini``.
        base_url: The server's API base URL; the provider's when omitted.
        api_key: The API key; read from the provider's environment variable when omitted.
        name: Identifies the model in metric signatures and the cache; ``model`` when
            omitted. The settings follow it, since they change the verdicts.
        cache: A SQLite file that keeps every verdict, so reruns reuse them; created when
            missing. A verdict is keyed by the model's name, the verdict's JSON schema, and
            the prompt. Delete the file to ask again.
        **settings: Arguments for every request, such as ``temperature=0`` or
            ``reasoning_effort="low"``.

    Raises:
        ModelError: If LiteLLM supports no provider of that name, the provider's API key
            or base URL is required but not set, or the cache file cannot be opened.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        name: str | None = None,
        cache: str | Path | None = None,
        **settings: Any,
    ) -> None:
        # Load LiteLLM's bundled model map, not the latest from GitHub, so runs are reproducible.
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        # Import on first use, since LiteLLM takes seconds to import.
        import litellm

        # Keep standard output clean for reports, since LiteLLM prints help text on errors.
        litellm.suppress_debug_info = True
        try:
            litellm.get_llm_provider(model, api_base=base_url)
        except openai.OpenAIError as error:
            raise ModelError(
                f"unknown model {model}; name it as provider/model, such as openai/gpt-5-mini"
            ) from error
        environment = litellm.validate_environment(model, api_key=api_key, api_base=base_url)
        if environment["missing_keys"]:
            missing = ", ".join(environment["missing_keys"])
            raise ModelError(f"missing environment variables: {missing}")
        self.name = (name or model) + "".join(
            f"|{key}={json.dumps(settings[key])}" for key in sorted(settings)
        )
        # Wait at most 10 minutes per attempt, as the OpenAI SDK does, not LiteLLM's 100.
        self._arguments = {
            "model": model,
            "api_base": base_url,
            "api_key": api_key,
            "timeout": 600,
            **settings,
        }
        self._cache = cache
        if cache is not None:
            _execute(
                cache,
                "CREATE TABLE IF NOT EXISTS verdicts (key TEXT PRIMARY KEY, verdict TEXT NOT NULL)",
            )

    def ask(self, prompt: str, output_type: type[_OutputT]) -> _OutputT:
        """Return the model's verdict on a prompt, from the cache when it has one.

        Args:
            prompt: The question for the model.
            output_type: The Pydantic model class the verdict must validate against.

        Returns:
            The validated verdict.

        Raises:
            VerdictError: If the server rejects the request as a bad one, as when the
                prompt exceeds the context window, or the verdict does not validate.
            ModelError: If the request fails otherwise, or the cache cannot be used.
        """
        if self._cache is None:
            return self._request(prompt, output_type)
        key = hashlib.sha256(
            json.dumps([self.name, output_type.model_json_schema(), prompt]).encode()
        ).hexdigest()
        rows = _execute(self._cache, "SELECT verdict FROM verdicts WHERE key = ?", key)
        if rows:
            # Ask again when a stored verdict no longer validates; the insert below replaces it.
            with suppress(ValidationError):
                return output_type.model_validate_json(rows[0][0])
        verdict = self._request(prompt, output_type)
        _execute(
            self._cache,
            "INSERT OR REPLACE INTO verdicts VALUES (?, ?)",
            key,
            verdict.model_dump_json(),
        )
        return verdict

    def _request(self, prompt: str, output_type: type[_OutputT]) -> _OutputT:
        try:
            response = _complete(
                messages=[{"role": "user", "content": prompt}],
                response_format=output_type,
                **self._arguments,
            )
            return output_type.model_validate_json(response.choices[0].message.content or "")
        except (openai.BadRequestError, ValidationError) as error:
            raise VerdictError(
                f"{self.name} gave no usable verdict for a prompt: {error}"
            ) from error
        except openai.OpenAIError as error:
            raise ModelError(f"{self.name} request failed: {error}") from error


def _is_transient(error: BaseException) -> bool:
    # Check the status, since LiteLLM raises 502 and 503 as no InternalServerError.
    status = getattr(error, "status_code", None)
    return isinstance(error, (openai.RateLimitError, openai.APIConnectionError)) or (
        isinstance(status, int) and status >= 500
    )


@tenacity.retry(
    retry=tenacity.retry_if_exception(_is_transient),
    wait=tenacity.wait_random_exponential(max=10),
    stop=tenacity.stop_after_attempt(10),
    reraise=True,
)
def _complete(**arguments: Any) -> Any:
    import litellm

    return litellm.completion(**arguments)


def _execute(path: str | Path, sql: str, *parameters: str) -> list[tuple[str, ...]]:
    # Open a connection per statement, so that concurrent requests never share one.
    try:
        with closing(sqlite3.connect(path)) as connection, connection:
            return connection.execute(sql, parameters).fetchall()
    except sqlite3.Error as error:
        raise ModelError(f"cannot use the model cache {path}: {error}") from error
