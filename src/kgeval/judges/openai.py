"""Judge with any model served through the OpenAI API."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TypeVar

import openai
from pydantic import BaseModel, ValidationError

from kgeval.errors import JudgeError, VerdictError

__all__ = ["OpenAIJudge"]

_OutputT = TypeVar("_OutputT", bound=BaseModel)

_JSON_INSTRUCTION = "Reply with only a JSON object that matches this JSON schema:\n{schema}"


class OpenAIJudge:
    """Judge with a model served through the OpenAI API, by OpenAI or any compatible server.

    Most providers and local servers speak the OpenAI API, so this one judge covers them
    all; see ``create_judge`` for ready-made provider settings. Each request asks for
    structured output. Servers that accept JSON schemas get the answer's schema as a
    response format, so answers always parse; for servers that accept only JSON objects,
    set ``json_schema=False`` and the schema goes into the prompt and the answer is
    validated on arrival.

    A prompt the model declines, answers unusably or that the server rejects as a bad
    request raises ``VerdictError``, since other prompts may still succeed; any other
    failure, such as a rejected API key, raises ``JudgeError``.

    Args:
        model: The model ID, as the server names it.
        base_url: The server's API base URL; OpenAI's when omitted.
        api_key: The API key; read from ``OPENAI_API_KEY`` when omitted.
        json_schema: Whether the server accepts JSON schema response formats.
        name: Identifies the judge in metric signatures and the judge cache; the model
            ID when omitted. The fields of ``request_body`` follow it, since they change
            the verdicts.
        client: The client to use; built from the other arguments when omitted.
        request_body: Fields added to every request, such as
            ``{"temperature": 0, "seed": 7}`` or ``{"reasoning_effort": "low"}``.

    Raises:
        JudgeError: If the client rejects its settings.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        json_schema: bool = True,
        name: str | None = None,
        client: openai.OpenAI | None = None,
        request_body: Mapping[str, object] | None = None,
    ) -> None:
        body = dict(request_body or {})
        settings = "".join(f"|{key}={json.dumps(body[key])}" for key in sorted(body))
        self.name = (name or model) + settings
        self._model = model
        self._json_schema = json_schema
        self._body = body
        if client is None:
            try:
                client = openai.OpenAI(base_url=base_url, api_key=api_key)
            except openai.OpenAIError as error:
                raise JudgeError(f"cannot create a client for {self.name}: {error}") from error
        self._client = client

    def ask(self, prompt: str, output_type: type[_OutputT]) -> _OutputT:
        """Answer a prompt with an instance of ``output_type``.

        Args:
            prompt: The question for the model.
            output_type: The Pydantic model class the answer must validate against.

        Returns:
            The validated answer.

        Raises:
            VerdictError: If the model declines, no answer parses, or the server rejects
                the request as a bad one, as when the prompt exceeds the context window.
            JudgeError: If the request fails otherwise.
        """
        try:
            if self._json_schema:
                return self._ask_with_schema(prompt, output_type)
            return self._ask_for_json(prompt, output_type)
        except (
            openai.BadRequestError,
            openai.LengthFinishReasonError,
            openai.ContentFilterFinishReasonError,
        ) as error:
            raise VerdictError(f"{self.name} could not judge a prompt: {error}") from error
        except openai.OpenAIError as error:
            raise JudgeError(f"{self.name} request failed: {error}") from error
        except ValidationError as error:
            raise VerdictError(
                f"{self.name} gave an invalid {output_type.__name__}: {error}"
            ) from error

    def _ask_with_schema(self, prompt: str, output_type: type[_OutputT]) -> _OutputT:
        completion = self._client.chat.completions.parse(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_format=output_type,
            extra_body=self._body,
        )
        message = completion.choices[0].message
        if message.refusal:
            raise VerdictError(f"{self.name} declined to judge: {message.refusal}")
        if message.parsed is None:
            reason = completion.choices[0].finish_reason
            raise VerdictError(
                f"{self.name} gave no {output_type.__name__} (finish reason {reason})"
            )
        return message.parsed

    def _ask_for_json(self, prompt: str, output_type: type[_OutputT]) -> _OutputT:
        schema = json.dumps(output_type.model_json_schema())
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _JSON_INSTRUCTION.format(schema=schema)},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            extra_body=self._body,
        )
        return output_type.model_validate_json(completion.choices[0].message.content or "")
