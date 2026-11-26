"""Implement the test doubles of the model."""

import threading
import time
from collections.abc import Mapping
from typing import TypeVar

from pydantic import BaseModel

from kgeval.errors import ModelError, VerdictError
from kgeval.model import Model

OutputT = TypeVar("OutputT", bound=BaseModel)


class ScriptedModel(Model):
    """Approve every prompt except those containing one of the given texts; record every prompt.

    Approving verdicts set their flag and leave their lists empty; rejecting verdicts clear
    their flag and take their lists from ``rejection``.
    """

    name = "scripted"

    def __init__(self, *rejected: str, rejection: Mapping[str, object] | None = None) -> None:
        self.rejected = rejected
        self.rejection = dict(rejection or {})
        self.prompts: list[str] = []
        self._lock = threading.Lock()

    def ask(self, prompt: str, output_type: type[OutputT]) -> OutputT:
        with self._lock:
            self.prompts.append(prompt)
        rejected = any(text in prompt for text in self.rejected)
        verdict: dict[str, object] = {}
        for name, field in output_type.model_fields.items():
            if field.annotation is bool:
                verdict[name] = not rejected
            elif field.annotation is str:
                verdict[name] = "Scripted."
            else:
                verdict[name] = self.rejection.get(name, []) if rejected else []
        return output_type.model_validate(verdict)


class FailingModel(Model):
    """Fail on every prompt after an optional delay, and count how many it was asked."""

    name = "failing"

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls = 0
        self._lock = threading.Lock()

    def ask(self, prompt: str, output_type: type[OutputT]) -> OutputT:
        with self._lock:
            self.calls += 1
        time.sleep(self.delay)
        raise ModelError("model unavailable")


class UnusableModel(ScriptedModel):
    """Give no usable verdict for prompts containing one of the given texts, up to a count.

    Approve every other prompt, as ``ScriptedModel`` does with nothing rejected.
    """

    name = "unusable"

    def __init__(self, *unusable: str, failures: int = 2) -> None:
        super().__init__()
        self.unusable = unusable
        self.failures: dict[str, int] = {}
        self.limit = failures

    def ask(self, prompt: str, output_type: type[OutputT]) -> OutputT:
        if any(text in prompt for text in self.unusable):
            with self._lock:
                count = self.failures[prompt] = self.failures.get(prompt, 0) + 1
            if count <= self.limit:
                raise VerdictError("unparsable verdict")
        return super().ask(prompt, output_type)
