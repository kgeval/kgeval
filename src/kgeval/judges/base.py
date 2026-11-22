"""Define the interface that every LLM judge implements."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

__all__ = ["Judge"]

_OutputT = TypeVar("_OutputT", bound=BaseModel)


@runtime_checkable
class Judge(Protocol):
    """Answer evaluation prompts with structured output.

    Implement this to judge with any model. ``name`` identifies the judge in metric
    signatures, so results from different judges are never mistaken for each other.

    Attributes:
        name: Identifies the judge in metric signatures.
    """

    name: str

    def ask(self, prompt: str, output_type: type[_OutputT]) -> _OutputT:
        """Answer a prompt with an instance of ``output_type``.

        Implementations must be safe to call from several threads at once.

        Args:
            prompt: The question for the model.
            output_type: The Pydantic model class the answer must validate against.

        Returns:
            The validated answer.

        Raises:
            JudgeError: If the model fails or declines to answer.
        """
        ...
