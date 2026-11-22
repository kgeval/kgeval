"""Define the exceptions that kgeval raises."""

__all__ = ["InputError", "JudgeError", "KgevalError", "SourceError", "VerdictError"]


class KgevalError(Exception):
    """Signal any failure raised by kgeval itself."""


class SourceError(KgevalError):
    """Signal that a graph source could not load the graph or run a query."""


class InputError(KgevalError):
    """Signal that a user-supplied file or argument is invalid."""


class JudgeError(KgevalError):
    """Signal that an LLM judge cannot be set up or fails to return a usable verdict."""


class VerdictError(JudgeError):
    """Signal that a judge gave no usable verdict for one prompt, though it can judge others.

    Judged metrics retry such a prompt once, then leave its unit out as unjudged, while
    any other ``JudgeError``, such as a rejected API key, fails the metric.
    """
