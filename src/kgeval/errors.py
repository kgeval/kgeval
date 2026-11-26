"""Implement the exceptions that kgeval raises."""

__all__ = ["InputError", "KgevalError", "ModelError", "SourceError", "VerdictError"]


class KgevalError(Exception):
    """Signal any failure raised by kgeval itself."""


class SourceError(KgevalError):
    """Signal that a graph source could not load the graph or run a query."""


class InputError(KgevalError):
    """Signal that a user-supplied file or argument is invalid."""


class ModelError(KgevalError):
    """Signal that a model cannot be set up or fails to return a usable verdict."""


class VerdictError(ModelError):
    """Signal that a model gave no usable verdict for one prompt, though others may succeed.

    Model metrics retry such a prompt once, then leave its unit out as undecided, while
    any other ``ModelError``, such as a rejected API key, fails the metric.
    """
