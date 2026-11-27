"""Implement the normalization of entity and type names, so spelling variants compare equal."""

from __future__ import annotations

import re
import unicodedata

__all__ = ["normalize_name", "normalize_type"]

# Keep the symbols that tell names apart, as in C, C++, and C#, rather than separate words.
_SEPARATORS = re.compile(r"[^\w+#]+|_+")

_IRREGULAR_PLURALS = {"children": "child", "men": "man", "people": "person", "women": "woman"}

# Keep the words whose singular ends in s, so that News and New stay apart.
_INVARIANT = frozenset({"means", "news", "series", "species"})

# Strip the es of plurals that add it after a sibilant, as in Classes and Matches.
_SIBILANT_PLURALS = ("sses", "shes", "ches", "xes", "zzes")


def normalize_name(name: str) -> str:
    """Return the key under which entity names count as the same name.

    The key is the name case-folded and NFKD-decomposed, without combining marks or
    invisible format characters such as soft hyphens, and with every run of characters
    other than letters, digits, ``+``, and ``#`` replaced by one space, then trimmed.
    ``Lee Kuan-Yew`` and ``lee kuan yew`` share a key, while ``C``, ``C++``, and ``C#``
    keep apart.

    Args:
        name: An entity name.

    Returns:
        The key; empty when the name has no letter or digit.
    """
    decomposed = unicodedata.normalize("NFKD", name.casefold())
    plain = "".join(
        char
        for char in decomposed
        if not unicodedata.combining(char) and unicodedata.category(char) != "Cf"
    )
    if not any(char.isalnum() for char in plain):
        return ""
    return " ".join(token for token in _SEPARATORS.split(plain) if token)


def normalize_type(type_name: str) -> str:
    """Return the key under which node types or relationship types count as the same type.

    The key splits camel case into words, normalizes the result as ``normalize_name``
    does, and reduces every word to a form its singular and plural share. By the English
    plural rules, ``ies`` and ``ie`` become ``y``, ``es`` after a sibilant and a final
    ``s`` are dropped, except after ``s``, ``u``, or ``i`` and in words of three letters
    or fewer, and a few irregular plurals, such as ``people``, map to their singulars.
    ``BornIn``, ``BORN_IN``, and ``born in`` share a key, and so do ``Organizations``
    and ``ORGANIZATION``, ``Companies`` and ``Company``, ``Classes`` and ``Class``, and
    ``People`` and ``Person``.

    Args:
        type_name: A node type or relationship type.

    Returns:
        The key.
    """
    words = normalize_name(_split_camel_case(type_name)).split()
    return " ".join(_singular(word) for word in words)


def _singular(word: str) -> str:
    """Return the form that a lowercase English word shares with its plural."""
    if word in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[word]
    if len(word) <= 3 or word in _INVARIANT:
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("ie"):
        return word[:-2] + "y"
    if word.endswith(_SIBILANT_PLURALS):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _split_camel_case(text: str) -> str:
    """Insert a space before every capital that starts a word, as in ``HTTPServerError``."""
    chars: list[str] = []
    for index, char in enumerate(text):
        previous = text[index - 1] if index else ""
        following = text[index + 1 : index + 2]
        if char.isupper() and (
            previous.islower() or previous.isdigit() or (previous.isupper() and following.islower())
        ):
            chars.append(" ")
        chars.append(char)
    return "".join(chars)
