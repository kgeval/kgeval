"""Cache judge verdicts on disk, so repeated evaluations reuse them."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing, suppress
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from kgeval.errors import JudgeError
from kgeval.judges.base import Judge

__all__ = ["CachedJudge"]

_OutputT = TypeVar("_OutputT", bound=BaseModel)


class CachedJudge:
    """Answer from a SQLite file of earlier verdicts, asking the wrapped judge on a miss.

    A verdict is keyed by the judge's name, the answer's JSON schema, and the prompt, so
    no other judge, answer model or prompt reuses it. Cached verdicts make reruns
    reproducible and free; delete the file to judge again.

    Args:
        judge: The judge to ask on a cache miss.
        path: The SQLite file; created when missing.

    Raises:
        JudgeError: If the file cannot be opened as a cache.
    """

    def __init__(self, judge: Judge, path: str | Path) -> None:
        self.name = judge.name
        self._judge = judge
        self._path = Path(path)
        self._execute(
            "CREATE TABLE IF NOT EXISTS verdicts (key TEXT PRIMARY KEY, answer TEXT NOT NULL)"
        )

    def ask(self, prompt: str, output_type: type[_OutputT]) -> _OutputT:
        """Return the cached answer to a prompt, or ask the wrapped judge and cache it.

        Args:
            prompt: The question for the model.
            output_type: The Pydantic model class the answer must validate against.

        Returns:
            The validated answer.

        Raises:
            JudgeError: If the wrapped judge fails or the cache cannot be used.
        """
        key = hashlib.sha256(
            json.dumps([self.name, output_type.model_json_schema(), prompt]).encode()
        ).hexdigest()
        rows = self._execute("SELECT answer FROM verdicts WHERE key = ?", key)
        if rows:
            # Ask again when a stored verdict no longer validates; the insert below replaces it.
            with suppress(ValidationError):
                return output_type.model_validate_json(rows[0][0])
        answer = self._judge.ask(prompt, output_type)
        self._execute(
            "INSERT OR REPLACE INTO verdicts VALUES (?, ?)", key, answer.model_dump_json()
        )
        return answer

    def _execute(self, sql: str, *parameters: str) -> list[tuple[str, ...]]:
        # Open a connection per statement, so that concurrent judge threads never share one.
        try:
            with closing(sqlite3.connect(self._path)) as connection, connection:
                return connection.execute(sql, parameters).fetchall()
        except sqlite3.Error as error:
            raise JudgeError(f"cannot use the judge cache {self._path}: {error}") from error
