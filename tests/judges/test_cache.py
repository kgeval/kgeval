"""Test caching judge verdicts on disk."""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from kgeval.errors import JudgeError
from kgeval.judges.cache import CachedJudge
from kgeval.metrics.judged.facts import SupportVerdict
from tests.doubles import ScriptedJudge


def test_cached_verdicts_are_reused_across_instances(tmp_path: Path) -> None:
    inner = ScriptedJudge()
    path = tmp_path / "verdicts.sqlite"

    first = CachedJudge(inner, path).ask("Is it so?", SupportVerdict)
    second = CachedJudge(inner, path).ask("Is it so?", SupportVerdict)

    assert first == second
    assert inner.prompts == ["Is it so?"]


def test_new_prompts_reach_the_wrapped_judge(tmp_path: Path) -> None:
    inner = ScriptedJudge()
    judge = CachedJudge(inner, tmp_path / "verdicts.sqlite")

    judge.ask("First?", SupportVerdict)
    judge.ask("Second?", SupportVerdict)

    assert judge.name == "scripted"
    assert inner.prompts == ["First?", "Second?"]


def test_invalid_cached_verdicts_are_judged_again_and_replaced(tmp_path: Path) -> None:
    inner = ScriptedJudge()
    path = tmp_path / "verdicts.sqlite"
    judge = CachedJudge(inner, path)
    judge.ask("Is it so?", SupportVerdict)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("UPDATE verdicts SET answer = '{}'")

    judge.ask("Is it so?", SupportVerdict)
    judge.ask("Is it so?", SupportVerdict)

    assert inner.prompts == ["Is it so?", "Is it so?"]


def test_an_unusable_cache_file_raises_a_judge_error(tmp_path: Path) -> None:
    with pytest.raises(JudgeError, match="cannot use the judge cache"):
        CachedJudge(ScriptedJudge(), tmp_path)
