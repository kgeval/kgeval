"""Test normalizing entity names and type names."""

import pytest

from kgeval.graph.names import normalize_name, normalize_type


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("Lee Kuan Yew", "lee kuan yew"),
        ("  LEE   kuan yew ", "lee kuan yew"),
        ("Lee-Kuan-Yew", "lee kuan yew"),
        ("Lée Kuan Yew", "lee kuan yew"),
        ("Apple Inc.", "apple inc"),
        ("snake_case", "snake case"),
        ("ﬁsh", "fish"),
        ("Straße", "strasse"),
        ("!!!", ""),
        ("C++", "c++"),
        ("C#", "c#"),
        ("++", ""),
    ],
)
def test_normalize_name_ignores_case_accents_and_punctuation(name: str, key: str) -> None:
    assert normalize_name(name) == key


@pytest.mark.parametrize(
    ("variants", "key"),
    [
        (["BORN_IN", "BornIn", "born in", "bornIn"], "born in"),
        (["WORKS_AT", "WorksAt", "WORK_AT"], "work at"),
        (["Organization", "ORGANIZATIONS", "organizations"], "organization"),
        (["HTTPServer", "HTTP_SERVER"], "http server"),
        (["Covid19Vaccine", "COVID19_VACCINE"], "covid19 vaccine"),
        (["Address", "ADDRESS", "Addresses"], "address"),
        (["HAS", "Has"], "has"),
        (["Company", "Companies", "COMPANY"], "company"),
        (["City", "Cities"], "city"),
        (["Person", "People", "PERSON"], "person"),
        (["Class", "Classes"], "class"),
        (["Match", "Matches"], "match"),
        (["Movie", "Movies"], "movy"),
        (["House", "Houses"], "house"),
        (["Status"], "status"),
    ],
)
def test_normalize_type_ignores_case_separators_camel_case_and_plurals(
    variants: list[str], key: str
) -> None:
    assert {normalize_type(variant) for variant in variants} == {key}


def test_normalize_type_keeps_different_words_apart() -> None:
    assert normalize_type("IS_PART_OF") != normalize_type("PART_OF")
    assert normalize_type("News") != normalize_type("New")
    assert normalize_type("Bus") != normalize_type("Bu")


def test_normalize_name_keeps_names_that_differ_by_a_symbol_apart() -> None:
    assert len({normalize_name(name) for name in ("C", "C++", "C#")}) == 3
