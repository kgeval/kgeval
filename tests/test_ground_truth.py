"""Test the ground truth that lists the facts a graph is expected to contain."""

from pathlib import Path

import pytest

from kgeval.errors import InputError
from kgeval.ground_truth import GroundTruth, load_ground_truth
from tests.factories import EXAMPLE_DIR, example_view


def write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "ground_truth.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_ground_truth_reads_the_example_ground_truth() -> None:
    ground_truth = load_ground_truth(EXAMPLE_DIR / "ground_truth.yaml")

    assert ground_truth.facts[0] == ("Lee Kuan Yew", "BORN_IN", "Singapore")
    assert len(ground_truth.facts) == 4
    assert ground_truth.documents == ({"fileName": "singapore.txt"},)
    assert ground_truth.symmetric_relationship_types == ("MARRIED",)


def test_load_ground_truth_reads_a_bare_list_of_ground_truth_facts(tmp_path: Path) -> None:
    ground_truth = load_ground_truth(write(tmp_path, "- [Ada, KNOWS, Bob]\n"))

    assert ground_truth == GroundTruth(facts=(("Ada", "KNOWS", "Bob"),))


@pytest.mark.parametrize(
    "content",
    [
        "[]\n",
        "- [Lee Kuan Yew, BORN_IN]\n",
        "- [Lee Kuan Yew, '-', Singapore]\n",
        "documents: [{fileName: a.pdf}]\n",
        "facts: [[Ada, KNOWS, Bob]]\ndocuments: [{}]\n",
        "facts: [[Ada, KNOWS, Bob]]\nsymmetric_relationship_types: [MARRIED]\n",
        "facts: [[Ada, KNOWS, Bob]]\naliases: {Cy: [C]}\n",
        "facts: [[Ada, KNOWS, Bob]]\naliases: {Ada: [bob]}\n",
        "facts: [[Ada, KNOWS, Bob]]\naliases: {Ada: [A], Bob: [a]}\n",
        "facts: [[Ada, KNOWS, Bob]]\nrelationship_type_aliases: {MET: [SAW]}\n",
        "facts: [[Ada, KNOWS, Bob]]\nunknown: true\n",
    ],
)
def test_load_ground_truth_rejects_invalid_ground_truths(tmp_path: Path, content: str) -> None:
    with pytest.raises(InputError):
        load_ground_truth(write(tmp_path, content))


def test_keys_map_aliases_to_their_ground_truth_names_and_sort_symmetric_facts() -> None:
    ground_truth = GroundTruth(
        facts=(("Lee Kuan Yew", "MARRIED", "Kwa Geok Choo"),),
        aliases={"Lee Kuan Yew": ("Harry Lee", "LKY")},
        relationship_type_aliases={"MARRIED": ("WED",)},
        symmetric_relationship_types=("MARRIED",),
    )

    assert ground_truth.entity_key("harry lee") == ground_truth.entity_key("LKY") == "lee kuan yew"
    assert ground_truth.fact_key("Kwa Geok Choo", "Wed", "LKY") == (
        "kwa geok choo",
        "married",
        "lee kuan yew",
    )


def test_graph_fact_keys_read_names_and_types_through_the_layout() -> None:
    graph = example_view()
    ground_truth = load_ground_truth(EXAMPLE_DIR / "ground_truth.yaml")

    keys = {ground_truth.graph_fact_key(graph, fact) for fact in graph.facts}

    assert ("lee kuan yew", "born in", "singapore") in keys
    assert ("lee kuan yew", "founded", "") in keys
