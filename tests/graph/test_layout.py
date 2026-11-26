"""Test the layout that assigns each node its role in a knowledge graph."""

from kgeval.graph.layout import Layout, Role
from tests.factories import graph, node, relationship


def test_detect_reads_names_from_id_with_the_entity_base_label_or_without_other_names() -> None:
    marked = Layout.detect(graph(node("a", "__Entity__", "Person", name="Ada")))
    named = Layout.detect(graph(node("a", "Person", name="Ada", id="p1"), node("b", "Person")))
    unnamed = Layout.detect(
        graph(node("a", "Person", id="Ada"), node("c", "Chunk", text="Ada.", title="1"))
    )

    assert marked.name_keys == ("name", "title", "id")
    assert named.name_keys == ("name", "title")
    # Ignore lexical nodes: a titled chunk says nothing about where entities keep names.
    assert unnamed.name_keys == ("name", "title", "id")


def test_detect_keeps_a_lone_entity_label_as_a_node_type_without_graphiti_episodes() -> None:
    assert Layout.detect(graph(node("a", "Entity"))).base_labels == frozenset()


def test_detect_keeps_node_types_named_like_lexical_or_community_labels() -> None:
    contracts = graph(
        node("ann", "Person", name="Ann"),
        node("deal", "Document", title="Contract 7"),
        node("club", "Community", name="Chess club"),
        relationship("signed", "ann", "SIGNED", "deal"),
        relationship("member", "ann", "MEMBER_OF", "club"),
    )

    layout = Layout.detect(contracts)

    # Read nodes as lexical only when they hold text or join nodes that do.
    assert (layout.lexical_labels, layout.community_labels) == (frozenset(), frozenset())
    assert layout.role(contracts.nodes[1]) is Role.ENTITY


def test_detect_keeps_the_overridden_fields_and_detects_with_their_keys() -> None:
    passages = graph(
        node("c", "Chunk", body="Ada worked with Babbage."),
        node("a", "Person", label="Ada"),
        relationship("h", "c", "HAS_ENTITY", "a"),
    )

    layout = Layout.detect(passages, Layout(text_keys=("body",), name_keys=("label",)))

    assert layout.lexical_labels == {"Chunk"}
    assert (layout.text_keys, layout.name_keys) == (("body",), ("label",))


def test_every_node_is_an_entity_without_base_labels() -> None:
    layout = Layout.detect(graph(node("p", "Person"), node("m", "Movie")))

    assert layout.role(node("p", "Person")) is Role.ENTITY
    assert layout.role(node("x")) is Role.ENTITY


def test_lexical_role_wins_over_shared_base_labels() -> None:
    layout = Layout(lexical_labels=frozenset({"Chunk"}), base_labels=frozenset({"__KGBuilder__"}))

    assert layout.role(node("c", "Chunk", "__KGBuilder__")) is Role.LEXICAL
    assert layout.role(node("e", "Person", "__KGBuilder__")) is Role.ENTITY


def test_nodes_without_a_base_label_are_ignored_when_base_labels_exist() -> None:
    layout = Layout(base_labels=frozenset({"__Entity__"}))

    assert layout.role(node("b", "_Bloom_Perspective_")) is Role.OTHER


def test_node_types_leave_out_base_labels() -> None:
    layout = Layout(base_labels=frozenset({"__Entity__"}))

    assert layout.node_types(node("a", "Scientist", "__Entity__", "Person")) == (
        "Person",
        "Scientist",
    )


def test_layout_serializes_label_sets_in_sorted_order() -> None:
    layout = Layout(lexical_labels=frozenset({"Document", "Chunk", "Page"}))

    assert layout.model_dump(mode="json")["lexical_labels"] == ["Chunk", "Document", "Page"]
