"""Test the view that indexes a graph's entities, facts, and chunks."""

from kgeval.graph.layout import Layout
from tests.factories import example_view, node, relationship, view


def test_view_separates_facts_from_provenance_and_lexical_links() -> None:
    example = example_view()

    assert [fact.id for fact in example.facts] == [
        "attended",
        "born_in",
        "born_in_copy",
        "duplicate_founded",
        "founded_pap",
        "founded_unnamed",
        "he_founded",
        "knows",
        "married",
        "raffles_founded",
        "studied_at",
    ]
    assert {relationship.type for relationship in example.provenance} == {"HAS_ENTITY"}
    assert (len(example.entities), len(example.lexical_nodes), len(example.provenance)) == (
        11,
        7,
        14,
    )


def test_every_node_is_an_entity_in_a_graph_without_builder_labels() -> None:
    movies = view(node("p", "Person"), node("m", "Movie"), relationship("r", "p", "ACTED_IN", "m"))

    assert [entity.id for entity in movies.entities] == ["m", "p"]
    assert [fact.id for fact in movies.facts] == ["r"]


def test_chunk_ids_follow_provenance_relationships_in_either_direction() -> None:
    grounded = view(
        node("c1", "Chunk", text="Ada."),
        node("c2", "Chunk", text="Ada again."),
        node("e", "__Entity__", "Person"),
        relationship("r1", "c1", "HAS_ENTITY", "e"),
        relationship("r2", "e", "FROM_CHUNK", "c2"),
    )

    assert grounded.chunk_ids("e") == {"c1", "c2"}
    assert grounded.chunk_ids("c1") == frozenset()


def test_facts_are_grounded_in_the_chunks_that_ground_both_entities() -> None:
    example = example_view()

    facts = {fact.id: fact for fact in example.facts}

    assert example.fact_chunk_ids(facts["born_in"]) == {"chunk1"}
    assert example.fact_chunk_ids(facts["attended"]) == {"chunk4"}
    assert example.fact_chunk_ids(facts["duplicate_founded"]) == frozenset()


def test_community_membership_is_neither_fact_nor_provenance() -> None:
    members = view(
        node("k", "Community"),
        node("a", "Person"),
        node("b", "Person"),
        relationship("r1", "a", "IN_COMMUNITY", "k"),
        relationship("r2", "k", "HAS_MEMBER", "b"),
        layout=Layout(community_labels=frozenset({"Community"})),
    )

    assert [community.id for community in members.community_nodes] == ["k"]
    assert members.facts == members.provenance == ()


def test_expired_facts_are_kept_apart_from_current_facts() -> None:
    history = view(
        node("e", "Episodic", content="Ada left Acme for Initech."),
        node("ada", "Entity", "Person", name="Ada"),
        node("acme", "Entity", "Company", name="Acme"),
        node("initech", "Entity", "Company", name="Initech"),
        relationship("left", "ada", "RELATES_TO", "acme", name="WORKS_AT", expired_at="2024"),
        relationship("joined", "ada", "RELATES_TO", "initech", name="WORKS_AT"),
    )

    assert [fact.id for fact in history.facts] == ["joined"]
    assert [fact.id for fact in history.expired_facts] == ["left"]


def test_chunks_are_lexical_nodes_with_a_label_of_a_grounding_node() -> None:
    example = example_view()

    # Count chunk3, which grounds nothing, and leave out the documents.
    assert [chunk.id for chunk in example.chunks] == [
        "chunk1",
        "chunk2",
        "chunk3",
        "chunk4",
        "chunk5",
    ]


def test_chunks_of_documents_include_selected_chunks_and_those_joined_to_other_selections() -> None:
    graph = view(
        node("d1", "Document", fileName="a.pdf"),
        node("d2", "Document", fileName="b.pdf"),
        node("c1", "Chunk", fileName="a.pdf"),
        node("c2", "Chunk"),
        node("c3", "Chunk"),
        node("e", "Person"),
        relationship("p1", "c2", "PART_OF", "d1"),
        relationship("p2", "c3", "PART_OF", "d2"),
        # Keep chunks selected directly from widening the selection to their neighbors.
        relationship("s", "c1", "SIMILAR", "c3"),
        *(relationship(f"h{number}", f"c{number}", "HAS_ENTITY", "e") for number in (1, 2, 3)),
        layout=Layout(lexical_labels=frozenset({"Chunk", "Document"})),
    )

    assert graph.chunks_of([{"fileName": "a.pdf"}]) == {"c1", "c2"}
    assert graph.chunks_of([{"fileName": "a.pdf"}, {"fileName": "b.pdf"}]) == {"c1", "c2", "c3"}
    assert graph.chunks_of([{"fileName": "c.pdf"}]) == frozenset()


def test_name_is_the_first_non_blank_name_key() -> None:
    ada = view(
        node("a", name="  ", title=7, label=" Ada "),
        layout=Layout(name_keys=("name", "title", "label")),
    )

    assert ada.name(ada.entities[0]) == "Ada"


def test_relationship_type_and_description_come_from_the_layouts_keys() -> None:
    graph = view(
        node("a", name="Ada"),
        node("b", name="Bob"),
        relationship("r1", "a", "RELATES_TO", "b", name=" WORKS_WITH ", fact="Ada works with Bob."),
        relationship("r2", "a", "RELATES_TO", "b", name=" "),
        layout=Layout(type_keys=("name",)),
    )

    first, second = graph.facts
    assert (graph.relationship_type(first), graph.relationship_type(second)) == (
        "WORKS_WITH",
        "RELATES_TO",
    )
    assert (graph.description(first), graph.description(second)) == ("Ada works with Bob.", None)


def test_describe_names_nodes_and_relationships_by_content() -> None:
    example = example_view()

    assert example.describe(example.node("lee")) == '"Lee Kuan Yew" (Person)'
    assert example.describe(example.node("chunk1")) == (
        '"Lee Kuan Yew was born in Singapore. He co-founded the Peopl…" (Chunk)'
    )
    assert example.describe(example.facts[1]) == (
        '"Lee Kuan Yew" (Person) -[:BORN_IN]-> "Singapore" (Location)'
    )


def test_describe_never_shows_element_ids() -> None:
    graph = view(node("4:a1b2:9", "Organization"), node("4:a1b2:10"))

    assert [graph.describe(node) for node in graph.entities] == [
        "unnamed",
        "unnamed (Organization)",
    ]
