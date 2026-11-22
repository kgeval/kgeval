# Inputs

Inputs tell kgeval what the graph should contain and what a retriever returned, and
each one enables more [metrics](metrics.md). Files can be YAML or JSON.

| Input                             | CLI option     | Python                                                                     | Enables                                   |
| --------------------------------- | -------------- | -------------------------------------------------------------------------- | ----------------------------------------- |
| [Graph schema](#graph-schema)     | `--schema`     | [`load_schema`][kgeval.load_schema]                                        | Schema metrics                            |
| [Ground truth](#ground-truth)     | `--ground`     | [`load_ground_truth`][kgeval.load_ground_truth]                            | Fact and entity precision, recall, and F1 |
| [Test questions](#test-questions) | `--questions`  | [`load_questions`][kgeval.load_questions]                                  | `competency`                              |
| [Retrievals](#retrievals)         | `--retrievals` | [`load_retrievals`][kgeval.load_retrievals], [`retrieve`][kgeval.retrieve] | Retrieval metrics                         |

## Graph schema

A graph schema declares which node types, relationship types, and patterns the graph may
contain. kgeval reads neo4j-graphrag's `GraphSchema` format, so a schema written for
`SimpleKGPipeline` works as it is.

```yaml title="schema.yaml"
node_types:
  - label: Person
    properties:
      - {name: name, type: STRING, required: true}
      - {name: born, type: DATE}
  - Organization
  - Location
relationship_types:
  - {label: BORN_IN, max_targets: 1}
  - FOUNDED
patterns:
  - [Person, BORN_IN, Location]
  - [Person, FOUNDED, Organization]
constraints:
  - {type: UNIQUENESS, node_type: Person, property_names: [name]}
```

| Field                                                                           | Description                                                                                                       |
| ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `node_types`                                                                    | Node types. Each is a label, or a mapping with `label`, `description`, `properties`, and `additional_properties`. |
| `relationship_types`                                                            | Relationship types, in the same form as node types, plus an optional `max_targets`.                               |
| `patterns`                                                                      | Allowed combinations, each as `[source node type, relationship type, target node type]`.                          |
| `constraints`                                                                   | `EXISTENCE`, `UNIQUENESS`, or `KEY` constraints on the `property_names` of one `node_type` or `relationship_type`. |
| `additional_node_types`, `additional_relationship_types`, `additional_patterns` | Whether undeclared node types, relationship types, or patterns are allowed.                                       |

A property has a `name`, a `type`, and optionally a `description` and `required`. The
types are `BOOLEAN`, `DATE`, `DURATION`, `FLOAT`, `INTEGER`, `LIST`, `LOCAL_DATETIME`,
`LOCAL_TIME`, `POINT`, `STRING`, `ZONED_DATETIME`, and `ZONED_TIME`. An embedding,
which kgeval reads from any non-empty list of floats, has the type `LIST`.

As in neo4j-graphrag, declaring something closes it. Each `additional_*` field
defaults to `true` only when its list is empty, and `additional_properties` defaults to
`true` only when a type declares no properties.

`max_targets` is a kgeval extension: the most distinct entities that one entity may
relate to with that type. With `max_targets: 1` on `BORN_IN`, a person with two
birthplaces fails `fact_conformance`.

### Draft a schema

Without a schema, draft one from the graph as it is today:

```console
kgeval infer-schema -o schema.yaml
```

The draft copies the graph's defects, such as a `PERSON` node type next to `Person`.
Review the draft and remove those defects before using it.

## Ground truth

A ground truth lists the facts the graph should contain, each as
`[source entity name, relationship type, target entity name]`, written as a person
would name them:

```yaml title="ground_truth.yaml"
facts:
  - [Lee Kuan Yew, BORN_IN, Singapore]
  - [Lee Kuan Yew, FOUNDED, "People's Action Party"]
  - [Kwa Geok Choo, MARRIED, Lee Kuan Yew]
documents:
  - {fileName: singapore.txt}
aliases:
  Lee Kuan Yew: [LKY, Harry Lee]
relationship_type_aliases:
  BORN_IN: [PLACE_OF_BIRTH]
symmetric_relationship_types: [MARRIED]
```

| Field                          | Description                                                                                                                                                                    |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `facts`                        | The ground truth facts. Required.                                                                                                                                              |
| `documents`                    | The documents the ground truth covers, each selected by property values of a lexical node. When set, only facts and entities grounded in those documents' chunks are compared. |
| `aliases`                      | Other names the graph may give an entity.                                                                                                                                      |
| `relationship_type_aliases`    | Other relationship types the graph may use for a type.                                                                                                                         |
| `symmetric_relationship_types` | Relationship types whose facts match in either direction.                                                                                                                      |

A ground truth with only facts can be a plain list of them.

List every fact that the covered documents state. Precision counts any graph fact
missing from the ground truth as wrong.

## Test questions

Test questions are questions the graph must be able to answer. Each one lists
the facts and entities that a correct answer rests on:

```yaml title="questions.yaml"
- text: What did Lee Kuan Yew co-found?
  supporting_facts:
    - [Lee Kuan Yew, FOUNDED, "People's Action Party"]
    - [Lee Kuan Yew, FOUNDED, "Lee & Lee"]
- text: When was Lee Kuan Yew born?
  supporting_entities: [Lee Kuan Yew]
```

Each question needs a unique `text` and at least one supporting fact or entity. Use
`supporting_entities` for the entities an answer needs beyond those of its facts, such
as the entity whose property the question asks about.

To use `aliases`, `relationship_type_aliases`, or `symmetric_relationship_types`, write
a mapping and put the questions under `questions`:

```yaml title="questions.yaml"
questions:
  - text: Who did Lee Kuan Yew marry?
    supporting_facts:
      - [Lee Kuan Yew, MARRIED, Kwa Geok Choo]
symmetric_relationship_types: [MARRIED]
```

## Retrievals

Retrievals record what a retriever returned for each test question, as element
IDs of nodes and relationships in rank order, keyed by question text:

```json title="retrievals.json"
{
  "What did Lee Kuan Yew co-found?": ["4:9c1e…:4", "5:9c1e…:12"],
  "When was Lee Kuan Yew born?": ["4:9c1e…:4"]
}
```

Every question needs an entry, even if it is an empty list. Element IDs belong to one
database, so the retrievals must come from the database of the evaluated graph. IDs of
elements that are not in the graph are ignored with a warning, and retrievals with no ID
of the graph at all are rejected, since they come from another database or hold
something other than element IDs, such as chunk texts.

A pipeline in any language can write this file. In Python, `retrieve` builds the same
mapping from a function that takes a question and returns, in rank order, element IDs,
or the nodes, relationships, and paths that the Neo4j driver returns, alone or within
records and lists:

```python
import kgeval


def retriever(question: str) -> list[object]:
    with driver.session() as session:
        return list(session.run(RETRIEVAL_QUERY, question=question))


questions = kgeval.load_questions("questions.yaml")
retrievals = kgeval.retrieve(questions, retriever)
metrics = kgeval.default_metrics(questions=questions, retrievals=retrievals)
```

## Name matching

Ground truths and questions name entities and relationship types the way people write
them, and kgeval matches those names to the graph's loosely:

- **Entity names** match regardless of case, accents, punctuation, and spacing, so
  `Lee Kuan-Yew` matches `lee kuan yew`. `+` and `#` count, so `C`, `C++`, and `C#`
  stay apart.
- **Relationship types** also match regardless of separators, camel case, and plural
  form, so `BORN_IN`, `BornIn`, and `born in` all match, and so do `WORKS_AT` and
  `WORK_AT`.

For names that differ in other ways, add `aliases` or `relationship_type_aliases`.
