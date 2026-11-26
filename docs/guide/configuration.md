# Configuration

kgeval needs to know where the graph is and how it is laid out. It detects the layout
of graphs from well-known builders, so most graphs need only a connection.

## Connection

The CLI reads the connection from environment variables or options:

| Variable         | Option       | Default                  | Description                                                        |
| ---------------- | ------------ | ------------------------ | ------------------------------------------------------------------ |
| `NEO4J_URI`      | `--uri`      | `neo4j://localhost:7687` | The connection URI.                                                |
| `NEO4J_USERNAME` | `--username` | `neo4j`                  | The user.                                                          |
| `NEO4J_PASSWORD` | `--password` | None                     | The password. Without one, kgeval connects without authentication. |
| `NEO4J_DATABASE` | `--database` | The server's default     | The database to read.                                              |

In a terminal, kgeval prompts for the URI, user, and password when they are unset, and
offers the defaults. Elsewhere, such as in CI, it uses the defaults.

For Aura, use the `neo4j+s://` URI from the instance's connection details.

In Python, pass an open driver to [`Neo4jSource`][kgeval.Neo4jSource], and the database
name as its second argument if it is not the default:

```python
graph = kgeval.Neo4jSource(driver, "neo4j").load()
```

## Snapshots

A snapshot holds every node and relationship that kgeval reads from the graph. Save one
to evaluate the same graph again without a database:

=== "CLI"

    ```console
    kgeval snapshot -o graph.json
    kgeval run --graph graph.json
    ```

=== "Python"

    ```python
    Path("graph.json").write_text(graph.model_dump_json(), encoding="utf-8")
    graph = kgeval.load_graph("graph.json")
    ```

A snapshot file holds a [`Graph`][kgeval.Graph] as JSON, so a graph from another
database can be evaluated by writing one:

```json title="graph.json"
{
  "snapshot_format": 1,
  "nodes": [
    {"id": "lee", "labels": ["Person"], "properties": {"name": "Lee Kuan Yew"}},
    {"id": "singapore", "labels": ["Location"], "properties": {"name": "Singapore"}}
  ],
  "relationships": [
    {"id": "born_in", "type": "BORN_IN", "source_id": "lee", "target_id": "singapore"}
  ]
}
```

A property value is a string, a number, a boolean, one of these objects, or a list of
them:

- A date, time, or datetime is an object with one key, `date`, `time`, or `datetime`,
  whose value is in ISO 8601 format, such as `{"date": "1923-09-16"}`.
- A duration is an object with the fields of [`Duration`][kgeval.Duration].
- A point is an object with the fields of [`Point`][kgeval.Point].

Embeddings appear only as their dimensions, under `embedding_dimensions`.

## Graph layout

The layout decides which nodes are entities, lexical nodes, and community nodes, and
which properties hold names, text, and descriptions. kgeval detects it from the labels
in the graph:

| Role            | Detected labels                                                                                                      |
| --------------- | -------------------------------------------------------------------------------------------------------------------- |
| Lexical nodes   | `Chunk`, `Document`, `Episodic`, `__Chunk__`, `__Document__`, on nodes that hold text or are joined to nodes that do |
| Community nodes | `__Community__`, and `Community` in Graphiti graphs                                                                  |
| Entities        | `__Entity__`, `__KGBuilder__`, `__Node__`, and `Entity` in Graphiti graphs                                           |

Labels are checked in that order, so a chunk that also has `__KGBuilder__` stays a
lexical node. A lexical label counts only on nodes that hold text, such as chunks, or
that are joined to them, such as the documents of those chunks, so that a `Document`
node type in a graph without a lexical graph stays a node type. A graph is a Graphiti
graph when it has Graphiti's `Episodic` label. When the graph has none of the entity
labels, every node that is not a lexical or community node is an entity. When it has
some, nodes without them are ignored.

Entity names come from the first of `name` and `title` that holds text. `id` is tried
last when the graph has the `__Entity__` label, or when no entity has a name or title,
because LLM Graph Builder and LangChain store names in `id`. In a Graphiti graph,
relationship types come from the `name` property of its `RELATES_TO` relationships, and
relationships with an `expired_at` property, which Graphiti sets on facts that later
information invalidated, are expired facts, which the metrics leave out.

The report's summary shows the layout kgeval used.

### Custom layout

For a graph with other labels or property keys, write a layout file with only the
fields to change. The other fields keep their detected values.

```yaml title="layout.yaml"
lexical_labels: [Passage]
community_labels: [Topic]
base_labels: [Concept]
name_keys: [label]
text_keys: [body]
type_keys: [kind]
description_keys: [summary]
expired_keys: [retracted_at]
```

| Field              | Default                                 | Description                                                                                                                  |
| ------------------ | --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `lexical_labels`   | Detected                                | Labels of lexical nodes.                                                                                                     |
| `community_labels` | Detected                                | Labels of community nodes.                                                                                                   |
| `base_labels`      | Detected                                | Labels that mark a node as an entity without being its node type.                                                            |
| `name_keys`        | `name`, `title`, and `id` when detected | Properties tried in order for an entity's name.                                                                              |
| `text_keys`        | `text`, `content`                       | Properties tried in order for a chunk's text.                                                                                |
| `type_keys`        | None, or `name` for Graphiti            | Properties tried in order for a fact's relationship type, for builders that store every fact under one generic type.         |
| `description_keys` | `fact`, `description`, `summary`        | Properties tried in order for an entity's or fact's description.                                                             |
| `expired_keys`     | None, or `expired_at` for Graphiti      | Properties that mark a relationship between entities as no longer current, so that it is an expired fact rather than a fact. |

=== "CLI"

    ```console
    kgeval run --layout layout.yaml
    ```

=== "Python"

    ```python
    overrides = kgeval.Layout(lexical_labels=frozenset({"Passage"}), text_keys=("body",))
    layout = kgeval.Layout.detect(graph, overrides)
    report = kgeval.evaluate(graph, layout=layout)
    ```

    `Layout.detect` keeps the fields set on the overrides and detects the others,
    guided by the overrides' text and name keys, as the CLI does with a layout file. A
    [`Layout`][kgeval.Layout] passed to `evaluate` is used as it is.

## Logging

Pass `--verbose` to log progress to standard error. In Python, kgeval logs through the
standard `logging` module, under the `kgeval` logger.
