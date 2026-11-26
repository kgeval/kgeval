# Custom metrics

A custom metric checks a quality that no built-in metric covers, such as a requirement
of your domain. Define it in one sentence as a criterion, which a model checks, or write
it in Python. Either way, it runs as a built-in metric does, and its result appears in
the report and in comparisons.

## Check a criterion

A criterion defines a metric in one sentence, such as "The fact gives the year of the
founding." A model decides whether each sampled entity or fact meets it. The score is
the share of the entities or facts the criterion applies to that meet it, and each one
that does not meet it becomes a finding with the model's reason.

Define criteria in a [suite](suites.md), each under its own name:

```yaml title="suite.yaml"
dated_people:
  criterion: The entity gives the day, month, and year of the person's birth.
  applies_to: entities
  node_types: [Person]
  threshold: 0.9
dated_foundings:
  criterion: The fact gives the year of the founding.
  applies_to: facts
  relationship_types: [FOUNDED]
```

=== "CLI"

    ```console
    kgeval run --model <model> --suite suite.yaml
    ```

=== "Python"

    ```python
    dated_foundings = kgeval.FactCriterion(
        name="dated_foundings",
        criterion="The fact gives the year of the founding.",
        relationship_types=frozenset({"FOUNDED"}),
    )
    report = kgeval.evaluate(graph, [dated_foundings], model=kgeval.Model("<model>"))
    ```

    [`EntityCriterion`][kgeval.EntityCriterion] takes `node_types` in the same way, and
    [`load_suite`][kgeval.load_suite] reads the criteria of a suite.

| Setting              | Description                                                                                |
| -------------------- | ------------------------------------------------------------------------------------------ |
| `criterion`          | The sentence that each entity or fact must meet.                                           |
| `applies_to`         | `entities` for an entity criterion, or `facts` for a fact criterion.                       |
| `node_types`         | The node types of the entities that an entity criterion checks. Every entity when omitted. |
| `relationship_types` | The relationship types of the facts that a fact criterion checks. Every fact when omitted. |

A criterion's name is made of lowercase letters, digits, and underscores, starting with
a letter. A criterion also takes the [settings](suites.md#settings) of the sampled model
metrics, such as `threshold` and `sample_size`.

The model sees the criterion and the entity or fact with its name, types, and
properties, and up to three chunks of text it was extracted from. For a fact, these are
the chunks that ground both of its entities. The model answers yes, no, or not
applicable, and the score leaves out the entities and facts the criterion does not apply
to. `kgeval compare` compares a criterion's scores only between runs with the same
sentence, `applies_to`, and types.

### Write a criterion

- State a condition on one entity or fact, such as "The fact gives the year of the
  founding", rather than on the whole graph, such as "The graph has dates".
- Make the condition checkable from what the model sees: names, types, properties, and
  chunks. Name properties as the graph does, such as `born`.
- Set `node_types` or `relationship_types` when you know the types the criterion is
  about. Otherwise the model decides which entities or facts it applies to, and those it
  does not apply to still take up the sample, so a criterion about only a few of them
  gets a wide confidence interval.

## Write a metric in Python

Subclass [`Metric`][kgeval.Metric] and implement `evaluate`, which checks each unit of
the metric's population, such as each person, and returns the counts with a finding for
each unit that fails:

```python
import kgeval


class BirthplaceCompleteness(kgeval.Metric):
    """Check that every person has a birthplace."""

    name = "birthplace_completeness"
    category = kgeval.MetricCategory.INTRINSIC
    unit = "people"

    def evaluate(self, graph: kgeval.GraphView, inputs: kgeval.Inputs) -> kgeval.MetricResult:
        people = [
            entity for entity in graph.entities if "Person" in graph.layout.node_types(entity)
        ]
        born = {
            fact.source_id for fact in graph.facts if graph.relationship_type(fact) == "BORN_IN"
        }
        findings = [
            kgeval.Finding.about(
                person,
                f"{graph.describe(person)} has no birthplace",
                subject=graph.describe(person),
            )
            for person in people
            if person.id not in born
        ]
        return self.result(inputs, len(people), len(findings), findings)


report = kgeval.evaluate(graph, [BirthplaceCompleteness(threshold=0.9)])
```

- `name` identifies the metric in reports, thresholds, and comparisons.
- `category` groups the metric in reports. A metric of the graph alone is intrinsic.
- `unit` is the plural noun for what the metric counts.
- [`GraphView`][kgeval.GraphView] gives the graph's entities, facts, and chunks, as the
  [layout](configuration.md#graph-layout) reads them.
- `result` scores the share of units that pass, and records the metric as skipped when
  it evaluated none.

Give each finding a subject that names the unit at fault by content, as `graph.describe`
does. `compare` matches findings by subject, so a defect keeps its subject when a rebuild
assigns new element IDs.

## Add parameters and inputs

Declare parameters as fields, as `threshold` is. When a parameter changes the score,
override `signature` to include it, so that `compare` compares only scores measured
alike. A metric that reads an input, such as a graph schema, lists it in `needs` and
reads it with `self.require(inputs.schema, "schema")`.

## Ask a model

To decide units with a model in Python, subclass [`SampledMetric`][kgeval.SampledMetric]
instead. Its `sample`, `ask_all`, and `estimate` methods pick a reproducible sample, ask
the model about each sampled unit, and estimate the score, with a confidence interval
when the sample leaves units out.
