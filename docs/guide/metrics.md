# Metrics

kgeval has 29 metrics in five categories. The intrinsic metrics run by default, and
each input you add enables another category. Run `kgeval metrics` to list them all.
Each metric name below links to its full definition in the
[Python API reference](../reference/api.md).

## How kgeval reads a graph

LLM pipelines store more in Neo4j than the knowledge they extract. kgeval gives each
node a role based on its labels:

- **Entities** are the things the graph is about, such as people and organizations.
  An entity's labels, minus base labels such as `__Entity__`, are its node types.
- **Lexical nodes** are the documents and chunks that entities were extracted from.
  **Chunks** are the lexical nodes that entities are grounded in.
- **Community nodes** are summaries that some builders add. Only
  `embedding_consistency` evaluates them.

Relationships between two entities are **facts**, unless the layout marks them as
expired, as Graphiti does for facts that later information invalidated. Relationships
between an entity and a lexical node are **provenance relationships**: they ground the
entity in a chunk.

kgeval detects roles from the labels that well-known builders use.
[Configuration](configuration.md#graph-layout) explains how to change them.

## Scores

A metric checks each unit of its population, such as each entity or each fact, and
scores the share that pass:

```text
score = (evaluated - failed) / evaluated
```

The retrieval metrics are the exception: they score the mean of each question's share,
so that every question counts equally. A score of 1.0 is perfect. A metric with nothing to evaluate is skipped, with the
reason, instead of scored. When a metric has a threshold, it passes if its score is at
or above the threshold, and fails if it scores lower or is skipped.

## Intrinsic metrics

These run by default and need only the graph.

| Metric                                                        | Unit                 | Passes when                                                                                                  |
| ------------------------------------------------------------- | -------------------- | ------------------------------------------------------------------------------------------------------------ |
| [`entity_uniqueness`][kgeval.EntityUniqueness]                | entities with a name | no other entity has the same normalized name                                                                 |
| [`entity_connectivity`][kgeval.EntityConnectivity]            | entities             | it is in at least one fact with another entity                                                               |
| [`entity_naming`][kgeval.EntityNaming]                        | entities             | it has a name that is neither a pronoun, such as "He", nor a bare value, such as "1923"                      |
| [`entity_provenance`][kgeval.EntityProvenance]                | entities             | it is grounded in at least one chunk                                                                         |
| [`fact_uniqueness`][kgeval.FactUniqueness]                    | facts                | no other fact has the same source, relationship type, and target, whatever its properties                    |
| [`fact_provenance`][kgeval.FactProvenance]                    | facts                | its source and target are grounded in a common chunk                                                         |
| [`vocabulary_uniqueness`][kgeval.VocabularyUniqueness]        | vocabulary terms     | it is not a spelling variant of a more common term, as `bornIn` is of `BORN_IN`                              |
| [`property_type_consistency`][kgeval.PropertyTypeConsistency] | property values      | it has the most common type among values of the same property on the same node types or relationship type    |
| [`embedding_consistency`][kgeval.EmbeddingConsistency]        | expected embeddings  | the element holds each embedding that at least half of its group holds, with the dimension most of them have |
| [`extraction_coverage`][kgeval.ExtractionCoverage]            | chunks               | at least one entity is grounded in it                                                                        |

## Schema metrics

These need a [graph schema](inputs.md#graph-schema), passed with `--schema`.

| Metric                                                 | Unit                        | Passes when                                                                         |
| ------------------------------------------------------ | --------------------------- | ----------------------------------------------------------------------------------- |
| [`entity_conformance`][kgeval.EntityConformance]       | entities                    | its node types, properties, and constraints conform to the schema                   |
| [`fact_conformance`][kgeval.FactConformance]           | facts                       | its type, pattern, properties, constraints, and `max_targets` conform to the schema |
| [`schema_completeness`][kgeval.SchemaCompleteness]     | declared types and patterns | the graph uses it                                                                   |
| [`property_completeness`][kgeval.PropertyCompleteness] | declared properties         | the entity or fact holds it                                                         |

## Reference metrics

These need a [ground truth](inputs.md#ground-truth), passed with `--ground`, or
[test questions](inputs.md#test-questions), passed with `--questions`.

| Metric                                       | Unit                  | Passes when                                                    |
| -------------------------------------------- | --------------------- | -------------------------------------------------------------- |
| [`fact_precision`][kgeval.FactPrecision]     | distinct facts        | the ground truth has it                                        |
| [`fact_recall`][kgeval.FactRecall]           | ground truth facts    | the graph has it                                               |
| [`entity_precision`][kgeval.EntityPrecision] | distinct entities     | the ground truth has it                                        |
| [`entity_recall`][kgeval.EntityRecall]       | ground truth entities | the graph has it                                               |
| [`competency`][kgeval.Competency]            | test questions        | the graph has every supporting fact and entity of the question |

[`fact_f1`][kgeval.FactF1] and [`entity_f1`][kgeval.EntityF1] are the harmonic means of
the matching precision and recall. The ground truth entities are the sources and targets of the
ground truth facts.

## Judged metrics

These need an [LLM judge](judges.md), passed with `--judge`. All but
`vocabulary_resolution` judge a reproducible sample, so their scores come with a 95%
confidence interval when the judged units are fewer than the population.

| Metric                                                     | Unit                                                     | Passes when                                                                                                                                      |
| ---------------------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| [`entity_resolution`][kgeval.EntityResolution]             | entities with a name                                     | no other entity is the same thing under another name, such as "IBM" and "International Business Machines"                                        |
| [`entity_disambiguation`][kgeval.EntityDisambiguation]     | entities with a name in two or more chunks with text     | it stands for one thing in all its chunks, not several things that share a name                                                                  |
| [`node_type_accuracy`][kgeval.NodeTypeAccuracy]            | entities with a name, a node type, and a chunk with text | its chunks support its node types                                                                                                                |
| [`fact_accuracy`][kgeval.FactAccuracy]                     | facts grounded in a chunk with text                      | the chunks its entities share support it, in its direction                                                                                       |
| [`vocabulary_resolution`][kgeval.VocabularyResolution]     | distinct vocabulary terms                                | the graph uses no more common term for the same thing, as it may use `EMPLOYED_BY` for what `WORKS_AT` says; spelling variants count as one term |
| [`extraction_completeness`][kgeval.ExtractionCompleteness] | facts stated in chunks                                   | the graph has the fact that the chunk states                                                                                                     |

## Retrieval metrics

These need [test questions](inputs.md#test-questions) and
[retrievals](inputs.md#retrievals), passed with `--questions` and `--retrievals`.

| Metric                                             | Unit                                       | Passes when                    |
| -------------------------------------------------- | ------------------------------------------ | ------------------------------ |
| [`retrieval_recall`][kgeval.RetrievalRecall]       | supporting facts and entities in the graph | the retriever returned it      |
| [`retrieval_precision`][kgeval.RetrievalPrecision] | retrieved entities, facts, and chunks      | the question's answer needs it |

A supporting fact counts as retrieved when its relationship was retrieved, or a chunk
that grounds it, but not when only its two entities were. So retrievers that return
chunks are evaluated as well as those that return graph elements. With `--top-k`, only
the top-ranked element IDs of each retrieval count, as in recall and precision at k.

## Graph gaps and retrieval misses

When an answer is wrong, the fault lies either with the graph, which lacks a fact, or
with the retriever, which missed one. Test questions list the facts each answer
needs, so kgeval can tell which:

| `competency` | `retrieval_recall`             | Fix                                            |
| ------------ | ------------------------------ | ---------------------------------------------- |
| fails        | doesn't count the missing fact | The graph: extraction missed the fact.         |
| passes       | fails                          | The retriever: it missed a fact the graph has. |

`retrieval_recall` evaluates only the supporting facts that the graph has, so the
retriever is never blamed for a gap in the graph.

## References

The metric names follow the established terms of their fields.

| Concept                               | Source                                                                                                                                         |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Uniqueness and consistency            | DAMA UK, *The Six Primary Dimensions for Data Quality Assessment*, 2013                                                                        |
| Conformance                           | W3C, *Shapes Constraint Language (SHACL)*, 2017; Mihindukulasooriya et al., *Text2KGBench*, ISWC 2023                                          |
| Schema and property completeness      | Zaveri et al., *Quality Assessment for Linked Data: A Survey*, Semantic Web Journal, 2016                                                      |
| Terms judged by their usage           | Galárraga et al., *Canonicalizing Open Knowledge Bases*, CIKM 2014                                                                             |
| Windows of terms sorted by usage      | Hernández and Stolfo, *The Merge/Purge Problem for Large Databases*, SIGMOD 1995                                                               |
| Accuracy from a judged sample         | Gao et al., *Efficient Knowledge Graph Accuracy Evaluation*, VLDB 2019                                                                         |
| Change between sampled scores         | Newcombe, *Interval Estimation for the Difference Between Independent Proportions: Comparison of Eleven Methods*, Statistics in Medicine, 1998 |
| Provenance                            | W3C, *PROV-O: The PROV Ontology*, 2013                                                                                                         |
| Test questions (competency questions) | Grüninger and Fox, *Methodology for the Design and Evaluation of Ontologies*, 1995                                                             |
| Supporting facts                      | Yang et al., *HotpotQA*, EMNLP 2018                                                                                                            |
