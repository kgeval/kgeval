# Python API

Everything on this page is importable from the top-level `kgeval` package, except the
schema types, which are in `kgeval.schema`, and `Reference`, which is in
`kgeval.reference`.

```python
import kgeval
from neo4j import GraphDatabase

with GraphDatabase.driver("<uri>", auth=("<username>", "<password>")) as driver:
    graph = kgeval.Neo4jSource(driver).load()

metrics = kgeval.default_metrics(
    schema=kgeval.load_schema("schema.yaml"),
    ground_truth=kgeval.load_ground_truth("ground_truth.yaml"),
    judge=kgeval.create_judge("openai:gpt-5-mini"),
)
report = kgeval.evaluate(graph, metrics)

report["entity_uniqueness"].findings  # each with .message, .subject, and its element IDs
report.passed  # False when a metric misses its threshold
```

## Evaluation

::: kgeval.evaluate

::: kgeval.default_metrics

## Reports

::: kgeval.Report

::: kgeval.MetricResult

::: kgeval.Finding

::: kgeval.GraphSummary

::: kgeval.compare

::: kgeval.Comparison

::: kgeval.MetricChange

::: kgeval.Trend

## Graphs

::: kgeval.Neo4jSource

::: kgeval.Graph

::: kgeval.Node

::: kgeval.Relationship

::: kgeval.GraphView

::: kgeval.Layout

::: kgeval.Role

::: kgeval.PropertyValue

::: kgeval.Point

::: kgeval.Duration

::: kgeval.value_type

## Graph schemas

::: kgeval.load_schema

::: kgeval.infer_schema

::: kgeval.GraphSchema

::: kgeval.schema.NodeType

::: kgeval.schema.RelationshipType

::: kgeval.schema.PropertyType

::: kgeval.schema.Pattern

::: kgeval.schema.Constraint

## Ground truths and questions

::: kgeval.load_ground_truth

::: kgeval.GroundTruth

::: kgeval.load_questions

::: kgeval.QuestionSet

::: kgeval.Question

::: kgeval.reference.Reference

## Retrievals

::: kgeval.retrieve

::: kgeval.load_retrievals

::: kgeval.Retriever

::: kgeval.Retrievals

## Judges

::: kgeval.create_judge

::: kgeval.Judge

::: kgeval.OpenAIJudge

::: kgeval.CachedJudge

## Metrics

::: kgeval.Metric

::: kgeval.MetricCategory

::: kgeval.JudgedMetric

::: kgeval.SampledMetric

::: kgeval.SchemaMetric

::: kgeval.GroundTruthMetric

::: kgeval.RetrievalMetric

::: kgeval.BUILTIN_METRICS

### Intrinsic metrics

::: kgeval.EntityUniqueness
    options: {heading_level: 4, members: false}

::: kgeval.EntityConnectivity
    options: {heading_level: 4, members: false}

::: kgeval.EntityNaming
    options: {heading_level: 4, members: false}

::: kgeval.EntityProvenance
    options: {heading_level: 4, members: false}

::: kgeval.FactUniqueness
    options: {heading_level: 4, members: false}

::: kgeval.FactProvenance
    options: {heading_level: 4, members: false}

::: kgeval.VocabularyUniqueness
    options: {heading_level: 4, members: false}

::: kgeval.PropertyTypeConsistency
    options: {heading_level: 4, members: false}

::: kgeval.EmbeddingConsistency
    options: {heading_level: 4, members: false}

::: kgeval.ExtractionCoverage
    options: {heading_level: 4, members: false}

### Schema metrics

::: kgeval.EntityConformance
    options: {heading_level: 4, members: false}

::: kgeval.FactConformance
    options: {heading_level: 4, members: false}

::: kgeval.SchemaCompleteness
    options: {heading_level: 4, members: false}

::: kgeval.PropertyCompleteness
    options: {heading_level: 4, members: false}

### Reference metrics

::: kgeval.FactPrecision
    options: {heading_level: 4, members: false}

::: kgeval.FactRecall
    options: {heading_level: 4, members: false}

::: kgeval.FactF1
    options: {heading_level: 4, members: false}

::: kgeval.EntityPrecision
    options: {heading_level: 4, members: false}

::: kgeval.EntityRecall
    options: {heading_level: 4, members: false}

::: kgeval.EntityF1
    options: {heading_level: 4, members: false}

::: kgeval.Competency
    options: {heading_level: 4, members: false}

### Judged metrics

::: kgeval.EntityResolution
    options: {heading_level: 4, members: false}

::: kgeval.EntityDisambiguation
    options: {heading_level: 4, members: false}

::: kgeval.NodeTypeAccuracy
    options: {heading_level: 4, members: false}

::: kgeval.FactAccuracy
    options: {heading_level: 4, members: false}

::: kgeval.VocabularyResolution
    options: {heading_level: 4, members: false}

::: kgeval.ExtractionCompleteness
    options: {heading_level: 4, members: false}

### Retrieval metrics

::: kgeval.RetrievalRecall
    options: {heading_level: 4, members: false}

::: kgeval.RetrievalPrecision
    options: {heading_level: 4, members: false}

## Errors

::: kgeval.KgevalError

::: kgeval.SourceError

::: kgeval.InputError

::: kgeval.JudgeError

::: kgeval.VerdictError
