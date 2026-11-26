"""Provide the model metrics."""

from kgeval.metrics.model._prompts import MAX_CHUNKS
from kgeval.metrics.model.base import DEFAULT_SAMPLE_SIZE, ModelMetric, SampledMetric
from kgeval.metrics.model.criteria import (
    CriterionMetric,
    CriterionVerdict,
    EntityCriterion,
    FactCriterion,
)
from kgeval.metrics.model.entities import (
    DuplicateVerdict,
    EntityDisambiguation,
    EntityResolution,
    IdentityVerdict,
    NodeTypeAccuracy,
    TypeVerdict,
)
from kgeval.metrics.model.extraction import (
    CompletenessVerdict,
    ExtractionCompleteness,
    StatedFact,
)
from kgeval.metrics.model.facts import FactAccuracy, SupportVerdict
from kgeval.metrics.model.vocabulary import SynonymGroup, SynonymVerdict, VocabularyResolution

__all__ = [
    "DEFAULT_SAMPLE_SIZE",
    "MAX_CHUNKS",
    "CompletenessVerdict",
    "CriterionMetric",
    "CriterionVerdict",
    "DuplicateVerdict",
    "EntityCriterion",
    "EntityDisambiguation",
    "EntityResolution",
    "ExtractionCompleteness",
    "FactAccuracy",
    "FactCriterion",
    "IdentityVerdict",
    "ModelMetric",
    "NodeTypeAccuracy",
    "SampledMetric",
    "StatedFact",
    "SupportVerdict",
    "SynonymGroup",
    "SynonymVerdict",
    "TypeVerdict",
    "VocabularyResolution",
]
