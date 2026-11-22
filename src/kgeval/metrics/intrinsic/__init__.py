"""Provide the intrinsic metrics."""

from kgeval.metrics.intrinsic.entities import EntityConnectivity, EntityNaming, EntityUniqueness
from kgeval.metrics.intrinsic.facts import FactUniqueness
from kgeval.metrics.intrinsic.properties import EmbeddingConsistency, PropertyTypeConsistency
from kgeval.metrics.intrinsic.provenance import (
    EntityProvenance,
    ExtractionCoverage,
    FactProvenance,
)
from kgeval.metrics.intrinsic.vocabulary import VocabularyUniqueness

__all__ = [
    "EmbeddingConsistency",
    "EntityConnectivity",
    "EntityNaming",
    "EntityProvenance",
    "EntityUniqueness",
    "ExtractionCoverage",
    "FactProvenance",
    "FactUniqueness",
    "PropertyTypeConsistency",
    "VocabularyUniqueness",
]
