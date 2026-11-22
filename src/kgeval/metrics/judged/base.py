"""Define the base classes of the judged metrics."""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from statistics import NormalDist
from typing import ClassVar, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from kgeval.errors import VerdictError
from kgeval.judges.base import Judge
from kgeval.metrics.base import Finding, Metric, MetricCategory, MetricResult

__all__ = ["JudgedMetric", "SampledMetric"]

logger = logging.getLogger(__name__)

_Z_95 = NormalDist().inv_cdf(0.975)

_UnitT = TypeVar("_UnitT")
_OutputT = TypeVar("_OutputT", bound=BaseModel)


class JudgedMetric(Metric):
    """Decide, with an LLM judge, a quality that rules cannot.

    LLM verdicts are not fully deterministic, so the signature names the judge.

    Attributes:
        judge: The LLM judge.
        concurrency: How many judge calls run at once.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    category: ClassVar[MetricCategory] = MetricCategory.JUDGED

    judge: Judge
    concurrency: int = Field(default=4, ge=1)

    def signature(self) -> str:
        """Return the metric name with the judge's name.

        Returns:
            The signature.
        """
        return f"{self.name}|judge={self.judge.name}"

    def ask_all(self, prompts: Sequence[str], output_type: type[_OutputT]) -> list[_OutputT | None]:
        """Ask the judge every prompt, ``concurrency`` at a time.

        A prompt the judge gives no usable verdict for, by raising ``VerdictError``, is
        asked once more, since LLM answers vary, and is otherwise left unjudged.

        Args:
            prompts: The prompts.
            output_type: The Pydantic model class each answer must validate against.

        Returns:
            The answers, in the order of the prompts, with ``None`` for each prompt left
            unjudged.

        Raises:
            JudgeError: If the judge fails otherwise on any prompt, or gives no usable
                verdict for any; queued prompts are cancelled.
        """
        pool = ThreadPoolExecutor(max_workers=self.concurrency)
        try:
            answers = list(pool.map(lambda prompt: self._ask(prompt, output_type), prompts))
        finally:
            # Cancel queued calls after a failure rather than pay for answers nobody reads.
            pool.shutdown(cancel_futures=True)
        if prompts and all(answer is None for answer in answers):
            raise VerdictError(f"{self.judge.name} gave no usable verdict for any prompt")
        return answers

    def _ask(self, prompt: str, output_type: type[_OutputT]) -> _OutputT | None:
        for attempt in (1, 2):
            try:
                return self.judge.ask(prompt, output_type)
            except VerdictError as error:
                logger.warning("No usable verdict on attempt %d of 2: %s", attempt, error)
        return None


class SampledMetric(JudgedMetric):
    """Estimate, with an LLM judge, a quality of a population from a reproducible sample.

    Subclasses choose the population, write one prompt per sampled unit, and decide
    which verdicts fail; this class samples, selects chunk texts, and builds the result.
    The sample is the ``sample_size`` units with the lowest SHA-256 of
    ``"{seed}:{key}"``, where a unit's key describes its content: the same on every run
    and platform, and only extended when ``sample_size`` grows. Since keys hold no
    element IDs, a rebuilt graph with the same content gets the same sample, and graphs
    built from one corpus in different ways mostly sample the units they have in common,
    which pairs their scores and lets cached verdicts serve both. When the judged units
    are fewer than the population, the result carries a 95% Wilson score interval.
    Sampled units the judge gives no usable verdict for are left out and counted as
    unjudged.

    Attributes:
        sample_size: The most units to judge.
        seed: Selects which units are sampled.
    """

    sample_size: int = Field(default=100, ge=1)
    seed: int = 0

    def signature(self) -> str:
        """Return the metric name with the judge and the sampling parameters.

        Returns:
            The signature.
        """
        return f"{super().signature()}|sample_size={self.sample_size}|seed={self.seed}"

    def sample(self, population: Sequence[_UnitT], key: Callable[[_UnitT], str]) -> list[_UnitT]:
        """Return the reproducible sample of a population.

        Args:
            population: Every unit that could be judged.
            key: Returns the content that ranks a unit, such as the text it is judged
                on; units with equal keys keep their population order.

        Returns:
            Up to ``sample_size`` units, in rank order.
        """
        ranked = sorted(population, key=lambda unit: self._rank(key(unit)))
        sample = ranked[: self.sample_size]
        logger.info(
            "Judging %d of %d %s with %s", len(sample), len(population), self.unit, self.judge.name
        )
        return sample

    def estimate(
        self, population: int, sampled: int, findings: Sequence[Finding], unjudged: int = 0
    ) -> MetricResult:
        """Build the result of judging a sample of a population, one unit per sampled item.

        Args:
            population: How many units could have been judged.
            sampled: How many units were sampled.
            findings: One finding per judged unit that failed.
            unjudged: How many sampled units the judge gave no usable verdict for.

        Returns:
            The result, with a confidence interval when fewer units were judged than
            the population holds.
        """
        judged = sampled - unjudged
        interval = None
        if 0 < judged < population:
            interval = _wilson_interval((judged - len(findings)) / judged, judged)
        return self.result(
            judged, len(findings), findings, confidence_interval=interval, unjudged=unjudged
        )

    def estimate_clusters(
        self,
        population: int,
        clusters: Sequence[tuple[int, int]],
        findings: Sequence[Finding],
        unjudged: int = 0,
    ) -> MetricResult:
        """Build the result of judging a sample of clusters of units, such as chunks of facts.

        The score is the share of units that passed over all judged clusters. Units of
        one cluster tend to pass or fail together, so the interval is Wilson's at Kish's
        effective sample size, which the variance of the clusters' shares sets between
        the number of clusters and the number of units.

        Args:
            population: How many clusters could have been judged.
            clusters: For each judged cluster, how many units it holds and how many of
                them passed.
            findings: One finding per judged cluster with a unit that failed.
            unjudged: How many sampled clusters the judge gave no usable verdict for.

        Returns:
            The result, with a confidence interval when fewer clusters were judged than
            the population holds.
        """
        units = sum(size for size, _ in clusters)
        passed = sum(passed for _, passed in clusters)
        interval = None
        if units and len(clusters) < population:
            share = passed / units
            interval = _wilson_interval(share, _effective_size(clusters, share))
        return self.result(
            units, units - passed, findings, confidence_interval=interval, unjudged=unjudged
        )

    def _rank(self, key: str) -> str:
        return hashlib.sha256(f"{self.seed}:{key}".encode()).hexdigest()


def _effective_size(clusters: Sequence[tuple[int, int]], share: float) -> float:
    """Return Kish's effective sample size of a cluster sample, between its clusters and units."""
    count = len(clusters)
    units = sum(size for size, _ in clusters)
    simple_variance = share * (1 - share) / units
    residuals = sum((passed - share * size) ** 2 for size, passed in clusters)
    if count < 2 or simple_variance == 0:
        # Fall back to one unit per cluster, the most conservative size, without a variance.
        return count
    ratio_variance = residuals / (count - 1) / count / (units / count) ** 2
    if ratio_variance == 0:
        return units
    return min(max(units * simple_variance / ratio_variance, count), units)


def _wilson_interval(proportion: float, trials: float) -> tuple[float, float]:
    """Return the 95% Wilson score interval for a binomial proportion."""
    spread = _Z_95**2 / trials
    center = (proportion + spread / 2) / (1 + spread)
    margin = _Z_95 * math.sqrt(proportion * (1 - proportion) / trials + spread / (4 * trials))
    margin /= 1 + spread
    return max(0.0, center - margin), min(1.0, center + margin)
