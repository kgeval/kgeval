"""Define the base classes of the model metrics."""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from statistics import NormalDist
from typing import ClassVar, TypeVar

from pydantic import BaseModel, Field

from kgeval.errors import VerdictError
from kgeval.metrics.base import Finding, InputName, Inputs, Metric, MetricCategory, MetricResult
from kgeval.model import Model

__all__ = ["DEFAULT_SAMPLE_SIZE", "ModelMetric", "SampledMetric"]

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_SIZE = 100
"""The default for the most units each sampled metric asks the model about."""

_ATTEMPTS = 2

_Z_95 = NormalDist().inv_cdf(0.975)

_UnitT = TypeVar("_UnitT")
_OutputT = TypeVar("_OutputT", bound=BaseModel)


class ModelMetric(Metric):
    """Decide, with a model, a quality that rules cannot.

    LLM verdicts are not fully deterministic, so the signature names the model.

    Attributes:
        concurrency: How many requests to the model run at once.
    """

    category: ClassVar[MetricCategory] = MetricCategory.MODEL
    needs: ClassVar[tuple[InputName, ...]] = ("model",)

    concurrency: int = Field(default=4, ge=1)

    def signature(self, inputs: Inputs) -> str:
        """Return the metric name with the model's name.

        Args:
            inputs: The inputs the metric is evaluated with.

        Returns:
            The signature.
        """
        return f"{self.name}|model={self.require(inputs.model, 'model').name}"

    def ask_all(
        self, inputs: Inputs, prompts: Sequence[str], output_type: type[_OutputT]
    ) -> list[_OutputT | None]:
        """Ask the model every prompt, ``concurrency`` at a time.

        A prompt the model gives no usable verdict for, by raising ``VerdictError``, is
        asked once more, since LLM verdicts vary, and is otherwise left undecided.

        Args:
            inputs: The inputs, whose model is asked.
            prompts: The prompts.
            output_type: The Pydantic model class each verdict must validate against.

        Returns:
            The verdicts, in the order of the prompts, with ``None`` for each prompt
            left undecided.

        Raises:
            InputError: If the model was not given.
            ModelError: If the model fails otherwise on any prompt, or gives no usable
                verdict for any; queued prompts are cancelled.
        """
        model = self.require(inputs.model, "model")
        pool = ThreadPoolExecutor(max_workers=self.concurrency)
        try:
            answers = list(pool.map(lambda prompt: _ask(model, prompt, output_type), prompts))
        finally:
            # Cancel queued calls after a failure rather than pay for verdicts nobody reads.
            pool.shutdown(cancel_futures=True)
        errors = [answer for answer in answers if isinstance(answer, VerdictError)]
        if answers and len(errors) == len(answers):
            cause = errors[0].__cause__ or errors[0]
            raise VerdictError(
                f"{model.name} gave no usable verdict for any prompt, such as: {cause}"
            ) from errors[0]
        return [None if isinstance(answer, VerdictError) else answer for answer in answers]


class SampledMetric(ModelMetric):
    """Estimate, with a model, a quality of a population from a reproducible sample.

    Subclasses choose the population, write one prompt per sampled unit, and decide
    which verdicts fail; this class samples, selects chunk texts, and builds the result.
    The sample is the ``sample_size`` units with the lowest SHA-256 of
    ``"{seed}:{key}"``, where a unit's key describes its content, so the sample is the
    same on every run and platform, and only extended when ``sample_size`` grows. Since
    keys hold no element IDs, a rebuilt graph with the same content gets the same
    sample, and graphs built from one corpus in different ways mostly sample the units
    they have in common, which pairs their scores and lets cached verdicts serve both.
    When the decided units are fewer than the population, the result carries a 95%
    Wilson score interval. Sampled units the model gives no usable verdict for are left
    out and counted as undecided.

    Attributes:
        sample_size: The most units to ask the model about.
        seed: Selects which units are sampled.
    """

    sample_size: int = Field(default=DEFAULT_SAMPLE_SIZE, ge=1)
    seed: int = 0

    def signature(self, inputs: Inputs) -> str:
        """Return the metric name with the model and the sampling parameters.

        Args:
            inputs: The inputs the metric is evaluated with.

        Returns:
            The signature.
        """
        return f"{super().signature(inputs)}|sample_size={self.sample_size}|seed={self.seed}"

    def sample(self, population: Sequence[_UnitT], key: Callable[[_UnitT], str]) -> list[_UnitT]:
        """Return the reproducible sample of a population.

        Args:
            population: Every unit the model could be asked about.
            key: Returns the content that ranks a unit, such as the text the model
                reads; units with equal keys keep their population order.

        Returns:
            Up to ``sample_size`` units, in rank order.
        """
        ranked = sorted(population, key=lambda unit: self._rank(key(unit)))
        sample = ranked[: self.sample_size]
        logger.info("Sampling %d of %d %s", len(sample), len(population), self.unit)
        return sample

    def estimate(
        self,
        inputs: Inputs,
        population: int,
        sampled: int,
        findings: Sequence[Finding],
        undecided: int = 0,
    ) -> MetricResult:
        """Build the result of asking about a sample of a population, one unit per item.

        Args:
            inputs: The inputs the metric was evaluated with.
            population: How many units the model could have been asked about.
            sampled: How many units were sampled.
            findings: One finding per decided unit that failed.
            undecided: How many sampled units the model gave no usable verdict for.

        Returns:
            The result, with a confidence interval when fewer units were decided than
            the population holds.
        """
        decided = sampled - undecided
        interval = None
        if 0 < decided < population:
            interval = _wilson_interval((decided - len(findings)) / decided, decided)
        return self.result(
            inputs,
            decided,
            len(findings),
            findings,
            confidence_interval=interval,
            undecided=undecided,
        )

    def estimate_clusters(
        self,
        inputs: Inputs,
        population: int,
        clusters: Sequence[tuple[int, int]],
        findings: Sequence[Finding],
        undecided: int = 0,
    ) -> MetricResult:
        """Build the result of asking about a sample of clusters of units, such as chunks.

        The score is the share of units that passed over all decided clusters. Units of
        one cluster tend to pass or fail together, so the interval is Wilson's at Kish's
        effective sample size, which the variance of the clusters' shares sets between
        the number of clusters and the number of units.

        Args:
            inputs: The inputs the metric was evaluated with.
            population: How many clusters the model could have been asked about.
            clusters: For each decided cluster, how many units it holds and how many of
                them passed.
            findings: One finding per decided cluster with a unit that failed.
            undecided: How many sampled clusters the model gave no usable verdict for.

        Returns:
            The result, with a confidence interval when fewer clusters were decided than
            the population holds.
        """
        units = sum(size for size, _ in clusters)
        passed = sum(passed for _, passed in clusters)
        interval = None
        if units and len(clusters) < population:
            share = passed / units
            interval = _wilson_interval(share, _effective_size(clusters, share))
        return self.result(
            inputs,
            units,
            units - passed,
            findings,
            confidence_interval=interval,
            undecided=undecided,
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


def _ask(model: Model, prompt: str, output_type: type[_OutputT]) -> _OutputT | VerdictError:
    """Ask the model a prompt, once more if the first verdict is unusable, else return the error."""
    errors: list[VerdictError] = []
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            return model.ask(prompt, output_type)
        except VerdictError as error:
            logger.warning("No usable verdict on attempt %d of %d: %s", attempt, _ATTEMPTS, error)
            errors.append(error)
    return errors[-1]
