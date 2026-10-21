"""Confidence intervals for small evaluation sets.

With a few dozen questions, a single percentage (e.g. "90% hit rate") hides
how uncertain it is. These helpers report a 95% interval next to every
metric so the reader can judge whether two methods really differ.
"""

import math
import random
from collections.abc import Sequence

Z_95 = 1.959963984540054


def wilson_interval(successes: int, total: int, z: float = Z_95) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (e.g. hit rate, % grounded).

    Preferred over the normal approximation because it stays inside [0, 1]
    and behaves well with small samples and proportions near 0% or 100%.
    """
    if total <= 0:
        raise ValueError("total must be greater than zero")
    if not 0 <= successes <= total:
        raise ValueError("successes must be between 0 and total")

    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z**2 / (4 * total**2)
    ) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def bootstrap_mean_interval(
    values: Sequence[float],
    resamples: int = 5_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of ``values`` (e.g. MRR).

    A fixed seed keeps the reported interval reproducible between runs.
    """
    if not values:
        raise ValueError("values must not be empty")

    rng = random.Random(seed)
    size = len(values)
    means = sorted(
        sum(rng.choice(values) for _ in range(size)) / size for _ in range(resamples)
    )
    lower_index = int(((1 - confidence) / 2) * resamples)
    upper_index = int((1 - (1 - confidence) / 2) * resamples) - 1
    return means[lower_index], means[upper_index]


def format_interval(interval: tuple[float, float], as_percent: bool = True) -> str:
    low, high = interval
    if as_percent:
        return f"[{low:.0%}, {high:.0%}]"
    return f"[{low:.2f}, {high:.2f}]"
