"""Retrieval metrics, and the confidence interval that decides what is claimed.

Everything is computed from one array: the **rank of the correct item** for
each query, 1-based. Every metric in the build plan is a function of that, so
computing it once and deriving the rest removes the class of bug where R@1 and
MRR disagree because two code paths ranked slightly differently.

`CLAUDE.md` section 4 governs the comparison, and the important part is that
retrieval eval is **deterministic given a checkpoint** — repeated identical
runs measure nothing. That differs from ListingLens' judged agent benchmark,
which has a measured ~37% run-to-run noise floor, and a reader who knows that
convention will otherwise assume repeated runs mean something here. The
variance that does exist is query-sampling and training-seed, so the test is a
**paired bootstrap over queries**: much tighter than two independent intervals,
and the correct question for "did we beat frozen CLIP".

**A delta whose CI crosses zero is not claimed.** That rule is enforced here
rather than left to the person writing the report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_RESAMPLES = 10_000
RECALL_KS = (1, 10, 50)


def ranks_from_scores(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """1-based rank of each query's correct item.

    Ties are broken **pessimistically** — a tied item counts as ranked above
    the target. A random-scoring model with many ties would otherwise look
    good, and the class-prior baseline produces ties by construction.
    """
    target_scores = scores[np.arange(scores.shape[0]), targets]
    better = (scores > target_scores[:, None]).sum(axis=1)
    tied = (scores == target_scores[:, None]).sum(axis=1) - 1
    return better + tied + 1


def recall_at_k(ranks: np.ndarray, k: int) -> np.ndarray:
    """Per query, so it can be bootstrapped. Mean it for the headline."""
    return (ranks <= k).astype(np.float64)


def reciprocal_rank(ranks: np.ndarray) -> np.ndarray:
    return 1.0 / ranks


def ndcg_at_k(ranks: np.ndarray, k: int = 10) -> np.ndarray:
    """With exactly one relevant item per query, IDCG is 1 and NDCG collapses
    to `1/log2(rank+1)` inside the cutoff, 0 outside."""
    inside = ranks <= k
    return np.where(inside, 1.0 / np.log2(ranks + 1.0), 0.0)


def per_query_metrics(ranks: np.ndarray) -> dict[str, np.ndarray]:
    metrics = {f"recall@{k}": recall_at_k(ranks, k) for k in RECALL_KS}
    metrics["ndcg@10"] = ndcg_at_k(ranks, 10)
    metrics["mrr"] = reciprocal_rank(ranks)
    return metrics


def summarise(ranks: np.ndarray) -> dict[str, float]:
    out = {name: float(values.mean()) for name, values in per_query_metrics(ranks).items()}
    out["median_rank"] = float(np.median(ranks))
    out["queries"] = int(ranks.shape[0])
    return out


@dataclass(frozen=True)
class Delta:
    """A model-minus-baseline difference, with the interval that qualifies it."""

    metric: str
    model: float
    baseline: float
    delta: float
    ci_low: float
    ci_high: float

    @property
    def claimed(self) -> bool:
        """False when the interval straddles zero. See CLAUDE.md section 4."""
        return not (self.ci_low <= 0.0 <= self.ci_high)

    def format_ci(self) -> str:
        marker = "" if self.claimed else " (crosses 0)"
        return f"{self.delta:+.4f} [{self.ci_low:+.4f}, {self.ci_high:+.4f}]{marker}"


def paired_bootstrap(
    model_values: np.ndarray,
    baseline_values: np.ndarray,
    *,
    metric: str = "",
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    alpha: float = 0.05,
) -> Delta:
    """Resample QUERIES, not scores, keeping model and baseline together.

    Pairing is the whole point: the same query is easy or hard for both
    systems, and resampling them independently would inflate the interval with
    variance that cancels.
    """
    if model_values.shape != baseline_values.shape:
        raise ValueError(f"{model_values.shape} against {baseline_values.shape}")
    n = model_values.shape[0]
    if n == 0:
        raise ValueError("no queries to bootstrap")

    differences = model_values - baseline_values
    rng = np.random.default_rng(seed)
    # One (resamples, n) index draw, then a single mean over axis 1 — a Python
    # loop over 10,000 resamples of 12,000 queries is minutes, this is
    # milliseconds.
    picks = rng.integers(0, n, size=(resamples, n))
    means = differences[picks].mean(axis=1)

    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return Delta(
        metric=metric,
        model=float(model_values.mean()),
        baseline=float(baseline_values.mean()),
        delta=float(differences.mean()),
        ci_low=float(low),
        ci_high=float(high),
    )
