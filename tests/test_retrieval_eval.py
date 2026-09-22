"""The retrieval eval: the index contract, the metrics, and the baselines.

Deliberately torch-free. `vislens.index.*` is numpy only, so these run in the
main CI job rather than behind the `train` extra — the rules they encode
(`CLAUDE.md` sections 3, 4 and 6) are the ones most worth failing loudly.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from vislens.index.baselines import BM25, class_prior_ranks, random_ranks, tokenize
from vislens.index.metrics import (
    ndcg_at_k,
    paired_bootstrap,
    per_query_metrics,
    ranks_from_scores,
    summarise,
)
from vislens.index.retrieval import build_index, ranks_from_embeddings


def _corpus(n: int = 500, dim: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(n, dim)).astype(np.float32)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


# ── CLAUDE.md section 6: the index is the whole corpus ────────────────────────


def test_build_index_covers_the_whole_corpus():
    corpus = _corpus(500)
    assert build_index(corpus).ntotal == len(corpus)


def test_build_index_takes_no_split_argument():
    """Asserted on the SIGNATURE, not just the behaviour.

    Reviewers assume a held-out evaluation indexes only held-out items, and an
    index built that way measures a much easier task than the product performs.
    The rule is that the queries are held out and the corpus is not, so there
    must be no parameter through which someone can quietly do otherwise.
    """
    params = set(inspect.signature(build_index).parameters)
    assert "split" not in params
    assert params == {"corpus_embeddings"}


# ── Metrics ───────────────────────────────────────────────────────────────────


def test_a_perfect_system_scores_one_everywhere():
    scores = np.eye(5) * 10
    assert summarise(ranks_from_scores(scores, np.arange(5)))["recall@1"] == 1.0


def test_ties_count_against_the_system_being_scored():
    """The class-prior baseline produces ties by construction, and a
    degenerate all-ties scorer must not be rewarded for them."""
    everything_tied = np.zeros((5, 5))
    ranks = ranks_from_scores(everything_tied, np.arange(5))
    assert (ranks == 5).all()
    assert summarise(ranks)["recall@1"] == 0.0


def test_ndcg_is_zero_outside_the_cutoff_and_decays_inside():
    assert ndcg_at_k(np.array([1]), 10)[0] == pytest.approx(1.0)
    assert ndcg_at_k(np.array([10]), 10)[0] > 0
    assert ndcg_at_k(np.array([11]), 10)[0] == 0.0


def test_chunked_and_dense_ranking_agree():
    """`ranks_from_embeddings` chunks to avoid a 5.8 GB score matrix; it must
    give byte-identical answers to scoring the whole thing at once."""
    corpus = _corpus(300)
    index = build_index(corpus)
    targets = np.arange(40)
    queries = corpus[targets]
    dense = ranks_from_scores(index.scores(queries), targets)
    chunked = ranks_from_embeddings(index, queries, targets, chunk=7)
    assert np.array_equal(dense, chunked)


# ── CLAUDE.md section 4: a delta whose CI crosses zero is not claimed ─────────


def test_an_identical_system_is_not_claimed_as_better():
    values = np.random.default_rng(0).random(500)
    delta = paired_bootstrap(values, values.copy(), metric="recall@1", resamples=500)
    assert delta.delta == pytest.approx(0.0)
    assert not delta.claimed
    assert "crosses 0" in delta.format_ci()


def test_pure_noise_is_not_claimed():
    rng = np.random.default_rng(1)
    delta = paired_bootstrap(rng.random(200), rng.random(200), resamples=500)
    assert not delta.claimed


def test_a_real_improvement_is_claimed():
    rng = np.random.default_rng(2)
    baseline = rng.random(2000) * 0.1
    better = baseline + 0.2
    delta = paired_bootstrap(better, baseline, resamples=500)
    assert delta.claimed and delta.ci_low > 0


def test_the_bootstrap_is_paired_not_independent():
    """Pairing is the point: the same query is easy or hard for both systems,
    and resampling independently would inflate the interval with variance that
    cancels. A constant offset therefore has an interval of essentially zero
    width, which independent resampling could never produce."""
    rng = np.random.default_rng(3)
    baseline = rng.random(400)
    delta = paired_bootstrap(baseline + 0.05, baseline, resamples=1000)
    assert delta.ci_high - delta.ci_low < 1e-9


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError):
        paired_bootstrap(np.zeros(5), np.zeros(6))


# ── CLAUDE.md section 3: baselines computed at runtime ────────────────────────


def test_random_lands_on_the_expected_floor():
    ranks = random_ranks(20_000, 1_000, seed=0)
    # R@50 for a uniform rank over 1000 items is 50/1000.
    assert per_query_metrics(ranks)["recall@50"].mean() == pytest.approx(0.05, abs=0.01)


def test_class_prior_ranks_every_query_the_same_way():
    """It is a prior, not a retrieval system: the ordering cannot depend on
    the query, so a given target always lands in the same position."""
    types = ["CASE"] * 60 + ["SHOES"] * 30 + ["HOME"] * 10
    first = class_prior_ranks(types, np.array([5, 70, 95]), seed=0)
    again = class_prior_ranks(types, np.array([5, 70, 95]), seed=0)
    assert np.array_equal(first, again)
    # The biggest product type ranks ahead of the smallest.
    assert first[0] < first[2]


def test_class_prior_ties_are_seeded_not_corpus_order():
    types = ["CASE"] * 50
    a = class_prior_ranks(types, np.arange(50), seed=0)
    b = class_prior_ranks(types, np.arange(50), seed=1)
    assert not np.array_equal(a, b), "ties must be broken by the seed"
    assert sorted(a.tolist()) == list(range(1, 51))


def test_bm25_finds_an_exact_title_and_is_blind_without_overlap():
    """Both halves matter. The first is why BM25 is a leakage detector on this
    task — the query IS the target's title. The second is that a query sharing
    no term scores 0, ties with most of the corpus, and lands at the bottom."""
    docs = [f"blue widget number {i}" for i in range(200)]
    bm25 = BM25(docs)
    targets = np.array([7, 42, 190])
    exact = bm25.ranks([docs[t] for t in targets], targets)
    assert (exact == 1).all()

    blind = bm25.ranks(["zzzz qqqq"] * 3, targets)
    assert (blind == len(docs)).all()


def test_tokenize_is_case_and_punctuation_insensitive():
    assert tokenize("Echo-Dot (3rd Gen.)") == ["echo", "dot", "3rd", "gen"]
