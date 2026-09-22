"""The baselines the model has to beat, computed at runtime.

`CLAUDE.md` section 3: never hardcode a floor. Each of these is computed from
the same loaded corpus and query set as the model's own numbers, inside the
same process — because the corpus changes, and a hardcoded floor "would
quietly start lying", reporting the full set's floor against a subset's score.

Four are required for text-to-image, and the honest `N/A` for image-to-image
is marked rather than omitted:

*   **Random** — the sanity floor. If a model cannot beat this, nothing else
    in the report matters.
*   **Class-prior** — the corpus ranked by descending global `product_type`
    frequency, ties broken by a fixed seed. This is the "guess the biggest
    category" strategy, and on this catalogue it is not trivial: one product
    type is 45% of the rows.
*   **BM25 over titles** — the text-only strong baseline, and the most
    uncomfortable one. It answers "how much of this task needs vision at all",
    and on a catalogue where the query is a title and the target's own title
    is in the corpus, it can be very hard to beat.
*   **Frozen zero-shot CLIP** — computed in the harness, since it needs the
    same embedding machinery as the model.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np


def random_ranks(n_queries: int, n_corpus: int, seed: int = 0) -> np.ndarray:
    """The rank of a fixed target under a uniformly random permutation.

    Sampled directly rather than by shuffling the corpus per query: the
    distribution of "where did the target land" is exactly uniform over
    1..n_corpus, and building 12,000 permutations of 121,307 items to observe
    one number each is 1.5 billion pointless writes.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(1, n_corpus + 1, size=n_queries, dtype=np.int64)


def class_prior_ranks(
    corpus_types: list[str],
    targets: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """One fixed corpus ordering, by descending global product_type frequency.

    Every query gets the SAME ranking — that is what makes it a prior rather
    than a retrieval system — so a target's rank is just its position in that
    ordering. Ties within a product type are broken by a seeded shuffle, not
    by corpus order, which would otherwise leak whatever ordering the parquet
    happened to have.
    """
    counts = Counter(corpus_types)
    rng = np.random.default_rng(seed)
    jitter = rng.permutation(len(corpus_types))
    order = sorted(
        range(len(corpus_types)),
        key=lambda i: (-counts[corpus_types[i]], jitter[i]),
    )
    position = np.empty(len(corpus_types), dtype=np.int64)
    position[np.array(order, dtype=np.int64)] = np.arange(1, len(corpus_types) + 1)
    return position[targets]


def tokenize(text: str) -> list[str]:
    return [t for t in "".join(c if c.isalnum() else " " for c in text.lower()).split() if t]


class BM25:
    """Okapi BM25 over the corpus titles.

    Implemented here rather than pulled in as a dependency: it is forty lines,
    and this repo has already declined opencv and Spark on the same grounds.

    Scoring is sparse — only documents sharing a query term can score above
    zero, so the posting lists are walked instead of the corpus. A dense
    12,000 x 121,307 score matrix would be 5.8 GB and most of it zeros.
    """

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(documents)
        self.doc_len = np.zeros(self.n_docs, dtype=np.float32)
        self.postings: dict[str, list[tuple[int, int]]] = {}

        for doc_id, text in enumerate(documents):
            tokens = tokenize(text)
            self.doc_len[doc_id] = len(tokens)
            for term, freq in Counter(tokens).items():
                self.postings.setdefault(term, []).append((doc_id, freq))

        self.avg_len = float(self.doc_len.mean()) if self.n_docs else 0.0
        self.idf = {
            term: math.log(1.0 + (self.n_docs - len(posting) + 0.5) / (len(posting) + 0.5))
            for term, posting in self.postings.items()
        }

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n_docs, dtype=np.float32)
        for term in set(tokenize(query)):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf[term]
            ids = np.fromiter((d for d, _ in posting), dtype=np.int64, count=len(posting))
            freqs = np.fromiter((f for _, f in posting), dtype=np.float32, count=len(posting))
            denom = freqs + self.k1 * (
                1 - self.b + self.b * self.doc_len[ids] / max(self.avg_len, 1e-9)
            )
            out[ids] += idf * (freqs * (self.k1 + 1)) / denom
        return out

    def ranks(self, queries: list[str], targets: np.ndarray) -> np.ndarray:
        """Rank of each target, ties counted against BM25.

        Every document a query shares no term with scores exactly 0.0, so a
        target scoring 0 is tied with most of the corpus — and pessimistic
        tie-breaking puts it near the bottom, which is the honest reading of
        "the text baseline had nothing to go on".
        """
        out = np.empty(len(queries), dtype=np.int64)
        for i, query in enumerate(queries):
            scores = self.scores(query)
            target_score = scores[targets[i]]
            better = int((scores > target_score).sum())
            tied = int((scores == target_score).sum()) - 1
            out[i] = better + tied + 1
        return out
