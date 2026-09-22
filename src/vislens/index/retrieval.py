"""The corpus index, and turning scores into the rank of the right answer.

`CLAUDE.md` section 6: **the index is the full catalog; the held-out thing is
the queries.** `build_index` therefore takes no split argument, and a test
asserts `index.ntotal == len(corpus)`. Reviewers assume the opposite by
default — that a held-out evaluation indexes only held-out items — and an
index built that way reports a much easier task, because the distractors a
real lookup faces have been removed.

Exact search, by matmul, rather than FAISS. At 121K items and a 256-dimension
embedding the whole corpus is ~124 MB and a chunked matmul answers a 12K-query
eval in seconds. An approximate index would introduce a recall loss of its own
into the middle of a measurement whose entire purpose is comparing recall.

Ranks are computed per query chunk and the score block discarded. The full
score matrix for this eval would be 12,000 x 121,307 floats — 5.8 GB — and
materialising it is the obvious way to write this and the reason it would not
run.
"""

from __future__ import annotations

import numpy as np

DEFAULT_CHUNK = 512


class ExactIndex:
    """A brute-force inner-product index over L2-normalised embeddings."""

    def __init__(self, embeddings: np.ndarray):
        if embeddings.ndim != 2:
            raise ValueError(f"expected (n, dim), got {embeddings.shape}")
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    def scores(self, queries: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(queries, dtype=np.float32) @ self.embeddings.T


def build_index(corpus_embeddings: np.ndarray) -> ExactIndex:
    """Index the WHOLE corpus.

    No `split` argument, deliberately and permanently. See the module
    docstring: holding out the corpus instead of the queries measures an
    easier task than the one the product performs.
    """
    return ExactIndex(corpus_embeddings)


def ranks_from_embeddings(
    index: ExactIndex,
    queries: np.ndarray,
    targets: np.ndarray,
    chunk: int = DEFAULT_CHUNK,
) -> np.ndarray:
    """1-based rank of each query's target, without holding the score matrix.

    Ties count against the model, matching `metrics.ranks_from_scores`.
    """
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    out = np.empty(queries.shape[0], dtype=np.int64)
    for start in range(0, queries.shape[0], chunk):
        stop = min(start + chunk, queries.shape[0])
        block = index.scores(queries[start:stop])
        target_scores = block[np.arange(stop - start), targets[start:stop]]
        better = (block > target_scores[:, None]).sum(axis=1)
        tied = (block == target_scores[:, None]).sum(axis=1) - 1
        out[start:stop] = better + tied + 1
    return out
