"""L3b/c/d: independent candidate-generation channels.

Each channel exposes a ``build(docs)`` step over the candidate pool and a
``query(...)`` step that returns the best ``k`` candidate doc-indices (ranked
best-first) for one or more reference records.

Channel A  ExactKeyChannel  - exact normalized name (+ postal/house) hash key
Channel C  SparseTfidfChannel - word-level TF-IDF rare-term inverted retrieval
Channel D  SparseTfidfChannel - character 2-4 gram TF-IDF rare-term retrieval

All channels share the same doc-index space so the engine can map results back
to ``entity_id`` values. Channels are deliberately recall-oriented: precision is
recovered later by RRF (L4), truncation (L5) and the GBDT matcher (L6-L8).

Retrieval note
--------------
Full sparse-matrix scoring (``Q @ D.T``) is not tractable at this scale: common
terms/grams have posting lists of hundreds of thousands, so every query pays for
all of them. Instead each query scans only its ``max_query_terms`` rarest terms,
which bounds per-query cost by posting-list length rather than vocabulary size.
The TF-IDF vectorizer is still used for tokenisation, IDF and vocabulary, and the
candidate matrix is kept in CSC form as the inverted index.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer

    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency is pinned in requirements
    TfidfVectorizer = None  # type: ignore[assignment]
    _SKLEARN_AVAILABLE = False


class ExactKeyChannel:
    """Channel A: hash lookup on exact normalized name and address components.

    Builds three sparse maps and queries them in decreasing specificity:
    ``name|postal`` -> ``name|house`` -> ``name``. Overly common keys (more than
    ``max_posting`` candidates) are capped to keep the result compact.
    """

    NAME = "A"

    def __init__(self, max_posting: int = 2000) -> None:
        self.max_posting = max_posting
        self.by_name: dict[str, list[int]] = {}
        self.by_name_postal: dict[str, list[int]] = {}
        self.by_name_house: dict[str, list[int]] = {}

    @staticmethod
    def _add(index: dict[str, list[int]], key: str, doc: int, cap: int) -> None:
        if not key:
            return
        bucket = index.get(key)
        if bucket is None:
            index[key] = [doc]
        elif len(bucket) < cap:
            bucket.append(doc)

    def build(
        self,
        names: Sequence[str],
        postals: Sequence[str],
        houses: Sequence[str],
    ) -> None:
        """Index candidate records by name and name+address composite keys."""
        cap = self.max_posting
        for doc, (name, postal, house) in enumerate(zip(names, postals, houses)):
            if not name:
                continue
            self._add(self.by_name, name, doc, cap)
            if postal:
                self._add(self.by_name_postal, f"{name}|{postal}", doc, cap)
            if house:
                self._add(self.by_name_house, f"{name}|{house}", doc, cap)

    def query(self, name: str, postal: str, house: str, k: int) -> list[int]:
        """Return up to ``k`` candidate doc-indices for one reference record."""
        if not name:
            return []

        out: list[int] = []
        seen: set[int] = set()

        def collect(index: dict[str, list[int]], key: str) -> bool:
            """Add postings for ``key``; return True if the key matched anything."""
            added = False
            for doc in index.get(key, ()):
                if doc not in seen:
                    seen.add(doc)
                    out.append(doc)
                    added = True
                    if len(out) >= k:
                        break
            return added

        # Most specific key first; stop as soon as one of them matches.
        if postal and collect(self.by_name_postal, f"{name}|{postal}"):
            return out
        if house and collect(self.by_name_house, f"{name}|{house}"):
            return out
        collect(self.by_name, name)
        return out


class SparseTfidfChannel:
    """Channels C and D: TF-IDF rare-term inverted retrieval over the pool.

    ``analyzer`` selects word tokens (channel C) or character n-grams
    (channel D). Generic terms are pruned with ``max_df`` and hapax terms with
    ``min_df``. Retrieval reuses the fitted vocabulary so reference tokens land
    in the same space as candidates.
    """

    def __init__(
        self,
        name: str,
        analyzer: str = "word",
        ngram_range: tuple[int, int] = (1, 1),
        min_df: int = 2,
        max_df: float = 0.05,
        max_query_terms: int = 12,
        max_posting_scan: int = 10000,
    ) -> None:
        if not _SKLEARN_AVAILABLE:  # pragma: no cover
            raise ImportError("scikit-learn is required for sparse channels")
        self.NAME = name
        self.max_query_terms = max_query_terms
        self.max_posting_scan = max_posting_scan
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,
            dtype=np.float32,
        )
        # Inverted index (CSC: terms x docs). ``self.inverted is None`` until built.
        self.inverted = None
        self.term_df = None

    def build(self, docs: Sequence[str]) -> None:
        """Fit on the candidate corpus and keep the CSC inverted index."""
        self.inverted = None
        self.term_df = None
        if len(docs) == 0:
            return
        try:
            matrix = self.vectorizer.fit_transform(docs).tocsr()
        except ValueError:
            # e.g. empty vocabulary after pruning - channel simply yields nothing.
            return
        self.inverted = matrix.tocsc()
        self.term_df = np.diff(self.inverted.indptr)
        del matrix

    def _select_terms(self, terms: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Keep only the ``max_query_terms`` rarest (smallest-DF) query terms."""
        if len(terms) <= self.max_query_terms:
            return terms, weights
        term_df = self.term_df[terms]
        keep = np.argpartition(term_df, self.max_query_terms)[: self.max_query_terms]
        return terms[keep], weights[keep]

    def query(
        self,
        docs: Sequence[str],
        k: int,
        batch_size: int = 64,  # kept for call-site compatibility
    ) -> list[list[int]]:
        """Return up to ``k`` candidate doc-indices for each reference doc."""
        if self.inverted is None or len(docs) == 0:
            return [[] for _ in docs]

        query_matrix = self.vectorizer.transform(docs).tocsr()
        indptr = self.inverted.indptr
        indices = self.inverted.indices
        scan_cap = self.max_posting_scan

        results: list[list[int]] = []
        for row in range(query_matrix.shape[0]):
            start, end = query_matrix.indptr[row], query_matrix.indptr[row + 1]
            terms = query_matrix.indices[start:end]
            weights = query_matrix.data[start:end]
            if len(terms) == 0:
                results.append([])
                continue

            terms, weights = self._select_terms(terms, weights)

            doc_chunks: list[np.ndarray] = []
            weight_chunks: list[np.ndarray] = []
            for term, weight in zip(terms, weights):
                a, b = indptr[term], indptr[term + 1]
                if b > a:
                    if b - a > scan_cap:
                        b = a + scan_cap
                    doc_chunks.append(indices[a:b])
                    weight_chunks.append(np.full(b - a, weight, dtype=np.float32))

            if not doc_chunks:
                results.append([])
                continue

            all_docs = np.concatenate(doc_chunks)
            all_weights = np.concatenate(weight_chunks)
            unique_docs, inverse = np.unique(all_docs, return_inverse=True)
            scores = np.zeros(len(unique_docs), dtype=np.float32)
            np.add.at(scores, inverse, all_weights)

            if len(unique_docs) <= k:
                order = np.argsort(-scores, kind="stable")
            else:
                part = np.argpartition(-scores, k)[:k]
                order = part[np.argsort(-scores[part], kind="stable")]

            results.append([int(unique_docs[i]) for i in order if scores[i] > 0.0])

        return results
