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

Block-wise mode (memory)
------------------------
Building the matrix for a whole country bucket at once needs multiple GB for the
train buckets (6.2M / 4.1M candidates). The block-wise path instead fixes the
vocabulary and IDF once, from a bounded sample (:meth:`SparseTfidfChannel.fit_vocab`),
and then builds one block's inverted index at a time
(:meth:`SparseTfidfChannel.build_with_vocab`). Because every block shares the same
vocabulary, IDF and term-selection statistics, scores stay comparable across
blocks and the per-block top-k results can be merged exactly.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency is pinned in requirements
    CountVectorizer = None  # type: ignore[assignment]
    TfidfVectorizer = None  # type: ignore[assignment]
    _SKLEARN_AVAILABLE = False


class ExactKeyChannel:
    """Channel A: hash lookup on exact normalized name and address components.

    Builds three sparse maps and queries them in decreasing specificity:
    ``name|postal`` -> ``name|house`` -> ``name``. Overly common keys (more than
    ``max_posting`` candidates) are capped to keep the result compact.
    """

    NAME = "A"

    # Specificity tier per key type: name+postal is stronger evidence than
    # name+house, which is stronger than a bare name hit.
    TIER_NAME_POSTAL = 3
    TIER_NAME_HOUSE = 2
    TIER_NAME = 1

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

    def query_tiered(
        self,
        name: str,
        postal: str,
        house: str,
        k: int,
    ) -> tuple[list[int], list[int]]:
        """Return up to ``k`` doc-indices plus the specificity tier of each hit.

        The tier is what makes cross-block merging possible: a ``name|postal`` hit
        outranks a bare-name hit no matter which candidate block produced it.
        """
        if not name:
            return [], []

        out: list[int] = []
        tiers: list[int] = []
        seen: set[int] = set()

        def collect(index: dict[str, list[int]], key: str, tier: int) -> bool:
            """Add postings for ``key``; return True if the key matched anything."""
            added = False
            for doc in index.get(key, ()):
                if doc not in seen:
                    seen.add(doc)
                    out.append(doc)
                    tiers.append(tier)
                    added = True
                    if len(out) >= k:
                        break
            return added

        # Most specific key first; stop as soon as one of them matches.
        if postal and collect(self.by_name_postal, f"{name}|{postal}", self.TIER_NAME_POSTAL):
            return out, tiers
        if house and collect(self.by_name_house, f"{name}|{house}", self.TIER_NAME_HOUSE):
            return out, tiers
        collect(self.by_name, name, self.TIER_NAME)
        return out, tiers

    def query(self, name: str, postal: str, house: str, k: int) -> list[int]:
        """Return up to ``k`` candidate doc-indices for one reference record."""
        return self.query_tiered(name, postal, house, k)[0]


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
        sublinear_tf: bool = True,
    ) -> None:
        if not _SKLEARN_AVAILABLE:  # pragma: no cover
            raise ImportError("scikit-learn is required for sparse channels")
        self.NAME = name
        self.max_query_terms = max_query_terms
        self.max_posting_scan = max_posting_scan
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            dtype=np.float32,
        )
        # Inverted index (CSC: terms x docs). ``self.inverted is None`` until built.
        self.inverted = None
        self.term_df = None

        # Block-wise state: a vocabulary + IDF shared by every block.
        self.vocabulary_: Optional[list[str]] = None
        self.idf_: Optional[np.ndarray] = None
        self.global_term_df: Optional[np.ndarray] = None
        self._vocab_index: Optional[dict[str, int]] = None
        self._counter: Optional[CountVectorizer] = None

    # ------------------------------------------------------------------
    # Classic whole-pool build (small pools, unit tests)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Block-wise build (bounded memory)
    # ------------------------------------------------------------------
    def fit_vocab(self, docs: Sequence[str]) -> int:
        """Fit the shared vocabulary + IDF on a bounded sample of the pool.

        The sample must be large enough to be representative; the number of
        distinct terms (and therefore the index size) is bounded by the sample,
        while the IDF values estimate the pool's document frequencies.

        Returns the vocabulary size (0 when the sample yields no terms).
        """
        self.vocabulary_ = None
        self.idf_ = None
        self.global_term_df = None
        self._vocab_index = None
        self._counter = None
        if len(docs) == 0:
            return 0
        try:
            matrix = self.vectorizer.fit_transform(docs)
        except ValueError:
            return 0

        self.vocabulary_ = [str(term) for term in self.vectorizer.get_feature_names_out()]
        self.idf_ = np.asarray(self.vectorizer.idf_, dtype=np.float32)
        csc = matrix.tocsc()
        self.global_term_df = np.asarray(np.diff(csc.indptr), dtype=np.int64)
        del matrix, csc

        self._vocab_index = {term: index for index, term in enumerate(self.vocabulary_)}
        self._counter = CountVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            vocabulary=self._vocab_index,
            dtype=np.float32,
        )
        return len(self.vocabulary_)

    def build_with_vocab(self, docs: Sequence[str]) -> None:
        """Build the inverted index for one block using the shared vocabulary."""
        if len(docs) == 0 or self._counter is None or self.vocabulary_ is None:
            self.inverted = None
            self.term_df = self.global_term_df
            return
        counts = self._counter.transform(list(docs))
        matrix = self._apply_idf(counts)
        self.inverted = matrix.tocsc()
        # Term selection uses the shared sample df so the rarest-term choice is
        # identical for every block (block-local df would vary per block).
        self.term_df = (
            self.global_term_df
            if self.global_term_df is not None and len(self.global_term_df) == self.inverted.shape[0]
            else np.diff(self.inverted.indptr)
        )

    def _apply_idf(self, counts):
        """Turn raw term counts into L2-normalised sublinear TF-IDF (float32)."""
        matrix = counts.tocsr()
        if self.sublinear_tf and matrix.nnz:
            matrix.data = 1.0 + np.log(matrix.data)
        if self.idf_ is not None:
            matrix = matrix.multiply(self.idf_[None, :]).tocsr()
        if matrix.nnz:
            squared = matrix.copy()
            squared.data = squared.data**2
            row_norms = np.sqrt(np.asarray(squared.sum(axis=1)).ravel())
            del squared
            row_norms[row_norms == 0.0] = 1.0
            matrix.data *= np.repeat(1.0 / row_norms, np.diff(matrix.indptr))
        return matrix

    def _transform_for_query(self, docs: Sequence[str]):
        """Vectorise queries into the same space as the candidate index."""
        if self._counter is not None and self.vocabulary_ is not None:
            return self._apply_idf(self._counter.transform(list(docs)))
        return self.vectorizer.transform(docs).tocsr()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    @staticmethod
    def _top_k(unique_docs: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
        """Deterministic top-k: score descending, ties broken by ascending doc index.

        ``np.argpartition`` alone returns an arbitrary subset when candidates tie at
        the cutoff score, which made candidate sets (and therefore the blocking
        artifact) non-reproducible between runs and between the whole-pool and
        block-wise engines. Ties are resolved explicitly here.
        """
        if len(unique_docs) <= k:
            return np.lexsort((unique_docs, -scores))

        part = np.argpartition(-scores, k)[:k]
        threshold = float(scores[part].min())
        better = np.flatnonzero(scores > threshold)
        ties = np.flatnonzero(scores == threshold)
        remaining = max(0, k - len(better))
        chosen_ties = ties[np.argsort(unique_docs[ties], kind="stable")][:remaining]
        chosen = np.concatenate([better, chosen_ties])
        return chosen[np.lexsort((unique_docs[chosen], -scores[chosen]))]

    def _select_terms(self, terms: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Keep the ``max_query_terms`` rarest (smallest-DF) query terms.

        The selection is canonical: ties in document frequency break on the term
        index, and the surviving terms are returned in ascending term order. That
        matters because the result used to depend on the order the vectorizer
        happened to emit the query's terms in (scikit-learn returns them
        descending), which made the "rarest terms" set - and therefore the
        candidate ranking - differ between code paths that scored identical data.
        """
        if len(terms) == 0:
            return terms, weights
        if len(terms) <= self.max_query_terms:
            order = np.argsort(terms, kind="stable")
            return terms[order], weights[order]
        term_df = np.asarray(self.term_df)[terms]
        chosen = np.lexsort((terms, term_df))[: self.max_query_terms]
        order = chosen[np.argsort(terms[chosen], kind="stable")]
        return terms[order], weights[order]

    def query_with_scores(
        self,
        docs: Sequence[str],
        k: int,
        batch_size: int = 64,  # kept for call-site compatibility
    ) -> list[tuple[list[int], list[float]]]:
        """Return up to ``k`` candidates *and their scores* for each query doc.

        Scores are the summed IDF weight of the shared rare query terms, so they
        are comparable across blocks of the same country index (which is what the
        block-wise engine merges on).
        """
        if self.inverted is None or len(docs) == 0:
            return [([], []) for _ in docs]

        query_matrix = self._transform_for_query(docs).tocsr()
        indptr = self.inverted.indptr
        indices = self.inverted.indices
        scan_cap = self.max_posting_scan

        results: list[tuple[list[int], list[float]]] = []
        for row in range(query_matrix.shape[0]):
            start, end = query_matrix.indptr[row], query_matrix.indptr[row + 1]
            terms = query_matrix.indices[start:end]
            weights = query_matrix.data[start:end]
            if len(terms) == 0:
                results.append(([], []))
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
                results.append(([], []))
                continue

            all_docs = np.concatenate(doc_chunks)
            all_weights = np.concatenate(weight_chunks)
            unique_docs, inverse = np.unique(all_docs, return_inverse=True)
            scores = np.zeros(len(unique_docs), dtype=np.float32)
            np.add.at(scores, inverse, all_weights)

            kept = [int(i) for i in self._top_k(unique_docs, scores, k) if scores[i] > 0.0]
            results.append(
                ([int(unique_docs[i]) for i in kept], [float(scores[i]) for i in kept])
            )

        return results

    def query(
        self,
        docs: Sequence[str],
        k: int,
        batch_size: int = 64,  # kept for call-site compatibility
    ) -> list[list[int]]:
        """Return up to ``k`` candidate doc-indices for each reference doc."""
        return [ids for ids, _ in self.query_with_scores(docs, k, batch_size)]
