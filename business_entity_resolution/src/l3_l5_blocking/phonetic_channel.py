from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Optional, Sequence
import numpy as np

try:
    import jellyfish
    _JELLYFISH_AVAILABLE = True
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "jellyfish>=1.0.0", "-q"])
    import jellyfish  # type: ignore[no-redef]
    _JELLYFISH_AVAILABLE = True

try:
    import pyarrow as pa
    _PYARROW_AVAILABLE = True
except ImportError:
    pa = None  # type: ignore[assignment]
    _PYARROW_AVAILABLE = False

from l2_normalization.transliteration import romanize
from l3_l5_blocking.channels import SparseTfidfChannel

_LEGAL_SUFFIX_RE = re.compile(
    r"\b("
    r"private\s+limited|pvt\s+ltd|pvt\s+limited|private\s+ltd|private|limited|ltd|pvt|llp|"
    r"limited\s+liability\s+partnership|limited\s+liability\s+company|llc|inc|incorporated|"
    r"corp|corporation|co|company|"
    r"praiveta\s+limiteda|praiveta\s+limited|praiveta|limiteda|"
    r"praivet\s+limited|praivet|praivarr\s+limirrad|praivatu\s+limitada|"
    r"praivett\s+limitett|pra\s+li|pra|li|"
    r"elaelapi|el\s+el\s+pi|elelbhi|elelpi|el\s+el\s+bhi"
    r")\b",
    re.IGNORECASE,
)

_ADDR_NOISE_RE = re.compile(
    r"\b("
    r"near|opp|opposite|road|rd|street|st|lane|marg|nagar|colony|sector|block|phase|"
    r"district|dist|pin|floor|building|bldg|plot|flat|no|behind|beside|circle|chowk|"
    r"cross|main|industrial|area|estate|complex|tower|bazaar|market|post|taluk|tehsil"
    r")\b",
    re.IGNORECASE,
)


def _canonical_core(name: str, script_hint: str = "latin") -> str:
    """Romanize (if Indic) then strip legal suffixes; return lowercased tokens."""
    if not name or not isinstance(name, str) or name.lower() == "nan":
        return ""
    if script_hint and script_hint != "latin":
        name = romanize(name, script_hint)
    cleaned = _LEGAL_SUFFIX_RE.sub(" ", name.lower())
    return " ".join(t for t in cleaned.split() if len(t) >= 2)


def _addr_core(addr: str, script_hint: str = "latin") -> str:
    """Clean address, romanize if Indic, and remove generic noise words."""
    if not addr or not isinstance(addr, str) or addr.lower() == "nan":
        return ""
    if script_hint and script_hint != "latin":
        addr = romanize(addr, script_hint)
    cleaned = _ADDR_NOISE_RE.sub(" ", addr.lower())
    tokens = [t for t in re.sub(r"[^\w\s]", " ", cleaned).split() if len(t) >= 2 and not t.isdigit()]
    return " ".join(tokens)


def _phonetic_key(name: str, script_hint: str = "latin") -> str:
    """Space-separated Metaphone codes of canonical core tokens."""
    if not _JELLYFISH_AVAILABLE:
        return _canonical_core(name, script_hint)
    core = _canonical_core(name, script_hint)
    codes = []
    for tok in core.split():
        if tok.isalpha() and len(tok) >= 2:
            m = jellyfish.metaphone(tok)
            if m:
                codes.append(m)
    return " ".join(codes)


def _addr_phonetic_key(addr: str, script_hint: str = "latin") -> str:
    """Space-separated Metaphone codes of address core tokens."""
    if not _JELLYFISH_AVAILABLE:
        return _addr_core(addr, script_hint)
    core = _addr_core(addr, script_hint)
    codes = []
    for tok in core.split():
        if tok.isalpha() and len(tok) >= 2:
            m = jellyfish.metaphone(tok)
            if m:
                codes.append(m)
    return " ".join(codes)


def _phonetic_doc(name: str, script_hint: str = "latin", addr: str = "") -> str:
    """M_<metaphone> + S_<soundex> tokens for name and address for TF-IDF index."""
    if not _JELLYFISH_AVAILABLE:
        return f"{_canonical_core(name, script_hint)} {_addr_core(addr, script_hint)}".strip()
    
    tokens = []
    core_name = _canonical_core(name, script_hint)
    for tok in core_name.split():
        if tok.isalpha() and len(tok) >= 2:
            m = jellyfish.metaphone(tok)
            if m:
                tokens.append("M_" + m)
            s = jellyfish.soundex(tok)
            if s:
                tokens.append("S_" + s)

    if addr:
        core_addr = _addr_core(addr, script_hint)
        for tok in core_addr.split():
            if tok.isalpha() and len(tok) >= 2:
                m = jellyfish.metaphone(tok)
                if m:
                    tokens.append("AM_" + m)
                s = jellyfish.soundex(tok)
                if s:
                    tokens.append("AS_" + s)

    return " ".join(tokens)


class PhoneticChannel:
    """Phonetic + transliteration candidate generation channel (Channel E).

    Deliverable schema (query_batch return value)::

        source1_entity_id   : string
        candidate_entity_ids: list<string>    (ranked best-first)
        phon_scores         : list<float32>

    Scoring convention:
        Exact canonical name + address hit >= 4.0
        Exact canonical name (+ postal)    >= 3.0
        Exact phonetic name + address hit  >= 2.5
        Exact phonetic name (+ postal)     >= 2.0
        Address-only phonetic hit          >= 1.5
        TF-IDF phonetic sparse score       in (0.0, 1.0]

    Parameters
    ----------
    max_posting : int
        Maximum candidates stored per inverted-index key.
    k_retrieve : int
        Candidates returned per reference entity.
    tfidf_min_df : int
        Minimum document frequency for phonetic tokens.
    tfidf_max_df : float
        Maximum document frequency ratio (stop-word threshold).
    tfidf_sublinear_tf : bool
        Apply sublinear TF scaling (log(1 + tf)).
    """

    NAME = "E"

    def __init__(
        self,
        max_posting: int = 2000,
        k_retrieve: int = 200,
        tfidf_min_df: int = 2,
        tfidf_max_df: float = 0.03,
        tfidf_sublinear_tf: bool = True,
    ) -> None:
        self.max_posting = max_posting
        self.k_retrieve = k_retrieve
        self.tfidf_min_df = tfidf_min_df
        self.tfidf_max_df = tfidf_max_df
        self.tfidf_sublinear_tf = tfidf_sublinear_tf
        self._cand_ids: list[str] = []
        self._by_canonical: dict[str, list[int]] = {}
        self._by_canonical_postal: dict[str, list[int]] = {}
        self._by_canonical_addr: dict[str, list[int]] = {}
        self._by_phonetic: dict[str, list[int]] = {}
        self._by_phonetic_postal: dict[str, list[int]] = {}
        self._by_phonetic_addr: dict[str, list[int]] = {}
        self._by_addr_only_phonetic: dict[str, list[int]] = {}
        self._sparse_channel: Optional[SparseTfidfChannel] = None

    @staticmethod
    def _add(index: dict, key: str, doc: int, cap: int) -> None:
        if not key:
            return
        bucket = index.get(key)
        if bucket is None:
            index[key] = [doc]
        elif len(bucket) < cap:
            bucket.append(doc)

    def build(
        self,
        cand_ids: Sequence[str],
        cand_names: Sequence[str],
        cand_scripts: Sequence[str],
        cand_postals: Sequence[str],
        cand_addrs: Optional[Sequence[str]] = None,
    ) -> None:
        """Build all blocking indices over the candidate pool.

        Parameters
        ----------
        cand_ids : Sequence[str]
            Entity IDs for each candidate record (S2/S3).
        cand_names : Sequence[str]
            Raw business names (may be Indic-script or Latin).
        cand_scripts : Sequence[str]
            Script-type labels from the normalized parquet (e.g. "devanagari").
        cand_postals : Sequence[str]
            Postal codes; empty string when absent.
        cand_addrs : Optional[Sequence[str]]
            Normalized address strings; empty string when absent.
        """
        if not _JELLYFISH_AVAILABLE:
            raise ImportError("jellyfish is required for PhoneticChannel")

        self._cand_ids = list(cand_ids)
        cap = self.max_posting
        self._by_canonical = defaultdict(list)
        self._by_canonical_postal = defaultdict(list)
        self._by_canonical_addr = defaultdict(list)
        self._by_phonetic = defaultdict(list)
        self._by_phonetic_postal = defaultdict(list)
        self._by_phonetic_addr = defaultdict(list)
        self._by_addr_only_phonetic = defaultdict(list)

        n = len(cand_ids)
        cand_addrs_clean = (
            [str(a) if (a is not None and str(a).lower() != "nan") else "" for a in cand_addrs]
            if cand_addrs is not None
            else [""] * n
        )

        ph_docs: list[str] = []
        for doc, (name, script, postal, addr) in enumerate(
            zip(cand_names, cand_scripts, cand_postals, cand_addrs_clean)
        ):
            core = _canonical_core(name, script)
            ph = _phonetic_key(name, script)
            addr_ph = _addr_phonetic_key(addr, script) if addr else ""
            ph_docs.append(_phonetic_doc(name, script, addr))

            post_str = str(postal).strip() if (postal is not None and str(postal).lower() != "nan") else ""

            if core:
                self._add(self._by_canonical, core, doc, cap)
                if post_str:
                    self._add(self._by_canonical_postal, core + "|" + post_str, doc, cap)
                if addr_ph:
                    self._add(self._by_canonical_addr, core + "|" + addr_ph, doc, cap)

            if ph:
                self._add(self._by_phonetic, ph, doc, cap)
                if post_str:
                    self._add(self._by_phonetic_postal, ph + "|" + post_str, doc, cap)
                if addr_ph:
                    self._add(self._by_phonetic_addr, ph + "|" + addr_ph, doc, cap)

            # Address-only phonetic key (only for multi-token address keys to prevent broad collisions)
            if addr_ph and len(addr_ph.split()) >= 2:
                self._add(self._by_addr_only_phonetic, addr_ph, doc, min(cap, 500))

        # Build Sparse TF-IDF channel
        eff_min_df = 1 if len(ph_docs) < 50 else self.tfidf_min_df
        eff_max_df = 1.0 if len(ph_docs) < 50 else self.tfidf_max_df

        self._sparse_channel = SparseTfidfChannel(
            name="E_tfidf",
            analyzer="word",
            ngram_range=(1, 1),
            min_df=eff_min_df,
            max_df=eff_max_df,
            max_query_terms=16,
            max_posting_scan=10000,
            sublinear_tf=self.tfidf_sublinear_tf,
        )
        self._sparse_channel.build(ph_docs)

    def query(
        self,
        ref_name: str,
        ref_script: str,
        ref_postal: str,
        ref_addr: str = "",
    ) -> tuple[list[str], list[float]]:
        """Return up to k_retrieve candidates for one reference entity."""
        res_table = self.query_batch(
            ref_ids=["q"],
            ref_names=[ref_name],
            ref_scripts=[ref_script],
            ref_postals=[ref_postal],
            ref_addrs=[ref_addr] if ref_addr else None,
            batch_size=1,
            verbose=False,
        )
        cands = res_table["candidate_entity_ids"][0].as_py()
        scores = res_table["phon_scores"][0].as_py()
        return cands, scores

    def query_batch(
        self,
        ref_ids: Sequence[str],
        ref_names: Sequence[str],
        ref_scripts: Sequence[str],
        ref_postals: Sequence[str],
        ref_addrs: Optional[Sequence[str]] = None,
        batch_size: int = 5000,
        verbose: bool = False,
    ):
        """Query all references; return PyArrow Table with frozen schema.

        Schema::

            source1_entity_id   : string
            candidate_entity_ids: list<string>   (ranked best-first)
            phon_scores         : list<float32>

        Parameters
        ----------
        ref_ids : Sequence[str]
            Source-1 entity IDs.
        ref_names : Sequence[str]
            Raw business names for each reference entity.
        ref_scripts : Sequence[str]
            Script-type labels for each reference entity.
        ref_postals : Sequence[str]
            Postal codes for each reference entity.
        ref_addrs : Optional[Sequence[str]]
            Normalized address strings for each reference entity.
        batch_size : int
            TF-IDF transform batch size.
        verbose : bool
            Print progress to stdout.

        Returns
        -------
        pa.Table
        """
        if not _PYARROW_AVAILABLE:
            raise ImportError("pyarrow is required for query_batch")

        all_ids: list[str] = []
        all_cands: list[list[str]] = []
        all_scores: list[list[float]] = []
        n = len(ref_ids)

        ref_addrs_clean = (
            [str(a) if (a is not None and str(a).lower() != "nan") else "" for a in ref_addrs]
            if ref_addrs is not None
            else [""] * n
        )

        ref_ph_docs = [
            _phonetic_doc(nm, sc, ad)
            for nm, sc, ad in zip(ref_names, ref_scripts, ref_addrs_clean)
        ]
        ref_cores = [_canonical_core(nm, sc) for nm, sc in zip(ref_names, ref_scripts)]
        ref_ph_keys = [_phonetic_key(nm, sc) for nm, sc in zip(ref_names, ref_scripts)]
        ref_addr_phs = [_addr_phonetic_key(ad, sc) if ad else "" for ad, sc in zip(ref_addrs_clean, ref_scripts)]
        ref_post_strs = [
            str(p).strip() if (p is not None and str(p).lower() != "nan") else ""
            for p in ref_postals
        ] if ref_postals is not None else [""] * n

        # Query Sparse TF-IDF channel in batches
        sparse_results: list[tuple[list[int], list[float]]] = []
        if self._sparse_channel is not None and self._sparse_channel.inverted is not None:
            for b_start in range(0, n, batch_size):
                b_end = min(b_start + batch_size, n)
                batch_docs = ref_ph_docs[b_start:b_end]
                batch_res = self._sparse_channel.query_with_scores(batch_docs, k=self.k_retrieve)
                sparse_results.extend(batch_res)
        else:
            sparse_results = [([], []) for _ in range(n)]

        k = self.k_retrieve
        for gi in range(n):
            core = ref_cores[gi]
            ph = ref_ph_keys[gi]
            addr_ph = ref_addr_phs[gi]
            postal = ref_post_strs[gi]
            score_map: dict[int, float] = {}

            # 1. Sparse TF-IDF hits
            s_docs, s_scs = sparse_results[gi]
            for doc_idx, sc in zip(s_docs, s_scs):
                score_map[doc_idx] = max(score_map.get(doc_idx, 0.0), float(sc))

            # 2. Address-only phonetic hits (when addr has >= 2 tokens)
            if addr_ph and len(addr_ph.split()) >= 2:
                for d in self._by_addr_only_phonetic.get(addr_ph, ()):
                    score_map[d] = max(score_map.get(d, 0.0), 1.5)

            # 3. Exact phonetic name hits
            if ph:
                if postal:
                    for d in self._by_phonetic_postal.get(ph + "|" + postal, ()):
                        score_map[d] = max(score_map.get(d, 0.0), 2.0)
                if addr_ph:
                    for d in self._by_phonetic_addr.get(ph + "|" + addr_ph, ()):
                        score_map[d] = max(score_map.get(d, 0.0), 2.5)
                for d in self._by_phonetic.get(ph, ()):
                    score_map[d] = max(score_map.get(d, 0.0), 2.0)

            # 4. Exact canonical name hits
            if core:
                if postal:
                    for d in self._by_canonical_postal.get(core + "|" + postal, ()):
                        score_map[d] = max(score_map.get(d, 0.0), 3.0)
                if addr_ph:
                    for d in self._by_canonical_addr.get(core + "|" + addr_ph, ()):
                        score_map[d] = max(score_map.get(d, 0.0), 4.0)
                for d in self._by_canonical.get(core, ()):
                    score_map[d] = max(score_map.get(d, 0.0), 3.0)

            # Rank best-first
            ranked = sorted(score_map.items(), key=lambda kv: -kv[1])[:k]
            all_ids.append(ref_ids[gi])
            all_cands.append([self._cand_ids[d] for d, _ in ranked])
            all_scores.append([float(sc) for _, sc in ranked])

            if verbose and ((gi + 1) % batch_size == 0 or (gi + 1) == n):
                print(f"  PhoneticChannel: {gi + 1}/{n}", flush=True)

        return pa.table({
            "source1_entity_id": pa.array(all_ids, type=pa.string()),
            "candidate_entity_ids": pa.array(all_cands, type=pa.list_(pa.string())),
            "phon_scores": pa.array(all_scores, type=pa.list_(pa.float32())),
        })
