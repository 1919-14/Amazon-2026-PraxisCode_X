"""Unit tests for transliteration and PhoneticChannel."""

import sys
from pathlib import Path
import pytest
import pyarrow as pa
import numpy as np

SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from l2_normalization.transliteration import romanize, detect_indic_script
from l3_l5_blocking.phonetic_channel import (
    PhoneticChannel,
    _canonical_core,
    _phonetic_key,
    _phonetic_doc,
)


def test_detect_indic_script():
    assert detect_indic_script("")[0] == "latin"
    assert detect_indic_script("Acme Corp")[0] == "latin"
    assert detect_indic_script("राज")[0] == "devanagari"
    assert detect_indic_script("ராஜ்")[0] == "tamil"
    assert detect_indic_script("ರಾಜ್")[0] == "kannada"
    assert detect_indic_script("రాజ్")[0] == "telugu"
    assert detect_indic_script("શક્તિ")[0] == "gujarati"
    assert detect_indic_script("গোল্ড")[0] == "bengali"
    assert detect_indic_script("ശക്തി")[0] == "malayalam"
    assert detect_indic_script("ଭିଜନ୍")[0] == "oriya"


def test_romanize_latin():
    assert romanize("Acme Corporation", script_hint="latin") == "Acme Corporation"
    assert romanize("Global Trading LLC") == "Global Trading LLC"
    assert romanize("") == ""
    assert romanize(None) == ""


def test_romanize_indic_scripts():
    devanagari_res = romanize("राज इन्वेस्टमेंट्स प्राइवेट लिमिटेड")
    assert "raj" in devanagari_res
    assert "invest" in devanagari_res

    tamil_res = romanize("ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி")
    assert "raj" in tamil_res

    kannada_res = romanize("ರಾಜ್ ಇನ್ವೆಸ್ಟ್ಮೆಂಟ್ಸ್")
    assert "raj" in kannada_res


def test_canonical_core():
    # Strips legal suffixes
    core1 = _canonical_core("Acme Private Limited", script_hint="latin")
    assert core1 == "acme"

    core2 = _canonical_core("राज इन्वेस्टमेंट्स प्राइवेट लिमिटेड", script_hint="devanagari")
    assert "raj" in core2

    # None and NaN handling
    assert _canonical_core(None) == ""
    assert _canonical_core("nan") == ""
    assert _canonical_core("") == ""


def test_phonetic_channel_build_and_query():
    cand_ids = ["C1", "C2", "C3", "C4"]
    cand_names = [
        "Raj Investments Pvt Ltd",
        "Dahlia Point Enterprise",
        "Shakti Care Limited",
        "Unrelated Business Corp",
    ]
    cand_scripts = ["latin", "latin", "latin", "latin"]
    cand_postals = ["560001", "10001", "560001", "90001"]

    channel = PhoneticChannel(max_posting=100, k_retrieve=5, tfidf_min_df=1)
    channel.build(cand_ids, cand_names, cand_scripts, cand_postals)

    # 1. Query with typo caught phonetically ("Dahlia Ponr")
    ids, scores = channel.query("Dahlia Ponr", ref_script="latin", ref_postal="10001")
    assert "C2" in ids
    assert len(ids) == len(scores)

    # 2. Query with Indic name ("राज इन्वेस्टमेंट्स")
    ids, scores = channel.query("राज इन्वेस्टमेंट्स", ref_script="devanagari", ref_postal="560001")
    assert "C1" in ids
    assert scores[ids.index("C1")] > 0.5  # High phonetic TF-IDF overlap

    # 3. Query with exact canonical match
    ids, scores = channel.query("Raj Investments", ref_script="latin", ref_postal="560001")
    assert "C1" in ids
    assert scores[ids.index("C1")] >= 3.0  # Exact canonical key match


def test_phonetic_channel_query_batch_schema():
    cand_ids = ["C1", "C2", "C3"]
    cand_names = ["Raj Investments", "Dahlia Point", "Shakti Care"]
    cand_scripts = ["latin", "latin", "latin"]
    cand_postals = ["560001", "10001", "560001"]

    channel = PhoneticChannel(max_posting=100, k_retrieve=10, tfidf_min_df=1)
    channel.build(cand_ids, cand_names, cand_scripts, cand_postals)

    ref_ids = ["R1", "R2", "R3"]
    ref_names = ["राज इन्वेस्टमेंट्स", "Dahlia Ponr", "Unknown XYZ"]
    ref_scripts = ["devanagari", "latin", "latin"]
    ref_postals = ["560001", "10001", "00000"]

    table = channel.query_batch(ref_ids, ref_names, ref_scripts, ref_postals, batch_size=2)
    assert isinstance(table, pa.Table)
    assert table.num_rows == 3

    # Check exact frozen schema
    assert table.schema.names == ["source1_entity_id", "candidate_entity_ids", "phon_scores"]
    assert table.schema.field("source1_entity_id").type == pa.string()
    assert pa.types.is_list(table.schema.field("candidate_entity_ids").type)
    assert table.schema.field("candidate_entity_ids").type.value_type == pa.string()
    assert pa.types.is_list(table.schema.field("phon_scores").type)
    assert table.schema.field("phon_scores").type.value_type == pa.float32()

    # Verify R1 matched C1 (Indic -> Latin via phonetic channel)
    row_r1_cands = table["candidate_entity_ids"][0].as_py()
    row_r1_scores = table["phon_scores"][0].as_py()
    assert "C1" in row_r1_cands
    assert row_r1_scores[row_r1_cands.index("C1")] > 0.5


def test_phonetic_channel_handles_none_and_nan():
    cand_ids = ["C1", "C2"]
    cand_names = [None, "Valid Name Ltd"]
    cand_scripts = ["latin", "latin"]
    cand_postals = [None, float("nan")]

    channel = PhoneticChannel(max_posting=50, k_retrieve=5, tfidf_min_df=1)
    channel.build(cand_ids, cand_names, cand_scripts, cand_postals)

    ref_ids = ["R1"]
    ref_names = [None]
    ref_scripts = ["latin"]
    ref_postals = [float("nan")]

    table = channel.query_batch(ref_ids, ref_names, ref_scripts, ref_postals)
    assert table.num_rows == 1
    assert table["candidate_entity_ids"][0].as_py() == []


def test_phonetic_channel_with_addresses():
    cand_ids = ["C1", "C2", "C3"]
    cand_names = [
        "Shree Ganesh Traders",
        "Apex Logistics Corp",
        "Completely Different Name",
    ]
    cand_scripts = ["latin", "latin", "latin"]
    cand_postals = ["560001", "110001", "400001"]
    cand_addrs = [
        "123 MG Road Indiranagar Bengaluru",
        "Plot 45 Okhla Phase 3 New Delhi",
        "77 Nariman Point Marine Drive Mumbai",
    ]

    channel = PhoneticChannel(max_posting=100, k_retrieve=5, tfidf_min_df=1)
    channel.build(cand_ids, cand_names, cand_scripts, cand_postals, cand_addrs=cand_addrs)

    # 1. Address-only retrieval: query with totally different/garbled name but matching address
    ids, scores = channel.query(
        "Unknown Shop",
        ref_script="latin",
        ref_postal="",
        ref_addr="Nariman Point Marine Drive Mumbai",
    )
    assert "C3" in ids
    assert scores[ids.index("C3")] >= 1.5  # Address-only phonetic hit

    # 2. Romanized address + Devanagari name match
    ids, scores = channel.query(
        "श्री गणेश ट्रेडर्स",
        ref_script="devanagari",
        ref_postal="560001",
        ref_addr="MG Road Indiranagar Bengaluru",
    )
    assert "C1" in ids
    assert scores[ids.index("C1")] >= 2.5  # Composite phonetic name + address match

    # 3. Exact canonical name + address match
    ids, scores = channel.query(
        "Shree Ganesh Traders",
        ref_script="latin",
        ref_postal="560001",
        ref_addr="123 MG Road Indiranagar Bengaluru",
    )
    assert "C1" in ids
    assert scores[ids.index("C1")] >= 4.0  # Exact canonical composite match
