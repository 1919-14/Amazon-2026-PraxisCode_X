"""Layer 3-5: Country-stratified multi-channel blocking and candidate selection.

Layers implemented here:
  L3a Country bucket assignment
  L3b Channel A - exact normalized name + postal/house key index
  L3c Channel C - high-IDF token inverted index
  L3d Channel D - character 2-4 gram TF-IDF nearest neighbours
  L3e Channel E - phonetic + transliteration (Indic romanization, Metaphone/Soundex)
  L3f Per-channel recall measurement on the validation split
"""

from l3_l5_blocking.phonetic_channel import PhoneticChannel  # noqa: F401
