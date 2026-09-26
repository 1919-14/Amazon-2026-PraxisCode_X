"""Layer 3-5: Country-stratified multi-channel blocking and candidate selection.

Layers implemented here:
  L3a Country bucket assignment
  L3b Channel A - exact normalized name + postal/house key index
  L3c Channel C - high-IDF token inverted index
  L3d Channel D - character 2-4 gram TF-IDF nearest neighbours
  L3e Per-channel recall measurement on the validation split
"""
