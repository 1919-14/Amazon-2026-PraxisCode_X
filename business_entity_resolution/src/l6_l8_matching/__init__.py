"""Layer 6-8: training-pair construction, feature engineering and GBDT matching.

Layers implemented here:
  L6 Training pair construction (positive / hard-negative / easy-negative) and
     the hard-negative A/B variant datasets.
  L7 Pairwise feature engineering (added in the L7 step).
  L8 LightGBM matcher with 5-fold out-of-fold probabilities (added in L8).
"""
