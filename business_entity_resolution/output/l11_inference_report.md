# Layer 11 Inference Report

Every run is kept: `runs` in `l11_inference_report.json` holds the full history.

| run | split | variant | references | coverage | signals |
|---|---|---|---|---|---|
| `legacy_previous_run` | test | a | 1732544 | 100.00% | sidecar |
| `test_a` (latest) | test | a | 1732544 | 100.00% | sidecar |

## Latest run: `test_a`

- thresholds: `tau_match=0.5` `tau_s=0.1` `margin=0.05` (from `l10_report`)
- open-set policy: `boost=0.1` `veto<0.5` (from `config_default`)
- retrieval signals: `sidecar`
- candidate coverage: `100.00%`
- non-empty (match): `1,579,049` | matched pairs: `4,256,296`
