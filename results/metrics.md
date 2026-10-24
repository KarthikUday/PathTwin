# Key metrics

Curated, source-linked summary of the headline numbers reported in the root
[`README.md`](../README.md). Every number here is read directly from the
result file cited next to it — nothing here is recomputed or rounded beyond
what's shown in that file.

## Molecular docking pose accuracy

| Target | Pose deviation from crystal ligand | Source |
|---|---|---|
| FosA vs. fosfomycin (PDB 5V3D) | 0.15 Å (centroid-to-centroid) | `results/docking/fosa_fosfomycin/validation_summary.json` |

## Docking screening accuracy (active vs. decoy, AUC-ROC, all 8 targets)

Bootstrap 95% CI, B=2000 resamples. Only targets whose CI excludes 0.5 are
"distinguishable from chance."

| Target | Point AUC | 95% CI | Distinguishable from chance? |
|---|---|---|---|
| FosA / fosfomycin | 0.90 | [0.77, 1.00] | Yes |
| PBP3 / JXJ | 0.30 | [0.13, 0.47] | Yes (wrong direction — worse than random) |
| SHV-1 / tazobactam | 0.62 | [0.45, 0.79] | No |
| OXA-23 / meropenem | 0.67 | [0.50, 0.83] | No |
| PBP2a / cefepime | 0.63 | [0.47, 0.80] | No |
| PDC-1 / avibactam | 0.50 | [0.33, 0.70] | No |
| PBP5 / benzylpenicillin | 0.37 | [0.20, 0.53] | No |
| P99 / cephalothin | 0.33 | [0.17, 0.53] | No |

Source: `results/docking/accuracy_benchmark/bootstrap_ci_summary.json`

## ML resistance-phenotype classifiers (Stage 5)

| Species / phenotype | Split | Accuracy | AUC | Source |
|---|---|---|---|---|
| K. pneumoniae, carbapenem | random | 97.4% | 0.986 | `results/ml_classifier/stage5_results.json` |
| K. pneumoniae, carbapenem | ST-grouped | 98.2% | 0.989 | `results/ml_classifier/stage5_grouped_split_comparison.json` |
| K. pneumoniae, carbapenem | study-grouped | 96.1% | 0.972 | `results/ml_classifier/stage5_grouped_split_comparison.json` |
| S. aureus, oxacillin | ST-grouped | 91.4% | 0.960 | `results/ml_classifier/saureus_stage5_results.json` |
| P. aeruginosa, fluoroquinolone | ST-grouped | 83.6% | 0.874 | `results/ml_classifier/paeruginosa_fq_stage5_results.json` |
| A. baumannii, carbapenem | ST-grouped | 80.5% | 0.696 | `results/ml_classifier/dominant_group_investigation.json` (designed split, dominant clone excluded) |
| E. faecium, vancomycin | ST-grouped | 98.8% | 1.00 | `results/ml_classifier/efaecium_vancomycin_stage5_results.json` |
| E. cloacae, ESC | ST-grouped | 98.1% | 0.875 | `results/ml_classifier/ecloacae_esc_stage5_results.json` |

**E. faecium vancomycin classifier — SHAP validation**: all 15 of the top-15
SHAP-important features are real *van*-cluster genes (VanH, VanX, VanA, VanR,
VanS) — the model learned the correct biological mechanism, not a spurious
correlate. Source: `results/ml_classifier/efaecium_vancomycin_stage5_results.json`
(`expected_marker_hits_in_top_n`).

## CRISPR-Cas / acquired-resistance-burden correlation (Stage 6)

| Species | n genomes | Test | p-value | Source |
|---|---|---|---|---|
| A. baumannii | 382 | Mann-Whitney, dominant clone excluded | 7.09 × 10⁻⁸ | `results/crispr_scan/abaumannii_crispr_amr_stats.json` |
| P. aeruginosa | 1020 | Mann-Whitney, population-wide | 2.94 × 10⁻¹² | `results/crispr_scan/paeruginosa_crispr_amr_stats.json` |

The A. baumannii population-wide (non-excluded) test alone is *not*
significant (p = 0.227) — the real effect only emerges once a single
dominant clone skewing the population is excluded. Full account in
`DECISIONS_AND_LIMITATIONS.md` under Stage 6.

## Plots

- `results/ml_classifier/shap_summary.png` — K. pneumoniae carbapenem classifier
- `results/ml_classifier/saureus_shap_summary.png` — S. aureus oxacillin classifier
- `results/ml_classifier/paeruginosa_combined_shap_summary.png` — P. aeruginosa fluoroquinolone classifier
- `results/ml_classifier/abaumannii_carbapenem_shap_summary.png` — A. baumannii carbapenem classifier
- `results/ml_classifier/efaecium_vancomycin_shap_summary.png` — E. faecium vancomycin classifier
- `results/ml_classifier/ecloacae_esc_shap_summary.png` — E. cloacae ESC classifier
