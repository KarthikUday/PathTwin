# Methodology

This document describes the methods, validation procedures, corrections and negative results of the PathTwin pipeline. Results are reported in [`README.md`](README.md); table numbers below refer to that file.

## 1. Data sources

| Data | Source | Used in |
|------|--------|-----|
| Reference and type-strain genomes (13) | NCBI assemblies (accessions in `config/reference_genomes.yml`) | Sections 2.1, 2.3 |
| Resistance database | CARD 4.0.2 | Sections 2.1, 2.3, 2.5, 2.6 |
| Protein structures (8 targets) | RCSB PDB | Sections 2.2, 2.4 |
| Decoy molecules | PubChem | Section 2.2 |
| *K. pneumoniae* genomes and ciprofloxacin phenotypes | KlebNET-GSP | Section 2.5 |
| Genomes and phenotypes of the other species | BV-BRC | Sections 2.5, 2.6 |

All inputs are public. Raw and processed data are not tracked in the repository.

## 2. Methods

### 2.1 Resistance-gene profiling

Genomes were screened with RGI 6.0.8 against CARD 4.0.2, run on contigs without the Loose or Nudge options. Retained hits are of the Perfect (146) or Strict (713) category. For the 1,020 *P. aeruginosa* genomes used in Sections 2.5 and 2.6, RGI was run with the DIAMOND aligner; for strain PA14 this gave 54 hits against 58 with BLAST.

*CARD database mis-load.* On two occasions the `rgi load` command left the database out of sync with RGI. The database is therefore loaded with `rgi auto_load --clean`, and the hit counts above were obtained after this correction. Details of the affected runs were not retained.

### 2.2 Molecular docking

Receptors and ligands were docked with AutoDock Vina 1.2.7 using a 20 × 20 × 20 Å search box, exhaustiveness 32, random seed 42 and nine poses. For PBP2a (PDB 5M18), which contains no cefepime, the box was centred on the bound muramic acid at the allosteric site, 45 Å from the catalytic Ser403.

*Decoys.* For each target, 30 decoys (29 for SHV-1) were drawn from PubChem within a molecular-weight and logP window around the active, restricted to C, H, N, O, S, P, F, Cl and Br, with a Tanimoto similarity to the active of at most 0.35 (Morgan fingerprint, radius 2, 2048 bits). The design follows the principle of DUD-E (property-similar, topologically dissimilar decoys) but does not use DUD-E data.

*Discrimination statistic.* Each target has a single active. The AUC is therefore the fraction of decoys that the active outranks by Vina affinity, with ties counted as half; it equals the percentile rank of the active in its decoy pool and can take only discrete values. Confidence intervals are percentile bootstrap intervals (2000 resamples of the decoys, active held fixed, seed 42). Pairwise differences between targets (28 pairs) were assessed with nominal α = 0.05 and with a Bonferroni-corrected α of 0.00179; 16 pairs were nominally significant and 4 remained significant after correction, all involving FosA.

*Pose reproduction.* The top docked pose was compared with the crystallographic ligand using heavy atoms only: closest-atom distance and distance between arithmetic-mean centroids (`scripts/analyze_docking_pose.py --pose-validation`). For SHV-1 the reference is tazobactam (TAZ A504) in the aligned structure 1VM1.

### 2.3 Comparative mutation tracking

Variants between pairs of strains were called with snippy 4.6.0 and annotated with snpEff 5.0. A substitution was retained only if CARD curates it, that is, if gene, residue position, wild-type residue and mutant residue all match a CARD entry. The wild-type requirement excluded 13 variants at curated positions in the same gene whose residue differed from the CARD reference; for example, pncA Asp31Glu and Ala17Thr were rejected because the curated entries carry Ile31 and Gly17. GyrA Ser83Tyr is not curated (the curated substitutions at position 83 are Phe, Ile, Leu and Trp).

Nine comparisons were run (Table 4). Two *K. pneumoniae* comparisons against NTUH-K2044 serve as a baseline. The query strains in the zero-result comparisons carry acquired resistance determinants (for example mecA, tet(M), ErmB and DHA-1), which this chromosomal-substitution approach does not detect.

### 2.4 Mutation-impact simulation

Point mutations were introduced with FoldX 5.1 `BuildModel` (three runs per mutation). ΔΔG is the mean over the three runs of the `total energy` column of the Average file, which equals the mean of the three per-run values in the Dif file; the standard deviation over runs is stored as `ddg_foldx_sd`. FoldX does not print a unit; kcal/mol is assumed by convention. Wild-type and mutant receptors were then re-docked with AutoDock Vina under the settings of Section 2.2 (ΔVina = mutant − wild type; positive values indicate weaker predicted binding). Signals are called concordant when both ΔVina and ΔΔG are positive.

For SHV-1 (wild-type score −6.544 kcal/mol) and PBP3 (−7.783 kcal/mol) the wild-type scores are those of Section 2.2, and their mutant receptors were prepared with Open Babel; for the other targets both receptors were prepared in the same way. This asymmetry may affect ΔVina for these two targets and was not evaluated. PBP5 carries Thr rather than Met at position 485, so T485A was used as a proxy. GyrA mutations could not be simulated for lack of a suitable structure. FoldX has no random-seed option.

### 2.5 Resistance-phenotype classification

*Labels.* *K. pneumoniae* ciprofloxacin labels come from KlebNET-GSP (12,167 records, 9,057 EUCAST and 3,110 CLSI); 602 intermediate calls were excluded, leaving 11,565 resistant or susceptible isolates. Labels for the other species are BV-BRC laboratory-method calls with intermediate and conflicting calls removed. Calls were used as submitted, and breakpoint standards were not harmonised.

*Features and models.* Features are gene-presence indicators. Random Forest and XGBoost models (200 trees, random state 42) were trained without hyperparameter search. Feature attribution used SHAP TreeExplainer on the Random Forest, ranked by mean absolute SHAP value.

*Splits.* Held-out sets were obtained by random splitting, or by grouping on sequence type (ST) or source bioproject with `StratifiedGroupKFold` (5 folds, shuffled, seed 42). Only the first fold was held out in the main analysis. A split was called reliable if the largest group was at most 15% of the isolates, the held-out set was at least half of the nominal fold size, and both classes were present. All five folds were later evaluated with the same settings (`scripts/grouped_cv_summary.py`); fold 0 reproduced the stored values exactly. Mean and standard deviation over folds use the sample standard deviation (n − 1).

### 2.6 CRISPR-Cas / AMR co-occurrence

CRISPR-Cas systems were typed with CRISPRCasTyper 1.8.0 (e-value 0.01, sequence and HMM coverage 0.3, prediction probability 0.75, seed 42). A genome is Cas-positive if at least one operon has a subtype prediction; operons with an Ambiguous or False prediction are treated as putative and do not count.

Acquired-resistance burden is the number of resistance genes classified as acquired after removal of intrinsic single-copy families. For *P. aeruginosa* the genes are RGI hits, and additional copies beyond the first of intrinsic families count as acquired. For the other species the features are BV-BRC annotated resistance genes, with genes of unclear origin excluded. Cas-positive and Cas-negative genomes were compared with a two-sided Mann-Whitney U test, and the proportion of genomes above the median burden with Fisher's exact test. No standardised effect size was computed.

## 3. Validation and controls

1. **Pose reproduction.** Closest-atom distances were below 1 Å for seven of eight targets (Table 2). The mean-centroid distances range from 0.15 to 5.14 Å, so the agreement of a single pose does not establish correct ranking.
2. **Decoy benchmark.** The mean AUC over the eight targets was 0.54 (median 0.56). Docking-based screening is generally reported to give AUCs of about 0.6 to 0.85 depending on target and study; no primary source for this range is retained in the repository.
3. **SHV-1 search box.** The original SHV-1 box was displaced by 12.98 Å from the corrected position. With the corrected box the AUC was 0.724 (95% CI 0.552–0.897) against 0.621; the paired difference (0.103, 95% CI −0.069 to 0.276) was not significant, and the closest-atom pose distance changed from 5.64 to 5.36 Å.
4. **Grouped splits.** Grouping by ST or bioproject limits leakage from clonal structure and study effects. In the ST-grouped split, 5 of 5 folds were reliable for *K. pneumoniae* and *P. aeruginosa*, 4 of 5 for *S. aureus* and *E. faecium*, and 0 of 5 for *A. baumannii*; no fold of any species was reliable in the study- or bioproject-grouped split.
5. **Clone exclusion.** The dominant *A. baumannii* sequence type (MLST.abaumannii_2.2; 168 of 382 genomes) contains no Cas-positive genome, so its CRISPR-Cas status has no within-clone variance. The whole-population test was run first and the exclusion was decided afterwards, so the exclusion is a post hoc sensitivity analysis (`scripts/crispr_clone_sensitivity.py`). Six genomes without a recorded ST were kept in both analyses. The same rule applied to *P. aeruginosa* (ST 235, 107 genomes) is reported as an additional analysis.

## 4. Corrections made during development

| Item | Problem | Change |
|------|---------|--------|
| FoldX ΔΔG | The stored `ddg_foldx` was the SD column of the Average file, not `total energy`. | Parser reads columns by name; eight `result.json` files regenerated from the existing FoldX output; two interpretations changed (E166A, T485A). |
| SHV-1 search box | Box displaced by 12.98 Å from the ligand. | Re-docked with the corrected box (Section 3). |
| CARD database | `rgi load` left the database out of sync. | `rgi auto_load --clean` (Section 2.1). |
| *K. pneumoniae* classifier | Described as a carbapenem classifier. | Corrected to ciprofloxacin in `results/metrics.md` and the README. |
| *A. baumannii* sample size | n = 382 stated for the clone-excluded test. | Corrected to n = 214 (75 Cas-positive, 139 Cas-negative). |
| Pose centroids | Stored centroids for PDC-1, PBP5 and P99 included polar hydrogens. | Recomputed on heavy atoms (3.10, 0.91 and 4.51 Å, against 2.77, 0.81 and 4.62 Å). |
| *A. baumannii* accuracy | A held-out accuracy of 1.000 came from a 6-isolate test set. | Flagged unreliable; the designed split (n = 389) is reported. |
| Results viewer | The SHV-1 E166A result was shown under a PBP2a label. | Relabelled as a pipeline test. |

## 5. Negative and inconclusive results

- **PBP3.** The active ranked below most decoys (AUC 0.30, 95% CI 0.133–0.467). The ligand is covalently bound to Ser307 in the crystal structure and was redocked non-covalently; whether this explains the result was not tested.
- **Covalent ligands.** No open-source covalent-docking tool was available locally. Rescoring SHV-1 poses with a machine-learning scoring function did not improve discrimination.
- **PBP2a.** The deposited structure contains no cefepime, so the search box is placed at the allosteric site. The direct comparison of 5M18 and 4CPK (cefepime −8.003 and −6.900 kcal/mol) has no stored result file.
- **P99.** Only the cefepime run is stored (`results/mutation_sim/A_293_L_P/result.json`). A first run with cephalothin (ΔVina −0.265) was overwritten and survives only as text in the dashboard source.
- **Unreliable splits.** The original ST-grouped splits for *E. faecium* (n = 81), *A. baumannii* (n = 6) and *E. cloacae* (n = 53, three susceptible genomes in total) did not meet the reliability rule. The *E. faecium* bioproject-grouped model predicted no resistant isolate.
- **Feature attribution.** For *E. cloacae*, none of the expected AmpC-family markers appears among the leading features, although an ACT/MIR family feature is present in 94.5% of genomes.
- **CRISPR-Cas.** In *A. baumannii*, Cas-positive genomes carry more acquired resistance genes after exclusion of the dominant clone, the opposite of the inverse association reported in the literature.

## 6. Limitations

The limitations of the individual analyses are listed in the Limitations section of the README. In addition: phenotype labels mix testing standards; features are gene-presence indicators and omit point mutations except in the *P. aeruginosa* model with regulatory SNPs; classifier metrics have no confidence intervals; and decoy ranking with a single active per target gives wide intervals.

## 7. Reproducibility

- **Software.** Versions are fixed in `environment.yml` and `envs/*.yml` (scikit-learn 1.7.2, XGBoost 3.2.0, SHAP 0.48.0 in the main environment). The analyses in Sections 2.2, 2.3 and 2.6 use separate environments because a dependency conflict made a shared environment unworkable.
- **Randomness.** Seed 42 is used for docking, decoy generation, bootstrap resampling, all classifiers and CRISPRCasTyper. No seed is set for FoldX.
- **Hardware.** Development was done under WSL2 (Ubuntu). XGBoost for *P. aeruginosa* used an NVIDIA RTX 4090 GPU; the all-folds re-run used a CPU and reproduced the stored first-fold values.
- **Scripts and outputs.** Table 8 of the README lists the script and result file for each table.
