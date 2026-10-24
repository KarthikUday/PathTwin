# PathTwin

A computational pipeline for antimicrobial resistance (AMR) profiling in ESKAPE pathogens.

## 1. Overview

The ESKAPE group (*Enterococcus faecium*, *Staphylococcus aureus*, *Klebsiella pneumoniae*, *Acinetobacter baumannii*, *Pseudomonas aeruginosa* and *Enterobacter* spp.) comprises the bacteria most responsible for hard-to-treat, antibiotic-resistant infections worldwide. Resistance can be examined at several levels: the genes a genome carries, whether the encoded proteins bind candidate drugs as predicted from structure, the effect of individual resistance mutations, the extent to which phenotype can be predicted from genomic data alone, and whether the CRISPR-Cas system of a bacterium is associated with the amount of resistance DNA it acquires.

PathTwin connects these levels in a single workflow built on public genomic, structural and phenotype data. The pipeline comprises six analyses (Table 1), each applied to a subset of the six species. The analysis-by-species coverage matrix is shown in the results viewer (see [Results viewer](#4-results-viewer)). Methodological decisions, corrections made during development and negative results are documented in [`METHODOLOGY.md`](METHODOLOGY.md).

## 2. Pipeline

**Table 1.** Pipeline components.

| Section | Analysis | Tool |
|-------|----------|------|
| 2.1 | Resistance-gene profiling | RGI 6.0.8, CARD 4.0.2 |
| 2.2 | Molecular docking | AutoDock Vina 1.2.7 |
| 2.3 | Comparative mutation tracking | snippy 4.6.0, snpEff 5.0 |
| 2.4 | Mutation-impact simulation | FoldX 5.1, AutoDock Vina |
| 2.5 | Resistance-phenotype classification | Random Forest, XGBoost, SHAP |
| 2.6 | CRISPR-Cas / AMR co-occurrence | CRISPRCasTyper 1.8.0 |

### 2.1 Resistance-gene profiling

Thirteen reference and type-strain genomes (K. pneumoniae 1, S. aureus 3, A. baumannii 3, P. aeruginosa 2, E. faecium 2, E. cloacae 2) were screened against the Comprehensive Antibiotic Resistance Database (CARD) with the Resistance Gene Identifier (RGI), run on contigs without the Loose or Nudge options. All retained hits are of the Perfect (146) or Strict (713) category. Accessions are listed in `config/reference_genomes.yml`, which also holds the K. pneumoniae strain NTUH-K2044 used only as the reference in Section 2.3.

### 2.2 Molecular docking

Candidate ligands were docked to eight resistance-associated protein structures with AutoDock Vina (20 × 20 × 20 Å search box, exhaustiveness 32, random seed 42, nine poses). Two properties of the protocol were assessed.

*Pose reproduction.* The top docked pose of each crystallographic ligand was compared with its position in the deposited structure, using heavy atoms only (closest-atom distance and distance between arithmetic-mean centroids; Table 2). For fosfomycin in FosA (PDB 5V3D, chain copy A203) the centroid deviation was 0.15 Å. The comparison is computed by `scripts/analyze_docking_pose.py --pose-validation`, which writes `results/docking/pose_validation_summary.json`. Earlier summary files for PDC-1, PBP5 and P99 had been computed with all pose atoms including polar hydrogens (centroid 2.77, 0.81 and 4.62 Å); the heavy-atom values in Table 2 supersede them.

**Table 2.** Agreement between the top docked pose and the crystallographic ligand (heavy atoms).

| Species | Target (reference ligand) | Closest atom (Å) | Centroid (Å) |
|---------|---------------------------|------------------|--------------|
| *K. pneumoniae* | FosA (5V3D, fosfomycin) | 0.14 | 0.15 |
| *A. baumannii* | OXA-23 | 0.06 | 1.60 |
| *S. aureus* | PBP2a (5M18, muramic acid A703) | 0.85 | 5.14 |
| *K. pneumoniae* | SHV-1† | 5.64 | 10.91 |
| *P. aeruginosa* | PDC-1 | 0.49 | 3.10 |
| *E. faecium* | PBP5 | 0.43 | 0.91 |
| *E. cloacae* | P99/AmpC | 0.45 | 4.51 |
| *K. pneumoniae* | PBP3 (8GPW, JXJ) | 0.46 | 2.81 |

†Top pose from the original search box, compared with tazobactam (TAZ A504) in the aligned structure 1VM1.

*Screening performance.* For each target, the crystallographic ligand was docked together with 30 decoys (29 for SHV-1) and ranked among them. Decoys were drawn from PubChem within a molecular-weight and logP window around the active, restricted to C, H, N, O, S, P, F, Cl and Br, with a Tanimoto similarity to the active of at most 0.35 (Morgan fingerprint, radius 2, 2048 bits). The procedure is inspired by DUD-E but does not use DUD-E data. Because each target has a single active, the discrimination statistic (AUC) is the fraction of decoys that the active outranks by Vina affinity (ties counted as half), equivalent to the percentile rank of the active in its decoy pool. Uncertainty was estimated with a percentile bootstrap (2000 resamples of the decoys, active held fixed, seed 42).

**Table 3.** Active-versus-decoy discrimination per target.

| Species | Target | Ligand | AUC | 95% CI | Excludes 0.5 |
|---------|--------|--------|-----|--------|--------------|
| *K. pneumoniae* | FosA (5V3D) | fosfomycin | 0.900 | 0.767–1.000 | yes |
| *A. baumannii* | OXA-23 (4JF4) | meropenem | 0.667 | 0.500–0.833 | no |
| *S. aureus* | PBP2a (5M18) | cefepime* | 0.633 | 0.467–0.800 | no |
| *K. pneumoniae* | SHV-1 (1SHV) | tazobactam | 0.621 | 0.448–0.793 | no |
| *P. aeruginosa* | PDC-1 (4HEF) | avibactam | 0.500 | 0.333–0.700 | no |
| *E. faecium* | PBP5 (6MKG) | benzylpenicillin | 0.367 | 0.200–0.533 | no |
| *E. cloacae* | P99/AmpC (1BLS) | cephalothin | 0.333 | 0.167–0.533 | no |
| *K. pneumoniae* | PBP3 (8GPW) | JXJ | 0.300 | 0.133–0.467 | yes |

\*The deposited PBP2a structure contains no cefepime; the search box is centred on the bound muramic acid at the allosteric site (see Limitations).

The mean over the eight targets was 0.54 (median 0.56), below the range of approximately 0.6 to 0.85 cited in `METHODOLOGY.md` for this docking method.

*SHV-1 box correction.* The original SHV-1 search box was displaced by 12.98 Å from the corrected position. With the corrected box the AUC rose from 0.621 to 0.724 (95% CI 0.552–0.897); the paired difference (0.103, 95% CI −0.069 to 0.276) was not significant, and the distance of the docked pose from the true ligand position changed little (closest atom 5.64 Å before, 5.36 Å after).

### 2.3 Comparative mutation tracking

Variants between pairs of strains were called with snippy and filtered to CARD-curated resistance substitutions (matching gene, residue position, wild-type and mutant amino acid). Nine comparisons were run, two of them K. pneumoniae comparisons against reference NTUH-K2044 that serve as a baseline; of the remaining seven, three yielded curated substitutions.

**Table 4.** Curated resistance substitutions between strain pairs (reference → query).

| Species | Reference → query | Curated substitutions |
|---------|-------------------|-----------------------|
| *A. baumannii* | ATCC 19606 → ACICU | gyrA S81L |
| *A. baumannii* | ATCC 19606 → AYE | gyrA S81L, parC S84L |
| *P. aeruginosa* | PAO1 → PA14 | nalC G71E, nalC S209R |
| *K. pneumoniae* | NTUH-K2044 → ATCC 13883, MGH 78578 | none |
| *S. aureus* | NCTC 8325 → USA300, MRSA252 | none |
| *E. faecium* | DSM 20477 → DO | none |
| *E. cloacae* | ATCC 13047 → 9553 | none |

### 2.4 Mutation-impact simulation

For eight resistance mutations, wild-type and mutant structures were compared by FoldX `BuildModel` (three runs per mutation) and by AutoDock Vina re-docking (ΔVina = mutant − wild type; positive values indicate weaker predicted binding). FoldX values are the mean ΔΔG over the three runs (the `total energy` column of the Average file, which equals the mean of the three per-run values in the Dif file), with the standard deviation in parentheses.

**Table 5.** Mutation-impact simulation.

| Species | Target / ligand | Mutation | ΔVina (kcal/mol) | FoldX ΔΔG (SD) |
|---------|-----------------|----------|------------------|----------------|
| *K. pneumoniae* | SHV-1 / tazobactam | E166A (pipeline test) | +0.749 | −0.443 (0.005) |
| *K. pneumoniae* | SHV-1 / tazobactam | D179N | +0.667 | 0.144 (0.108) |
| *K. pneumoniae* | SHV-1 / tazobactam | G238S | +0.627 | 3.393 (0.068) |
| *P. aeruginosa* | PDC-1 / avibactam | E247K | +0.184 | 1.112 (0.248) |
| *E. cloacae* | P99 / cefepime | L293P | +0.278 | 2.697 (0.074) |
| *K. pneumoniae* | PBP3 / JXJ | L367Q | +1.183 | 3.481 (0.058) |
| *E. faecium* | PBP5 / benzylpenicillin | T485A (proxy for M485A) | −0.702 | −0.834 (0.131) |
| *A. baumannii* | OXA-23 / meropenem | S109L + D222N + P225S | +0.321 | 0.203 (0.029) |

Six of the seven resistance mutations (all except T485A) show both a weaker predicted Vina affinity and a positive FoldX ΔΔG. T485A shows tighter Vina binding together with a stabilising FoldX value. For the PBP2a target, the experimental structures 5M18 and 4CPK were compared directly (cefepime −8.003 vs −6.900 kcal/mol, ΔVina +1.103), without FoldX; this comparison has no stored result file.

### 2.5 Resistance-phenotype classification

Random Forest and XGBoost classifiers (200 trees each, random state 42, no hyperparameter search) predict a binary resistance phenotype from gene-presence features. Labels for K. pneumoniae (ciprofloxacin) come from the KlebNET-GSP dataset (12,167 phenotype records, of which 9,057 EUCAST and 3,110 CLSI; 602 intermediate calls excluded, leaving 11,565 resistant or susceptible isolates). Labels for the other species come from BV-BRC laboratory-method phenotype calls, with intermediate and conflicting calls removed. These calls were used as submitted: each record carries its own testing standard and breakpoint standards were not harmonised (for example, S. aureus oxacillin records: EUCAST 669, CLSI 881, unspecified 325). SHAP values (TreeExplainer on the Random Forest, ranked by mean absolute SHAP value) identify the features that contribute most to predictions.

To limit leakage from clonal population structure and study-specific effects, held-out sets were also built by grouping isolates by sequence type (ST) or by source bioproject (5-fold `StratifiedGroupKFold`, first fold held out). Each split was checked with a reliability rule: dominant group at most 15% of isolates, held-out fraction at least half of the nominal fold size, and both classes present in the held-out set.

**Table 6.** Random Forest performance on held-out sets.

| Species | Phenotype | Isolates (R / S) | Split | Held-out n | Accuracy | AUC |
|---------|-----------|------------------|-------|-----------|----------|-----|
| *K. pneumoniae* | ciprofloxacin | 11,565 (6,177 / 5,388) | random | 2,313 | 0.974 | 0.986 |
| | | | ST-grouped | 2,853 | 0.982 | 0.989 |
| | | | study-grouped | 1,678 | 0.961 | 0.972 |
| *S. aureus* | oxacillin | 1,475 (606 / 869) | ST-grouped | 758 | 0.914 | 0.960 |
| | | | bioproject-grouped | 80 | 0.875 (recall 0.50) | 0.746 |
| *P. aeruginosa* | fluoroquinolones | 1,020 (583 / 437) | ST-grouped | 134 | 0.836 genes only; 0.851 with regulatory SNPs | 0.874; 0.893 |
| *E. faecium* | vancomycin | 2,971 (1,443 / 1,528) | ST-grouped, original (unreliable) | 81 | 0.988 | 1.000 |
| | | | ST-grouped, designed | 830 | 0.989 | 0.989 |
| | | | bioproject-grouped | 132 | 0.614 (recall 0) | 0.967 |
| *A. baumannii* | carbapenems | 1,357 (693 / 664) | ST-grouped, original (unreliable) | 6 | 1.000 | not reported |
| | | | ST-grouped, designed | 389 | 0.805 (recall 0.46) | 0.696 |
| *E. cloacae* | cephalosporins (ESC) | 110 (107 / 3) | ST-grouped (unreliable) | 53 | 0.981 | 0.875 |

*Variability across folds.* Table 6 reports the first fold only. Evaluating all five folds with the same settings (`scripts/grouped_cv_summary.py`) reproduced the first-fold values exactly and gave, for example, an ST-grouped Random Forest accuracy of 0.972 ± 0.012 (mean ± SD) for K. pneumoniae and 0.882 ± 0.140 for A. baumannii. The number of folds that satisfy the reliability rule in the ST-grouped split was 5 of 5 (K. pneumoniae), 5 of 5 (P. aeruginosa), 4 of 5 (S. aureus), 4 of 5 (E. faecium) and 0 of 5 (A. baumannii); no fold of any species satisfied it in the study- or bioproject-grouped split. E. cloacae has three susceptible isolates, so at most three folds form and its AUCs are not interpretable.

XGBoost accuracies were within 0.002–0.04 of these values, the largest difference being P. aeruginosa with regulatory SNPs (0.813). The E. faecium bioproject-grouped model predicted no resistant isolate, and bioproject information was available for only 25.2% of genomes. Bioproject-grouped held-out sets for P. aeruginosa (n = 12), A. baumannii (n = 118, 6 susceptible) and E. cloacae (n = 5) were too small or too unbalanced to interpret, and their metrics are not reported here. No confidence intervals were computed for any classifier metric.

*Feature attribution.* For E. faecium, the 15 highest-ranked SHAP features all correspond to genes of the *van* cluster (VanA, VanH, VanX and others); several of the 15 are alternative annotations of the same gene. For A. baumannii, three of the five expected carbapenem markers (OXA-23, OXA-24, OXA-40, OXA-58, ISAba1) appear in the top four features. For E. cloacae, none of the expected AmpC-family markers appears among the leading features. The SHAP plots for these three species were produced from the original ST-grouped splits that were flagged unreliable; the K. pneumoniae plot is from the random split, and the P. aeruginosa plot is from the model with regulatory SNPs.

### 2.6 CRISPR-Cas / AMR co-occurrence

A genome is classed as CRISPR-Cas positive when CRISPRCasTyper (e-value 0.01, sequence and HMM coverage 0.3, prediction probability 0.75, seed 42) reports at least one operon with a subtype prediction; operons with an Ambiguous or False prediction are treated as putative and do not count. Acquired-resistance burden is the number of resistance genes classified as acquired (RGI output for *P. aeruginosa*, for which RGI was run with the DIAMOND aligner; BV-BRC annotated resistance features for the other species), after removing intrinsic single-copy families (for *P. aeruginosa*, additional copies beyond the first of intrinsic families are counted as acquired). Burden was compared between Cas-positive and Cas-negative genomes with a two-sided Mann-Whitney U test, and the proportion of high-burden genomes (above the median) with Fisher's exact test.

**Table 7.** CRISPR-Cas status and acquired-resistance burden.

| Species | Genomes (Cas+ / Cas−) | Median burden (Cas+ / Cas−) | Mann-Whitney U | p |
|---------|-----------------------|-----------------------------|----------------|---|
| *P. aeruginosa* | 441 / 579 | 8 / 9 | 96,131.5 | 2.94 × 10⁻¹² |
| *A. baumannii*, all genomes | 75 / 307 | 18 / 12 | 12,546.5 | 0.227 |
| *A. baumannii*, dominant ST excluded | 75 / 139 | 18 / 11 | 7,532 | 7.09 × 10⁻⁸ |
| *E. faecium* | 4 / 318 | 6 / 17 | 194 | 0.017 |
| *E. cloacae* | 6 / 100 | 11.5 / 12 | 300 | 1.0 |

In *P. aeruginosa*, Cas-positive genomes carried fewer acquired resistance genes (Fisher odds ratio of high burden 0.391, p = 9.4 × 10⁻¹³). In *A. baumannii*, the dominant sequence type (MLST.abaumannii_2.2; 168 of 382 genomes, 44%) contains no Cas-positive genomes and has a median burden of 23. The whole-population test was run first (p = 0.227); the clonal structure was then examined and this sequence type was excluded because it has no within-clone variance in CRISPR-Cas status. After exclusion, Cas-positive genomes carried more acquired resistance genes (odds ratio 5.55, p = 1.5 × 10⁻⁸), a direction opposite to the inverse association reported in the literature (whole-population Fisher odds ratio 1.54, p = 0.12; sensitivity analysis in `scripts/crispr_clone_sensitivity.py`). For *E. faecium* (4 Cas-positive of 322), *E. cloacae* (6 of 106) and *S. aureus* (3 of 400) too few Cas-positive genomes were available for a reliable test. As an additional sensitivity analysis, the same rule (exclusion of the largest sequence type, MLST.paeruginosa.235; 107 of 1,020 genomes, 7 Cas-positive) was applied to *P. aeruginosa*; the association remained significant but weaker (n = 913, U = 87,002.5, p = 8.8 × 10⁻⁶). No standardised effect size was computed.

## 3. Limitations

- **Docking screening.** Each target has a single active and 29–30 decoys, so the AUC is discrete and its confidence intervals are wide. With the original boxes only FosA (0.90) and PBP3 (0.30) were distinguishable from chance; with the corrected SHV-1 box a third (SHV-1, 0.724) is. For PBP3 (AUC 0.30, 95% CI 0.133–0.467) the active ranked below most decoys, so the protocol did not identify the active for this target; the value is retained in the summary statistics. The ligand (JXJ) is covalently bound to Ser307 in the crystal structure and was redocked non-covalently, which may contribute to this result; this was not tested.
- **PBP2a.** The search box is centred on the crystallographic muramic acid at the allosteric site, 45 Å from the catalytic Ser403, because the deposited structure contains no cefepime.
- **Covalent ligands.** No open-source covalent-docking tool was available locally; rescoring SHV-1 poses with a machine-learning scoring function did not improve discrimination (`METHODOLOGY.md`).
- **Pose agreement.** Closest-atom distances are small for most targets, but centroid deviations reach 3–5 Å for PDC-1, P99 and PBP2a (Table 2), and the SHV-1 pose from the original box lies 5.6 Å from the reference ligand. Agreement of a single pose with one crystallographic ligand does not establish that the scoring function ranks ligands correctly.
- **Mutation simulation.** Seven resistance mutations plus one pipeline test were simulated, each with three FoldX runs. PBP5 carries Thr rather than Met at the target position, so T485A was used as a proxy; GyrA mutations could not be simulated for lack of a structure.
- **Classifiers.** Features are gene-presence indicators. Table 6 reports one held-out fold per split; results over all five folds are in `results/ml_classifier/grouped_cv_summary.json` (see Section 2.5), and no confidence intervals were computed. Phenotype labels were taken as submitted without harmonising breakpoint standards. Three splits (E. faecium original, A. baumannii original, E. cloacae) are flagged unreliable, and E. cloacae has 3 susceptible isolates.
- **CRISPR-Cas analysis.** Results are associations. Only the *P. aeruginosa* and *A. baumannii* tests were interpretable, and they differ in direction. The exclusion of the dominant A. baumannii sequence type followed inspection of the whole-population result and was therefore a post hoc decision; it is implemented in `scripts/crispr_clone_sensitivity.py`. Six A. baumannii genomes without a recorded sequence type were retained in both analyses. Eighteen A. baumannii genomes lacked a CRISPR summary.

## 4. Results viewer

An interactive viewer built with Streamlit shows the analysis-by-species coverage matrix and per-analysis tables from files under `results/` and performs no computation; the four headline figures on the Overview page are fixed text. Missing results appear as explicit gaps. The repository tracks only `results/metrics.md` and the SHAP plots, so on a fresh clone most pages show gaps until the pipeline has been run.

```bash
mamba activate pathtwin
streamlit run dashboard/app.py
```

The viewer is served at `localhost:8501`. The Overview page shows the analysis-by-species coverage matrix.

## 5. Installation

The pipeline requires Linux. On Windows, use WSL2 (Ubuntu), because several dependencies (notably the BLAST, DIAMOND and Prodigal components of RGI) are not available natively.

```bash
mamba env create -f environment.yml
conda activate pathtwin
rgi auto_load --clean   # one-time CARD database setup
```

The analyses in Sections 2.2, 2.3 and 2.6 use separate environments defined in `envs/` (`pathtwin-docking`, `pathtwin-snippy`, `pathtwin-crispr`), because a dependency conflict made a shared environment unworkable. The header of each script states the environment it requires.

> **Note.** Load the CARD database with `rgi auto_load --clean`. The `rgi load` command alone left the database out of sync with RGI on two occasions (`METHODOLOGY.md`, "CARD database mis-load").

## 6. Repository structure

```
scripts/      pipeline scripts, one per task
dashboard/    Streamlit results viewer
config/       reference genome accessions and docking box definitions
envs/         per-analysis conda environments
results/      metrics.md and SHAP plots (tracked); other outputs are produced by the pipeline
data/         inputs (not tracked; obtainable from public sources)
```

## 7. Reproducibility

*Data sources.* Genomes and structures are public (NCBI assemblies, RCSB PDB); phenotype labels come from KlebNET-GSP (K. pneumoniae) and BV-BRC (other species); decoys come from PubChem.

*Software.* Versions are fixed in `environment.yml` and `envs/*.yml` (for example scikit-learn 1.7.2, XGBoost 3.2.0, SHAP 0.48.0 in the main environment).

*Randomness.* A random seed of 42 is used for docking, decoy generation, bootstrap resampling, all classifiers and CRISPRCasTyper. No seed is set for FoldX.

*Hardware.* Development was done under WSL2 (Ubuntu); XGBoost for P. aeruginosa used an NVIDIA RTX 4090 GPU. No consolidated runtime log exists.

*Execution.* There is no single driver script. Table 8 lists, for each table above, the scripts that produce it (in the order in which they are run) and the file in which the result is stored.

Scripts are run from the repository root. The analysis scripts added for this report refuse to overwrite an existing output file.

```bash
python scripts/bootstrap_auc_ci.py
python scripts/analyze_docking_pose.py --pose-validation --data-root . --out results/docking/pose_validation_summary.json
python scripts/crispr_clone_sensitivity.py --data-root . --out-dir results/crispr_scan
python scripts/grouped_cv_summary.py --data-root . --out results/ml_classifier/grouped_cv_summary.json
```

These were run in the `pathtwin` environment (`analyze_docking_pose.py` uses only the Python standard library). Example output of `crispr_clone_sensitivity.py`:

```
A. baumannii whole population: n=382 (75/307) U=12546.5 p=0.2271; Fisher OR=1.541 p=0.1217
A. baumannii excluding MLST.abaumannii_2.2 (168, 44.0%, Cas-positive 0): n=214 (75/139) U=7532.0 p=7.092e-08; Fisher OR=5.552 p=1.54e-08
```

Example output of `bootstrap_auc_ci.py` (first rows):

```
target                  active               n     AUC            95% CI      vs 0.5
fosa_fosfomycin         fosfomycin          30     0.9    [0.767, 1.000]        DISTINCT
oxa23_meropenem         meropenem           30   0.667    [0.500, 0.833]    overlaps 0.5
p99_cephalothin         cephalothin         30   0.333    [0.167, 0.533]    overlaps 0.5
```

**Table 8.** Scripts and result files for each table.

| Table | Scripts (in run order) | Result file |
|-------|------------------------|-------------|
| 3 | `generate_all_decoys.py`, `run_accuracy_benchmark.py`, `bootstrap_auc_ci.py` | `results/docking/accuracy_benchmark/bootstrap_ci_summary.json` |
| 3 (SHV-1 box) | `run_shv1_corrected_box.py` | `results/docking/accuracy_benchmark/shv1_box_correction_and_audit.json` |
| 2 (pose agreement) | `analyze_docking_pose.py --pose-validation` | `results/docking/pose_validation_summary.json` |
| 4 | `mutation_scan.py`, `filter_resistance_snps.py` | `results/*_mutations/resistance_relevant_snps.tab` |
| 5 | `mutation_simulator.py` | `results/mutation_sim/*/result.json` |
| 6 (K. pneumoniae) | `prep_klebnet_data.py`, `train_classifier.py` | `results/ml_classifier/stage5_results.json`, `stage5_grouped_split_comparison.json` |
| 6 (S. aureus) | `prep_saureus_bvbrc_data.py`, `train_saureus_classifier.py` | `results/ml_classifier/saureus_stage5_results.json` |
| 6 (P. aeruginosa) | `prep_paeruginosa_bvbrc_data.py`, `run_paeruginosa_pipeline.py`, `train_paeruginosa_classifier.py` | `results/ml_classifier/paeruginosa_fq_stage5_results.json` |
| 6 (other species) | `prep_*_bvbrc_data.py`, `train_bvbrc_classifier.py`, `investigate_dominant_group_splits.py` | `results/ml_classifier/*_stage5_results.json`, `dominant_group_investigation.json` |
| 6 (all five folds) | `grouped_cv_summary.py` | `results/ml_classifier/grouped_cv_summary.json` |
| 7 | `prep_crispr_assemblies.py`, `batch_crispr_scan.py`, `analyze_crispr_amr_correlation.py` | `results/crispr_scan/*_crispr_amr_stats.json` |
| 7 (dominant-ST exclusion) | `crispr_clone_sensitivity.py` | `results/crispr_scan/abaumannii_clone_sensitivity.json` |

## 8. Citation

If you use this repository, please cite it using the "Cite this repository" entry on GitHub (generated from [`CITATION.cff`](CITATION.cff)), or as:

> Uday, K. (2026). *PathTwin: a computational pipeline for antimicrobial resistance profiling in ESKAPE pathogens* [Computer software]. https://github.com/KarthikUday/PathTwin

## 9. License

MIT. See [`LICENSE`](LICENSE).
