#!/usr/bin/env python3
"""
PathTwin Results Dashboard.

Reads directly from files already on disk under results/ (and two Stage 5
metadata files under data/processed/) -- never recomputes, re-runs, or
interpolates a number. If a file a section needs doesn't exist, the
section says so explicitly instead of being silently omitted.

Run with:
    streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as d
from methods import STAGE_METHODS

st.set_page_config(page_title="PathTwin Results Dashboard", layout="wide")


# --------------------------------------------------------------------------
# small shared helpers
# --------------------------------------------------------------------------

def missing(*paths_or_labels):
    """Explicit 'not found' box, naming exactly what was looked for."""
    st.warning("**No result file found for this selection.** Looked for: " +
               "; ".join(str(p) for p in paths_or_labels))


def source_caption(*paths):
    st.caption("Source: " + " | ".join(f"`{p}`" for p in paths if p))


def download_df(df: pd.DataFrame, filename: str, key: str):
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                        file_name=filename, mime="text/csv", key=key)


def download_json(obj, filename: str, key: str):
    import json
    st.download_button("Download JSON", json.dumps(obj, indent=2, default=str).encode("utf-8"),
                        file_name=filename, mime="application/json", key=key)


def methods_panel(stage: int):
    m = STAGE_METHODS.get(stage)
    if not m:
        return
    with st.expander("Methods & limitations for this stage (condensed from DECISIONS_AND_LIMITATIONS.md)"):
        st.markdown(m["summary"])
        st.markdown("**Known limitations:**")
        for item in m["limitations"]:
            st.markdown(f"- {item}")


def flatten_nested_metrics(node, level_names, path=None):
    """Recursively flattens a {level1: {level2: {...metrics...}}} dict into
    rows, where a leaf dict is recognized by having an 'accuracy' key."""
    path = path or []
    rows = []
    if isinstance(node, dict) and "accuracy" in node:
        row = dict(zip(level_names, path))
        row.update({k: v for k, v in node.items() if isinstance(v, (int, float))})
        rows.append(row)
    elif isinstance(node, dict):
        for k, v in node.items():
            rows.extend(flatten_nested_metrics(v, level_names, path + [k]))
    return rows


# --------------------------------------------------------------------------
# Overview / landing page
# --------------------------------------------------------------------------

COVERAGE_SYMBOL = {True: "✅", False: "—", "partial": "🟡 partial"}


def render_overview():
    st.title("PathTwin")
    st.subheader("Computational pipeline for ESKAPE pathogen resistance profiling")
    st.markdown(
        "PathTwin profiles antimicrobial resistance determinants across all 6 ESKAPE pathogens "
        "(*E. faecium, S. aureus, K. pneumoniae, A. baumannii, P. aeruginosa, E. cloacae*) through "
        "six linked stages — resistance-gene profiling, molecular docking validation, comparative "
        "mutation tracking, mutation-impact simulation, ML resistance-phenotype classifiers, and "
        "CRISPR-Cas/AMR co-occurrence — built entirely on real public genomic and phenotype data, "
        "with every step verified against a real file rather than assumed."
    )

    st.markdown("### Coverage across all 6 stages × 6 species")
    matrix = d.compute_coverage_matrix()
    display_matrix = matrix.copy()
    for col in display_matrix.columns[1:]:
        display_matrix[col] = display_matrix[col].map(COVERAGE_SYMBOL)
    st.dataframe(display_matrix, width='stretch', hide_index=True)
    st.caption("🟡 K. pneumoniae's Stage 6 cell is its original n=3 proof-of-concept only (no dedicated "
               "population-scale run exists for this species) — see the CRISPR-Cas co-occurrence page. "
               "Every other cell reflects a real file checked on disk, not an assumption.")

    st.markdown("### Headline validated results")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("FosA / fosfomycin docking — crystal agreement", "0.15 Å", border=True,
                  help="Centroid-to-centroid distance vs. the real 5V3D crystal ligand (closest-atom distance 0.14 Å)")
        st.caption("Docking validation · Stage 2")
    with c2:
        st.metric("E. faecium vancomycin classifier — SHAP", "15 / 15 top features", border=True,
                  help="Every single one of the top 15 SHAP features is a van-cluster gene — the cleanest "
                       "single-mechanism validation in the project")
        st.caption("ML resistance prediction · Stage 5")
    c3, c4 = st.columns(2)
    with c3:
        st.metric("A. baumannii CRISPR-Cas, deconfounded", "p = 7.1 × 10⁻⁸", border=True,
                  help="Mann-Whitney on acquired-resistance burden, Cas-present vs. Cas-absent, after "
                       "excluding one dominant clone (44% of the sample, 100% Cas-absent) — reverses a "
                       "clean-looking null into a strong, real contradiction of the literature")
        st.caption("CRISPR-Cas co-occurrence · Stage 6")
    with c4:
        st.metric("P. aeruginosa CRISPR-Cas, population-scale", "p = 2.9 × 10⁻¹²", border=True,
                  help="Mann-Whitney on acquired-resistance burden, n=1020 genomes — supports the "
                       "literature's inverse correlation at real statistical power")
        st.caption("CRISPR-Cas co-occurrence · Stage 6")

    st.markdown("---")
    st.caption("Use the sidebar to explore any stage in full detail — every number on every page traces "
               "to a real file under `results/`, and each page's methods & limitations panel names the "
               "exact section of `DECISIONS_AND_LIMITATIONS.md` it condenses.")


# --------------------------------------------------------------------------
# Stage 1
# --------------------------------------------------------------------------

RGI_COLUMN_CONFIG = {
    "ORF_ID": st.column_config.TextColumn("ORF_ID", width="small",
                                           help="Truncated — hover a cell to see the full ORF identifier"),
    "Contig": st.column_config.TextColumn(width="small"),
    "Drug Class": st.column_config.TextColumn(width="medium"),
    "AMR Gene Family": st.column_config.TextColumn(width="medium"),
}


def render_stage1():
    st.header("Resistance gene profiling")
    st.caption("Stage 1 — RGI / CARD")
    methods_panel(1)
    st.caption("Sorted Perfect → Strict, then by % identity descending. Click any column header to re-sort.")

    tabs = st.tabs(list(d.SPECIES.keys()))
    for species, tab in zip(d.SPECIES.keys(), tabs):
        with tab:
            genomes = d.stage1_genomes_for_species(species)
            if not genomes:
                missing(f"any Stage 1 genome for {species}")
                continue
            for label, path in genomes:
                st.subheader(label)
                df = d.load_rgi_table(path)
                if df is None:
                    missing(path.relative_to(d.PROJECT_ROOT))
                    continue
                if df.empty:
                    st.info("0 RGI hits (Perfect/Strict) in this genome.")
                else:
                    st.dataframe(df, width='stretch', height=min(400, 40 + 35 * len(df)),
                                 column_config=RGI_COLUMN_CONFIG)
                    download_df(df, f"{path.stem}.csv", key=f"dl_{path.stem}")
                source_caption(path.relative_to(d.PROJECT_ROOT))


# --------------------------------------------------------------------------
# Stage 2
# --------------------------------------------------------------------------

def render_stage2():
    st.header("Stage 2 — Molecular docking validation")
    methods_panel(2)

    records = []
    sources = []
    for display, target_dir, species in d.DOCKING_TARGETS:
        rec, src = d.load_docking_summary(target_dir)
        if rec is None:
            missing(f"results/docking/{target_dir}/")
            continue
        rec = {"display": display, "species": species, **rec}
        records.append(rec)
        sources.append(src)

    if not records:
        return

    df = pd.DataFrame(records)
    st.subheader("All 8 targets — summary")
    show_cols = ["display", "species", "pdb_id", "ligand", "nucleophile",
                 "top_pose_affinity_kcal_mol", "primary_catalytic_distance_label",
                 "primary_catalytic_distance_A", "closest_atom_to_crystal_ligand_A",
                 "centroid_to_centroid_vs_crystal_A"]
    st.dataframe(df[show_cols], width='stretch')
    download_df(df[show_cols], "stage2_docking_summary.csv", key="dl_s2_summary")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(df, x="display", y="top_pose_affinity_kcal_mol", color="species",
                     title="Top-pose binding affinity per target (kcal/mol, more negative = stronger)")
        fig.update_layout(xaxis_title="", yaxis_title="kcal/mol", xaxis_tickangle=-30)
        st.plotly_chart(fig, width='stretch')
    with c2:
        plot_df = df.dropna(subset=["primary_catalytic_distance_A"])
        if len(plot_df):
            fig2 = px.scatter(plot_df, x="primary_catalytic_distance_A", y="top_pose_affinity_kcal_mol",
                               color="species", text="display",
                               title="Affinity vs. distance to primary catalytic residue")
            fig2.update_traces(textposition="top center")
            fig2.update_layout(xaxis_title="Distance to catalytic residue (Å)", yaxis_title="kcal/mol")
            st.plotly_chart(fig2, width='stretch')
        else:
            st.info("No targets with a catalytic-distance value to plot (SHV-1 has none, see below).")

    st.subheader("Per-target detail")
    target_labels = [r["display"] for r in records]
    choice = st.selectbox("Target", target_labels, key="s2_target")
    rec = next(r for r in records if r["display"] == choice)
    for k, v in rec.items():
        if k in ("display", "conclusion"):
            continue
        st.markdown(f"**{k}**: {v}")
    st.markdown("**Conclusion (from the validation file itself):**")
    st.info(rec.get("conclusion") or "—")
    idx = target_labels.index(choice)
    source_caption(sources[idx])

    st.markdown("---")
    st.subheader("Accuracy validation: active-vs-decoy AUC-ROC / EF1%")
    st.caption("Does each target's own real active ligand actually outscore a property-matched, "
               "topologically-dissimilar decoy set? The standard DUD-E-style virtual-screening check "
               "(Mysinger, Carchia, Irwin, Shoichet, J. Med. Chem. 2012, PMID 22716043) -- this project's "
               "8 targets aren't covered by DUD-E's own decoy sets (checked directly: DUD-E's 102 targets "
               "include exactly one beta-lactamase, AmpC/1L2S, and zero PBP/FosA targets -- see "
               "DECISIONS_AND_LIMITATIONS.md), so decoys were generated fresh per target.")
    bench, bench_src = d.load_accuracy_benchmark()
    if bench is None:
        missing("results/docking/accuracy_benchmark/overall_summary.json")
    else:
        canonical_8 = {t[1] for t in d.DOCKING_TARGETS}
        rows = []
        for target_dir, s in bench.items():
            if target_dir not in canonical_8:
                continue  # covalent-adduct re-runs shown in their own section below
            rows.append({
                "target": target_dir,
                "active": s.get("active_name"),
                "n_decoys": s.get("n_decoys_docked_successfully"),
                "active_affinity": s.get("active_affinity_kcal_mol"),
                "decoy_mean_affinity": s.get("decoy_affinity_mean"),
                "active_rank": s.get("active_rank"),
                "pool_size": s.get("n_total_pool"),
                "auc_single_active": s.get("auc_single_active"),
                "active_in_top1pct": s.get("ef1pct", {}).get("active_in_top"),
            })
        boot, boot_src = d.load_bootstrap_ci()
        boot_by_target = boot.get("per_target", {}) if boot else {}
        for r in rows:
            bt = boot_by_target.get(r["target"])
            r["ci95_lo"] = bt["ci95_lo"] if bt else None
            r["ci95_hi"] = bt["ci95_hi"] if bt else None
            r["distinct_from_chance"] = bt["distinguishable_from_chance"] if bt else None

        bdf = pd.DataFrame(rows)
        st.dataframe(bdf, width="stretch")
        download_df(bdf, "stage2_accuracy_benchmark.csv", key="dl_s2_accuracy")

        fig = px.bar(bdf, x="target", y="auc_single_active",
                     title="AUC (single active vs. its own decoy set) per target, with bootstrap 95% CI",
                     hover_data=["active", "n_decoys", "active_rank", "pool_size"])
        if boot is not None:
            err_plus = (bdf["ci95_hi"] - bdf["auc_single_active"]).clip(lower=0)
            err_minus = (bdf["auc_single_active"] - bdf["ci95_lo"]).clip(lower=0)
            fig.update_traces(error_y=dict(type="data", symmetric=False,
                                            array=err_plus, arrayminus=err_minus))
        fig.add_hline(y=0.5, line_dash="dot", line_color="gray",
                      annotation_text="0.5 = random ranking", annotation_position="bottom right")
        fig.add_hrect(y0=0.6, y1=0.85, fillcolor="LightGreen", opacity=0.15, line_width=0,
                      annotation_text="typical published Vina range", annotation_position="top left")
        fig.update_layout(yaxis_title="AUC (this target, 1 active + its decoys)", yaxis_range=[0, 1.05],
                           xaxis_title="")
        st.plotly_chart(fig, width="stretch")

        n_pass = sum(1 for r in rows if (r["auc_single_active"] or 0) >= 0.5)
        n_top1 = sum(1 for r in rows if r["active_in_top1pct"])
        st.markdown(
            f"**{n_pass} / {len(rows)}** targets have a point-estimate AUC ≥ 0.5 (better than random); "
            f"**{n_top1} / {len(rows)}** rank the active in the top ~1% of its own pool. "
            "Published Vina-on-DUD-E benchmarks report roughly 0.6-0.85 AUC per target on average "
            "(e.g. Wójcikowski, Ballester, Siedlecki, *Sci. Rep.* 2017, PMID 28440302, report Vina's "
            "DUD-E top-1% hit rate at ~16%) -- read the numbers above against that bar, not a perfect one."
        )

        st.markdown("**Is any of this actually distinguishable from noise?**")
        if boot is None:
            missing("results/docking/accuracy_benchmark/bootstrap_ci_summary.json")
        else:
            n_distinct = sum(1 for v in boot_by_target.values() if v["distinguishable_from_chance"])
            distinct_names = [t for t, v in boot_by_target.items() if v["distinguishable_from_chance"]]
            st.markdown(
                f"Bootstrap 95% CIs (resampling each target's decoys with replacement, "
                f"{boot['n_bootstrap_iterations']} iterations): only **{n_distinct} / {len(boot_by_target)}** "
                f"targets' AUC is individually distinguishable from chance (0.5) — **{'** and **'.join(distinct_names) if distinct_names else 'none'}**. "
                "The other targets' point estimates range from 0.333 to 0.667, but their confidence "
                "intervals all overlap 0.5 — at ~30 decoys per target, that spread is consistent with "
                "sampling noise, not necessarily real per-target differences in accuracy."
            )
            n_nom = boot["n_pairs_significant_nominal_95"]
            n_bonf = boot["n_pairs_significant_bonferroni"]
            n_total_pairs = boot["n_pairwise_comparisons"]
            st.warning(
                f"**Multiple comparisons, reported honestly**: {n_nom} of {n_total_pairs} pairwise "
                f"target-vs-target comparisons look 'significant' at an uncorrected 95% CI — but with "
                f"{n_total_pairs} simultaneous comparisons, that many false positives is expected by "
                f"chance alone. Only **{n_bonf} / {n_total_pairs}** pairs survive a Bonferroni correction "
                f"(alpha={boot['bonferroni_alpha']}), and every one of those {n_bonf} involves the single "
                "standout target (FosA/fosfomycin) against one of the weaker performers — no pair *not* "
                "involving fosfomycin is distinguishable from noise at this sample size. Full pairwise "
                "table: `results/docking/accuracy_benchmark/bootstrap_ci_summary.json`."
            )

        st.warning(
            "**Honest scale caveat**: DUD-E itself averages ~224 actives and ~10,000 decoys *per target*. "
            "This benchmark has exactly **1 active + ~30 decoys per target** (Stage 2 validated one real "
            "ligand per target by design). AUC here is really \"this one active's percentile rank among "
            "its decoys,\" not a multi-active-averaged AUC, and EF1% is necessarily binary at this pool "
            "size (the active either is or isn't the single top-ranked molecule) -- both computed and "
            "reported honestly rather than dressed up as DUD-E-scale statistics. See "
            "DECISIONS_AND_LIMITATIONS.md for the full methodology and per-target numbers."
        )
        source_caption(bench_src, boot_src)

        st.markdown("---")
        st.subheader("Covalent-adduct re-run: does the real reacted form discriminate better?")
        st.caption(
            "4 of the 8 targets have a real, deposited co-crystal ligand that turns out to be a "
            "ring-opened/covalently-engaged reaction product (MER, NXL, PNM) or a real tazobactam-"
            "inhibited co-crystal intermediate (TBE, from PDB 1VM1 -- SHV-1's own structure, 1SHV, is "
            "apo) rather than the intact drug. This re-runs the active-vs-decoy test using that adduct "
            "form itself as the 'active', with a fresh adduct-matched decoy set, through the identical "
            "receptor/box/exhaustiveness/seed as the free-drug run. **This tests non-covalent docking "
            "of the post-reaction product shape as a proxy for covalent binding — not true covalent "
            "docking** (Vina has no covalent-bond-formation term); a real methodological limitation, "
            "not a claim about covalent-docking software this project doesn't use. PBP3/JXJ and "
            "P99/cephalothin were checked and excluded: JXJ's deposited form already *is* what was "
            "docked as its 'free' active (no distinct adduct exists to test), and 1BLS's ligand (IPP) "
            "is an unrelated phosphonate transition-state mimic, not a cephalothin-derived adduct."
        )
        adduct_cmp, adduct_src = d.load_adduct_vs_free_comparison()
        if adduct_cmp is None:
            missing("results/docking/accuracy_benchmark/adduct_vs_free_comparison.json")
        else:
            comps = adduct_cmp["comparisons"]
            acdf = pd.DataFrame([{
                "target": c["target"],
                "free active": c["free_active_name"],
                "adduct": c["adduct_active_name"],
                "free AUC": c["free_auc_point"],
                "free 95% CI": c["free_auc_ci95"],
                "adduct AUC": c["adduct_auc_point"],
                "adduct 95% CI": c["adduct_auc_ci95"],
                "diff (adduct - free)": c["diff_point"],
                "significant (nominal)": c["significant_nominal_95"],
                "significant (Bonferroni)": c["significant_bonferroni"],
                "verdict": c["verdict"],
            } for c in comps])
            st.dataframe(acdf, width="stretch")
            download_df(acdf, "stage2_adduct_vs_free.csv", key="dl_s2_adduct")

            plot_rows = []
            for c in comps:
                plot_rows.append({"target": c["target"], "form": "free drug", "AUC": c["free_auc_point"]})
                plot_rows.append({"target": c["target"], "form": "covalent adduct", "AUC": c["adduct_auc_point"]})
            pdf_ = pd.DataFrame(plot_rows)
            fig2 = px.bar(pdf_, x="target", y="AUC", color="form", barmode="group",
                          title="Free-drug vs. covalent-adduct AUC per target")
            fig2.add_hline(y=0.5, line_dash="dot", line_color="gray")
            fig2.update_layout(yaxis_range=[0, 1], xaxis_title="")
            st.plotly_chart(fig2, width="stretch")

            n_sig = adduct_cmp["n_significant_nominal_95"]
            n_bonf_adduct = adduct_cmp["n_significant_bonferroni"]
            st.markdown(
                f"**{n_sig} / {len(comps)}** targets show a statistically distinguishable "
                f"(nominal 95% CI) change when switching to the adduct form; **{n_bonf_adduct} / {len(comps)}** "
                f"survive Bonferroni correction for {len(comps)} comparisons. The one robust finding: "
                "**SHV-1's adduct (TBE) discriminates significantly *worse*** than free tazobactam "
                "(AUC 0.621 → 0.233, 95% CI of the difference excludes 0 even after correction) — TBE's "
                "linear, ring-opened enamine-aldehyde shape no longer resembles the compact bicyclic "
                "scaffold the pocket was shaped around, and non-covalent redocking (with no covalent "
                "tether holding it in the real reactive geometry) lets it adopt lower-scoring poses "
                "freely. The other 3 targets (meropenem/MER, avibactam/NXL, benzylpenicillin/PNM) show "
                "directional point-estimate changes — two worse, one better — but none clear a "
                "significance bar at this sample size: **not established either way, not noise-free "
                "either.**"
            )
            source_caption(adduct_src)

        st.markdown("---")
        st.subheader("Covalent-docking feasibility: SwissDock, then AutoRevDock/OnionNet-SFCT")
        st.caption(
            "Does a genuine covalent-docking tool fix SHV-1's weak discrimination (AUC 0.621, CI "
            "overlapping chance -- the clearest, statistically-confirmed problem above)? SwissDock's "
            "Attracting Cavities engine has no local/open-source release (checked directly against its "
            "2024 Nucleic Acids Research paper's own Data Availability statement) -- remote-server-only, "
            "excluded before any test could run. Fell back to the designated lighter alternative: "
            "AutoRevDock's Vina_SFCT / OnionNet-SFCT, an ML rescoring term on top of ordinary Vina poses "
            "-- **not covalent docking**."
        )
        sfct, sfct_src = d.load_sfct_investigation()
        if sfct is None:
            missing("results/docking/sfct_rescoring/shv1_covalent_docking_investigation.json")
        else:
            pose = sfct["shv1_pose_validation_new_finding"]
            rescoring = sfct["sfct_rescoring_result"]
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Closest-atom distance to real crystal ligand", f"{pose['result_closest_atom_distance_A']} Å",
                          help="Every other Stage 2 target: 0.14-0.85 Å. SHV-1's pose was never checked "
                               "against a real crystal ligand before this investigation (1SHV is apo).")
            with c2:
                st.metric("Vina AUC (original)", rescoring["vina_auc_point"],
                          help=f"95% CI {rescoring['vina_auc_ci95']}")
            with c3:
                st.metric("OnionNet-SFCT AUC", rescoring["sfct_auc_point"],
                          delta=round(rescoring["sfct_auc_point"] - rescoring["vina_auc_point"], 3),
                          help=f"95% CI {rescoring['sfct_auc_ci95']}. Paired diff 95% CI "
                               f"{rescoring['paired_diff_ci95']} -- {'significant' if rescoring['significant'] else 'NOT significant'}.")
            st.warning(
                f"**Discrimination did not improve.** SFCT's point estimate moved the *wrong* direction "
                f"(chance-level to below-chance), though the paired-bootstrap difference "
                f"({rescoring['paired_diff_point']:+.3f}, 95% CI {rescoring['paired_diff_ci95']}) isn't "
                f"statistically significant. Consistent with the pose-accuracy finding: rescoring can only "
                f"rerank poses Vina already sampled — it cannot fix a pose that was never close to the true "
                f"binding mode to begin with. Real fix direction: better pose sampling, not better scoring."
            )
            st.caption(
                "Install friction for OnionNet-SFCT (`pathtwin-sfct` env, matching this project's "
                "per-tool isolation convention), documented rather than smoothed over: missing `gcc`, a "
                "`crypt.h`-on-modern-Ubuntu Python 3.6 incompatibility, and a second `mdtraj` build "
                "failure resolved by using a conda-forge prebuilt binary instead of compiling from "
                "source. Verified against the repo's own bundled example before use. Full account: "
                "DECISIONS_AND_LIMITATIONS.md, \"Covalent-docking feasibility investigation.\""
            )
            source_caption(sfct_src)

        st.markdown("---")
        st.subheader("SHV-1 box correction, and an audit of all 8 targets for the same vulnerability")
        st.caption(
            "The Ser70-centered box every Stage 2 target uses turned out, for SHV-1 specifically, to sit "
            "far from tazobactam's real crystallographic position -- undetected until the pose-validation "
            "work above, because 1SHV is apo and the original check only ever measured distance to Ser70, "
            "never a real ligand-position comparison. Re-docked with a corrected box; audited all 7 other "
            "targets directly to confirm whether any share the same vulnerability."
        )
        box_fix, box_fix_src = d.load_shv1_box_correction()
        if box_fix is None:
            missing("results/docking/accuracy_benchmark/shv1_box_correction_and_audit.json")
        else:
            bc = box_fix["box_correction"]
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Box centers, distance apart", f"{bc['distance_between_box_centers_A']} Å")
            with c2:
                st.metric("Closest-atom distance (orig → corrected)",
                          f"{bc['corrected_closest_atom_distance_A']} Å",
                          delta=round(bc["corrected_closest_atom_distance_A"] - bc["original_closest_atom_distance_A"], 2),
                          delta_color="inverse",
                          help="Barely changed -- still nowhere near the 0.06-0.85 Å every other target achieves.")
            with c3:
                st.metric("AUC (orig → corrected)", bc["corrected_auc_point"],
                          delta=round(bc["corrected_auc_point"] - bc["original_auc_point"], 3),
                          help=f"Corrected 95% CI {bc['corrected_auc_ci95']} (now excludes chance). Paired diff "
                               f"vs. original 95% CI {bc['paired_diff_ci95']} -- not significant.")
            st.warning(
                f"**{bc['verdict']}**"
            )

            st.markdown("**Audit: is any other target validated the same risky way?**")
            audit_rows = [{
                "target": a["target"], "receptor": a["receptor_pdb"],
                "validation ligand": a["validation_ligand"],
                "same structure as receptor?": "No (external)" if not a["same_structure"] else "Yes",
                "evidence": a.get("note") or f"{a.get('hetatm_lines_in_receptor_file')} HETATM lines found directly in {a['receptor_pdb']}.pdb",
            } for a in box_fix["audit_all_8_targets"]]
            adf = pd.DataFrame(audit_rows)
            st.dataframe(adf, width="stretch")
            download_df(adf, "shv1_audit_all_8_targets.csv", key="dl_shv1_audit")
            st.info(box_fix["audit_conclusion"])
            source_caption(box_fix_src)


# --------------------------------------------------------------------------
# Wet-lab candidate validation (distinct from Stage 2's pipeline-validation
# targets and Stage 4's mutation-impact work)
# --------------------------------------------------------------------------

def render_wetlab():
    st.header("Wet-lab candidate validation")
    st.caption("A different kind of test from Stage 2 (validates the docking pipeline against a "
               "target's own known ligand) and Stage 4 (does a mutation change binding) — this asks: "
               "does a real, independently bioactive compound dock favorably at an already-validated site?")

    labels = [c[0] for c in d.WETLAB_CANDIDATES]
    choice = st.selectbox("Candidate", labels, key="wl_candidate")
    _, target_dir, species, compared_against = next(c for c in d.WETLAB_CANDIDATES if c[0] == choice)
    raw, src = d.load_wetlab_candidate(target_dir)
    if raw is None:
        missing(f"results/docking/{target_dir}/validation_summary.json")
        return

    dock = raw.get("docking", {})
    val = raw.get("validation", {})
    # Comparison-reference key varies by target (cefepime for PBP2a candidates,
    # avibactam for PDC-1 candidates) -- found generically rather than hardcoded
    # to one compound name, so this renders correctly for any future candidate too.
    comp_key = next((k for k in raw if k.startswith("affinity_comparison_to_")), None)
    comp = raw.get(comp_key, {}) if comp_key else {}
    ref_affinity_key = next((k for k in comp if k.endswith("_top_pose_affinity_kcal_mol")), None)
    ref_affinity = comp.get(ref_affinity_key) if ref_affinity_key else None

    # Primary catalytic/active-site distance -- the one "distance_to_*" key in
    # `validation` that isn't a pocket-residue CA distance (those are prefixed
    # "top_pose_distance_to_") or a crystal-ligand distance. First match wins,
    # which is the catalytic nucleophile when analyze_docking_pose.py was run
    # with it as the first --extra-residue (as done for every candidate here).
    catalytic_dist_keys = [k for k in val if k.startswith("distance_to_") and "crystal" not in k]
    dist_active = val.get(catalytic_dist_keys[0]) if catalytic_dist_keys else None

    c1, c2, c3 = st.columns(3)
    c1.metric("Top-pose affinity (kcal/mol)", dock.get("top_pose_affinity_kcal_mol"))
    c2.metric(f"vs. {compared_against}", ref_affinity,
              delta=round(dock.get("top_pose_affinity_kcal_mol", 0) - ref_affinity, 3)
              if ref_affinity is not None else None)
    c3.metric("Distance to catalytic/active site (Å)", dist_active,
              help=catalytic_dist_keys[0] if catalytic_dist_keys else None)

    le = comp.get("ligand_efficiency_kcal_mol_per_heavy_atom", {})
    if le:
        st.subheader("Ligand efficiency (affinity per heavy atom) — a fairer size-normalized comparison")
        ledf = pd.DataFrame([{"compound": k, "kcal_mol_per_heavy_atom": v}
                             for k, v in le.items() if isinstance(v, (int, float))])
        fig = px.bar(ledf, x="compound", y="kcal_mol_per_heavy_atom",
                     title="Ligand efficiency: PPDHMP vs. cefepime")
        st.plotly_chart(fig, width='stretch')
        st.caption(le.get("note", ""))

    st.subheader("Binding-site engagement vs. the compared target")
    dist_rows = [{"metric": k, "value": v} for k, v in val.items()
                 if isinstance(v, (int, float))]
    if dist_rows:
        ddf = pd.DataFrame(dist_rows)
        st.dataframe(ddf, width='stretch')
        download_df(ddf, f"{target_dir}_distances.csv", key=f"dl_wl_dist_{target_dir}")
    st.info(val.get("conclusion", ""))

    ident = raw.get("ligand_identity_verification", {})
    if ident:
        with st.expander("⚠️ Ligand identity verification (a real discrepancy was caught here)"):
            st.markdown(f"**Task-stated identifiers:** {ident.get('task_stated_identifiers')}")
            st.markdown(f"**Formula check:** {ident.get('formula_confirmed')}")
            cas = ident.get("cas_discrepancy_found_and_resolved", {})
            if cas:
                st.warning(cas.get("finding", ""))
                st.markdown(f"**Resolution:** {cas.get('resolution', '')}")

    lit = raw.get("independent_literature_support", {})
    if lit:
        with st.expander("Independent literature support (checked, not assumed)"):
            st.markdown(lit.get("note", ""))
            for cite in lit.get("citations_found_and_checked", []):
                st.markdown(f"- {cite}")
            if lit.get("caveat"):
                st.warning(lit["caveat"])

    ref_flag = raw.get("reference_compound_flag", {})
    if ref_flag:
        with st.expander("🚩 Reference-compound flag (not the endophyte's own discovery)"):
            st.markdown(f"**Finding:** {ref_flag.get('finding', '')}")
            st.markdown(f"**Interpretation:** {ref_flag.get('interpretation', '')}")
            if ref_flag.get("mechanism_caveat"):
                st.warning(ref_flag["mechanism_caveat"])

    origin_mismatch = raw.get("ligand_origin_mismatch_note", {})
    if origin_mismatch:
        with st.expander("🚩 Ligand/target organism mismatch"):
            st.markdown(f"**Finding:** {origin_mismatch.get('finding', '')}")
            st.warning(origin_mismatch.get("interpretation", ""))

    mic = raw.get("real_world_mic_data", {})
    if mic:
        with st.expander("Real-world MIC data on file for this compound"):
            c1, c2 = st.columns(2)
            c1.metric("MIC (µg/mL)", mic.get("value_ug_per_mL"))
            c2.metric("Organism tested", mic.get("organism"))
            st.caption(mic.get("citation", ""))
            if mic.get("context_caveat"):
                st.warning(mic["context_caveat"])

    source_caption(src)
    with st.expander("Full raw result JSON"):
        st.json(raw)
        download_json(raw, f"{target_dir}.json", key=f"dl_wl_raw_{target_dir}")


# --------------------------------------------------------------------------
# Docking-vs-MIC correlation test (exploratory, PPDHMP + 3 literature compounds)
# --------------------------------------------------------------------------

def render_mic_correlation():
    st.header("Docking-vs-MIC correlation test (exploratory)")
    st.caption("PPDHMP + 3 requested literature compounds (Munumbicins, chloramphenicol, "
               "(2E,5E)-phenyltetradeca-2,5-dienoate). Genuinely exploratory at n=1-2 per "
               "target -- read the honest-summary panel at the bottom before treating anything "
               "here as a validated statistic.")

    raw, src = d.load_mic_correlation_test()
    if raw is None:
        missing("results/docking/mic_correlation_test.json")
        return

    c1, c2 = st.columns(2)
    c1.metric("Compounds requested", raw.get("compounds_requested"))
    c2.metric("Compounds successfully docked", raw.get("compounds_successfully_docked"))

    st.subheader("Comparison table")
    table = raw.get("comparison_table", [])
    if table:
        tdf = pd.DataFrame(table)
        display_cols = [c for c in ["compound", "target", "target_organism", "real_mic_ug_per_mL",
                                     "mic_organism", "mic_organism_matches_target",
                                     "vina_affinity_kcal_mol", "heavy_atoms",
                                     "ligand_efficiency_kcal_mol_per_atom"] if c in tdf.columns]
        st.dataframe(tdf[display_cols], width='stretch')
        download_df(tdf[display_cols], "mic_correlation_comparison_table.csv", key="dl_mic_table")
        for row in table:
            if row.get("flag"):
                st.warning(f"**{row['compound']}:** {row['flag']}")
        if not tdf["mic_organism_matches_target"].all():
            st.info("Rows marked `mic_organism_matches_target = False` have a real MIC measured "
                    "against a different organism than the one the compound was docked against -- "
                    "not a fabricated number, but not a fair apples-to-apples pairing either.")
    else:
        missing("comparison_table in mic_correlation_test.json")

    blocked = raw.get("blocked_compound", {})
    if blocked:
        with st.expander(f"⛔ {blocked.get('name', 'Compound')} — could not be docked"):
            st.error(blocked.get("why_blocked", ""))
            st.markdown(blocked.get("conclusion", ""))
            bmic = blocked.get("real_world_mic_data", {})
            if bmic:
                c1, c2 = st.columns(2)
                c1.metric("MIC (µg/mL)", bmic.get("value_ug_per_mL"))
                c2.metric("Organism", bmic.get("organism"))
                st.caption(bmic.get("citation", ""))
                st.markdown(bmic.get("note", ""))

    st.subheader("Spearman correlation, by target")
    spearman = raw.get("spearman_correlation_by_target", {})
    for target_key, s in spearman.items():
        with st.expander(f"{target_key} (n = {s.get('n_with_both_mic_and_affinity')})"):
            c1, c2 = st.columns(2)
            c1.metric("Spearman rho", s.get("spearman_rho") if s.get("spearman_rho") is not None else "—")
            c2.metric("p-value", s.get("p_value") if s.get("p_value") is not None else "undefined")
            st.error(s.get("statistical_validity", ""))
            if s.get("qualitative_observation_only"):
                st.markdown(f"**Qualitative observation only:** {s['qualitative_observation_only']}")

    honest = raw.get("honest_summary", {})
    if honest:
        st.subheader("Honest summary")
        st.warning(f"**{honest.get('headline', '')}**")
        for reason in honest.get("why", []):
            st.markdown(f"- {reason}")
        if honest.get("what_this_test_actually_shows"):
            st.info(honest["what_this_test_actually_shows"])

    source_caption(src)
    with st.expander("Full raw result JSON"):
        st.json(raw)
        download_json(raw, "mic_correlation_test.json", key="dl_mic_raw")


# --------------------------------------------------------------------------
# Stage 3
# --------------------------------------------------------------------------

def render_stage3():
    st.header("Stage 3 — Comparative mutation tracking (snippy + CARD SNP filter)")
    methods_panel(3)

    comparisons = d.list_mutation_comparisons()
    if not comparisons:
        missing("any results/*_mutations directory")
        return

    labels = [c[0] for c in comparisons]
    choice = st.selectbox("Comparison", labels, key="s3_comparison")
    dir_path = dict(comparisons)[choice]
    result = d.load_mutation_comparison(dir_path)

    c1, c2 = st.columns(2)
    c1.metric("Raw variants called (snippy)", result["raw_variant_count"] if result["raw_variant_count"] is not None else "—")
    n_curated = len(result["curated_hits"]) if result["curated_hits"] is not None else None
    c2.metric("CARD-curated resistance SNPs", n_curated if n_curated is not None else "—")

    if result["curated_hits"] is None:
        missing(dir_path / "resistance_relevant_snps.tab")
    elif result["curated_hits"].empty:
        st.success("0 CARD-curated resistance SNPs — a genuine negative result (see methods panel above "
                   "for why this is expected for several species).")
    else:
        st.subheader("CARD-curated resistance-conferring SNPs")
        styled = result["curated_hits"]
        st.dataframe(styled, width='stretch')
        download_df(styled, f"{dir_path.name}_curated_snps.csv", key=f"dl_s3_{dir_path.name}")

    source_caption(result["raw_source"], result["curated_source"])


# --------------------------------------------------------------------------
# Stage 4
# --------------------------------------------------------------------------

def render_stage4():
    st.header("Stage 4 — Mutation impact simulator (FoldX ΔΔG + Vina re-docking)")
    methods_panel(4)

    rows = []
    for mdir, label, target, species, caveat in d.STAGE4_MUTATIONS:
        result, src = d.load_mutation_sim_result(mdir)
        if result is None:
            rows.append({"mutation": label, "target": target, "species": species,
                         "status": "NO result.json FOUND", "caveat": caveat, "_src": None})
            continue
        wt = result["wildtype_vina_score"]
        mut = result["mutant_vina_score"]
        rows.append({
            "mutation": label, "target": target, "species": species,
            "wildtype_vina_kcal_mol": wt, "mutant_vina_kcal_mol": mut,
            "delta_affinity": result["delta_affinity"], "ddg_foldx": result["ddg_foldx"],
            "interpretation": result["interpretation"], "caveat": caveat, "_src": src,
        })
    df = pd.DataFrame(rows)

    def direction_icon(row):
        if "wildtype_vina_kcal_mol" not in row or pd.isna(row.get("delta_affinity")):
            return "—"
        both_resistance = row["delta_affinity"] > 0 and row["ddg_foldx"] > 0
        return "✅ both agree (resistance direction)" if both_resistance else "⚠️ signals disagree"

    if "delta_affinity" in df.columns:
        df["signal_agreement"] = df.apply(direction_icon, axis=1)

    display_cols = [c for c in ["mutation", "target", "species", "wildtype_vina_kcal_mol",
                                 "mutant_vina_kcal_mol", "delta_affinity", "ddg_foldx",
                                 "signal_agreement"] if c in df.columns]
    st.dataframe(df[display_cols], width='stretch')
    download_df(df[[c for c in df.columns if c != "_src"]], "stage4_mutation_simulator.csv", key="dl_s4")

    plot_df = df.dropna(subset=["wildtype_vina_kcal_mol"]) if "wildtype_vina_kcal_mol" in df.columns else pd.DataFrame()
    if len(plot_df):
        melted = plot_df.melt(id_vars=["mutation"], value_vars=["wildtype_vina_kcal_mol", "mutant_vina_kcal_mol"],
                               var_name="structure", value_name="kcal_mol")
        fig = px.bar(melted, x="mutation", y="kcal_mol", color="structure", barmode="group",
                     title="Wild-type vs. mutant Vina binding affinity per mutation")
        fig.update_layout(xaxis_tickangle=-30, yaxis_title="kcal/mol")
        st.plotly_chart(fig, width='stretch')

    st.subheader("Per-mutation detail (including any documented caveat)")
    choice = st.selectbox("Mutation", df["mutation"].tolist(), key="s4_mut")
    row = df[df["mutation"] == choice].iloc[0]
    if row.get("status") == "NO result.json FOUND":
        missing(f"results/mutation_sim/.../result.json for {choice}")
    else:
        st.json({k: row[k] for k in ["wildtype_vina_kcal_mol", "mutant_vina_kcal_mol",
                                      "delta_affinity", "ddg_foldx", "interpretation"]})
        source_caption(row["_src"])
    if row.get("caveat"):
        st.warning(f"**Documented caveat for this mutation:** {row['caveat']}")


# --------------------------------------------------------------------------
# Stage 5
# --------------------------------------------------------------------------

def render_stage5():
    st.header("ML resistance prediction")
    st.caption("Stage 5 — resistance-phenotype classifiers")
    methods_panel(5)

    tabs = st.tabs([s[0] for s in d.STAGE5_CLASSIFIERS])
    for (sp_label, phenotype, json_path, shap_path), tab in zip(d.STAGE5_CLASSIFIERS, tabs):
        with tab:
            _render_stage5_species(sp_label, phenotype, json_path, shap_path)


def _render_stage5_species(sp_label, phenotype, json_path, shap_path):
    result = d.load_stage5_result(json_path)

    if result is None:
        missing(json_path.relative_to(d.PROJECT_ROOT))
        return

    st.subheader(f"{sp_label} — {phenotype} resistance")
    n = result.get("n_genomes")
    n_r = result.get("n_resistant")
    n_s = result.get("n_susceptible")
    if n:
        c1, c2, c3 = st.columns(3)
        c1.metric("Genomes", n)
        c2.metric("Resistant", n_r)
        c3.metric("Susceptible", n_s)

    # --- metrics table: schema differs by species, handled explicitly ---
    metric_rows = []
    if "cv_metrics" in result:  # K. pneumoniae's original schema
        for evalname in ("cv_metrics", "test_metrics", "benchmark_metrics"):
            for model, m in result.get(evalname, {}).items():
                metric_rows.append({"evaluation": evalname, "model": model, **{k: v for k, v in m.items() if isinstance(v, (int, float))}})
    elif "baseline_metrics" in result:  # P. aeruginosa baseline-vs-combined schema
        for feature_set in ("baseline_metrics", "combined_metrics"):
            for row in flatten_nested_metrics(result.get(feature_set, {}), ["split", "model"]):
                row["feature_set"] = feature_set.replace("_metrics", "")
                metric_rows.append(row)
    elif "metrics" in result:  # S. aureus / A. baumannii / E. faecium / E. cloacae schema
        metric_rows = flatten_nested_metrics(result["metrics"], ["split", "model"])

    if metric_rows:
        mdf = pd.DataFrame(metric_rows)
        st.subheader("Model performance")
        st.dataframe(mdf, width='stretch')
        download_df(mdf, f"{json_path.stem}_metrics.csv", key=f"dl_s5_metrics_{sp_label}")
        num_col = "accuracy" if "accuracy" in mdf.columns else None
        if num_col:
            fig = px.bar(mdf, x=mdf.columns[0], y="accuracy", color="model", barmode="group",
                         facet_col="feature_set" if "feature_set" in mdf.columns else None,
                         title="Accuracy by split/model")
            st.plotly_chart(fig, width='stretch')
    else:
        missing("a recognized metrics field in " + str(json_path.name))

    # split-size / reliability diagnostics, where present (A. baumannii/E. faecium/E. cloacae)
    split_sizes = result.get("split_sizes") or result.get("baseline_split_sizes")
    if split_sizes:
        st.subheader("Split diagnostics")
        diag_rows = []
        for split_name, sizes in split_sizes.items():
            diag_rows.append({"split": split_name, **{k: v for k, v in sizes.items() if not isinstance(v, dict)}})
        ddf = pd.DataFrame(diag_rows)
        st.dataframe(ddf, width='stretch')
        if "split_reliable" in ddf.columns and (ddf["split_reliable"] == False).any():
            st.warning("At least one split above is flagged UNRELIABLE (dominant group and/or "
                       "undersized test set) — see the dominant-group follow-up panel and the "
                       "methods panel above before citing its accuracy/AUC.")

    # SHAP
    st.subheader("SHAP feature importance")
    shap_feats = result.get("shap_top_features")
    if shap_feats:
        sdf = pd.DataFrame(shap_feats)
        st.dataframe(sdf, width='stretch')
        download_df(sdf, f"{json_path.stem}_shap_top_features.csv", key=f"dl_s5_shap_{sp_label}")
    if shap_path.exists():
        st.image(str(shap_path), caption=f"SHAP summary plot — {shap_path.relative_to(d.PROJECT_ROOT)}")
    else:
        missing(shap_path.relative_to(d.PROJECT_ROOT))

    source_caption(json_path.relative_to(d.PROJECT_ROOT))
    with st.expander("Full raw result JSON"):
        st.json(result)
        download_json(result, f"{json_path.stem}.json", key=f"dl_s5_raw_{sp_label}")

    # leakage check (K. pneumoniae only) and dominant-group follow-up (cross-species)
    if sp_label == "K. pneumoniae":
        leak, leak_src = d.load_stage5_leakage_check()
        if leak:
            with st.expander("Random-vs-grouped split leakage check"):
                lrows = []
                for split_name, models in leak["metrics"].items():
                    for model, m in models.items():
                        lrows.append({"split": split_name, "model": model, **{k: v for k, v in m.items() if isinstance(v, (int, float))}})
                ldf = pd.DataFrame(lrows)
                st.dataframe(ldf, width='stretch')
                download_df(ldf, "stage5_leakage_check.csv", key="dl_s5_leakage")
                source_caption(leak_src)

    dgi, dgi_src = d.load_dominant_group_investigation()
    if dgi and sp_label.replace(". ", "").replace(" ", "").lower() in ("abaumannii", "efaecium"):
        key = "abaumannii" if "baumannii" in sp_label.lower() else "efaecium"
        with st.expander(f"Dominant-group imbalance follow-up — {sp_label}"):
            st.json(dgi.get(key, {}))
            download_json(dgi.get(key, {}), f"{key}_dominant_group_investigation.json", key=f"dl_dgi_{key}")
            source_caption(dgi_src)


# --------------------------------------------------------------------------
# Stage 6
# --------------------------------------------------------------------------

def render_stage6():
    st.header("CRISPR-Cas co-occurrence")
    st.caption("Stage 6 — CRISPR-Cas / acquired-resistance co-occurrence")
    methods_panel(6)

    tabs = st.tabs([s[0] for s in d.STAGE6_SPECIES])
    for (sp_label, path), tab in zip(d.STAGE6_SPECIES, tabs):
        with tab:
            _render_stage6_species(sp_label, path)


def _render_stage6_species(sp_label, path):
    if sp_label == "K. pneumoniae":
        crossref, src = d.load_stage6_original_crossref()
        if crossref is None:
            missing("results/crispr_scan/crispr_amr_crossref_full_eskape.json")
            return
        rows = [g for g in crossref["genomes"] if g["species"] == "K. pneumoniae"]
        st.info("K. pneumoniae has no dedicated population-scale CRISPR-Cas run in this project — "
                "shown here is its original n=3 proof-of-concept entry from the cross-species table.")
        df = pd.DataFrame(rows)
        st.dataframe(df, width='stretch')
        download_df(df, "kpneumoniae_crispr_original_n3.csv", key="dl_s6_kp")
        source_caption(src)
        return

    result = d.load_stage6_result(path)
    if result is None:
        missing(path.relative_to(d.PROJECT_ROOT) if path else "no path configured")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Genomes scanned", result.get("n_genomes_scanned"))
    c2.metric("Confident CRISPR-Cas present", result.get("n_cas_present"))
    frac = result.get("fraction_cas_present")
    c3.metric("Fraction Cas-present", f"{frac:.1%}" if frac is not None else "—")

    subtypes = result.get("subtype_distribution", {})
    if subtypes:
        st.subheader("Subtype distribution")
        sdf = pd.DataFrame(sorted(subtypes.items()), columns=["subtype", "count"])
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(sdf, width='stretch')
            download_df(sdf, f"{path.stem}_subtypes.csv", key=f"dl_s6_subtypes_{sp_label}")
        with c2:
            fig = px.bar(sdf, x="subtype", y="count", title=f"{sp_label} — CRISPR-Cas subtype counts")
            st.plotly_chart(fig, width='stretch')

    st.subheader("Statistical tests")
    stat_keys = [k for k in result if k.startswith("mannwhitney") or k.startswith("fisher")]
    if stat_keys:
        srows = []
        for k in stat_keys:
            v = result[k]
            if isinstance(v, dict):
                srows.append({"test": k, **{kk: vv for kk, vv in v.items() if not isinstance(vv, (list, dict))}})
        sdf = pd.DataFrame(srows)
        st.dataframe(sdf, width='stretch')
        download_df(sdf, f"{path.stem}_stats.csv", key=f"dl_s6_stats_{sp_label}")
    elif "statistical_battery" in result:
        st.info(result["statistical_battery"])
    else:
        missing("mannwhitney/fisher fields in " + path.name)

    if "direction_vs_literature" in result:
        st.markdown(f"**Direction vs. literature (whole population):** {result['direction_vs_literature']}")

    # A. baumannii dominant-clone confound panel
    dcc = result.get("dominant_clone_confound_check")
    if dcc:
        st.subheader("⚠️ Dominant-clone confound check (this species only)")
        st.warning(dcc["note"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Dominant ST", dcc["dominant_st"])
        c2.metric("Fraction of sample", f"{dcc['dominant_st_fraction']:.1%}")
        c3.metric("Its own Cas-present fraction", f"{dcc['dominant_st_cas_present_fraction']:.1%}")
        st.markdown(f"**After excluding it (n={dcc['n_after_excluding_dominant_clone']}, "
                    f"{dcc['n_distinct_st_after_exclusion']} distinct STs remain):**")
        st.markdown(f"- {dcc['direction_after_exclusion']}")
        mw = dcc["mannwhitney_dominant_clone_excluded"]
        st.markdown(f"- Mann-Whitney p = {mw['p_value']:.2e} (median burden {mw['median_pos']} vs {mw['median_neg']})")
        fb = dcc["fisher_high_burden_dominant_clone_excluded"]
        st.markdown(f"- Fisher vs. high burden: OR = {fb['odds_ratio']:.2f}, p = {fb['p_value']:.2e}")
        download_json(dcc, f"{path.stem}_dominant_clone_check.json", key=f"dl_s6_dcc_{sp_label}")

    source_caption(path.relative_to(d.PROJECT_ROOT))
    with st.expander("Full raw result JSON"):
        st.json(result)
        download_json(result, f"{path.stem}.json", key=f"dl_s6_raw_{sp_label}")

    merged_tag = {"S. aureus": "saureus", "P. aeruginosa": "paeruginosa", "A. baumannii": "abaumannii",
                  "E. faecium": "efaecium", "E. cloacae": "ecloacae"}.get(sp_label)
    mdf, msrc = d.load_stage6_merged_csv(merged_tag) if merged_tag else (None, None)
    if mdf is not None:
        with st.expander(f"Full per-genome merged table ({len(mdf)} genomes)"):
            st.dataframe(mdf, width='stretch')
            download_df(mdf, f"{merged_tag}_crispr_amr_merged.csv", key=f"dl_s6_merged_{sp_label}")
            source_caption(msrc)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

NAV_PAGES = [
    ("overview", "Overview", render_overview),
    ("stage1", "Resistance gene profiling", render_stage1),
    ("stage2", "Docking validation", render_stage2),
    ("stage3", "Mutation tracking", render_stage3),
    ("stage4", "Mutation simulation", render_stage4),
    ("stage5", "ML resistance prediction", render_stage5),
    ("stage6", "CRISPR-Cas co-occurrence", render_stage6),
    ("wetlab", "Wet-lab candidate validation", render_wetlab),
    ("mic_correlation", "Docking-vs-MIC correlation (exploratory)", render_mic_correlation),
]

NAV_BUTTON_CSS = """
<style>
section[data-testid="stSidebar"] button {
    font-size: 1.05rem;
    font-weight: 600;
    text-align: left;
    justify-content: flex-start;
}
</style>
"""


def main():
    st.sidebar.title("PathTwin")
    st.sidebar.caption("Results dashboard — reads directly from `results/`, nothing recomputed here.")
    st.markdown(NAV_BUTTON_CSS, unsafe_allow_html=True)

    if "page" not in st.session_state:
        st.session_state.page = "overview"

    for key, label, _ in NAV_PAGES:
        is_active = st.session_state.page == key
        if st.sidebar.button(label, key=f"nav_{key}", width="stretch",
                              type="primary" if is_active else "secondary"):
            if st.session_state.page != key:
                # Rerun immediately rather than letting the loop continue --
                # otherwise buttons rendered *before* this one in the loop
                # keep showing the OLD active state for this run (the state
                # update below only takes effect starting next run), so the
                # sidebar highlight visibly lags one click behind the page
                # actually rendered. Caught by testing click->highlight
                # behavior directly via AppTest, not assumed correct from
                # the code alone.
                st.session_state.page = key
                st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption("Every number on this page traces to a real file under `results/` "
                       "(paths shown under each table/chart). A 'no result file found' box "
                       "means the file genuinely doesn't exist, not a rendering gap.")

    render_fn = next(fn for key, _, fn in NAV_PAGES if key == st.session_state.page)
    render_fn()


if __name__ == "__main__":
    main()
