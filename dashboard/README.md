# PathTwin Results Dashboard

A read-only Streamlit dashboard presenting validated results across all 6
project stages and all 6 ESKAPE pathogens.

## Design principle

**Nothing here is computed by the dashboard.** Every table, chart, and
metric is parsed directly from a file already on disk under `results/`
(or `data/processed/` for two Stage 5 metadata files) at the moment the
page loads. If a file a section needs doesn't exist for the selected
stage/species combination, the section says so explicitly (a "no result
file found" box naming the exact path it looked for) rather than being
silently omitted or filled with a placeholder. Every table has a
`Download CSV`/`Download JSON` button so a reviewer can pull the exact
numbers shown, and a "Source: `path`" caption underneath naming the file
they came from.

## Running it

```bash
mamba activate pathtwin   # streamlit/plotly are installed into this env
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501` by default.

## Requirements

See `requirements.txt` — `streamlit`, `plotly`, `pandas` (already present
in the `pathtwin` conda env; `pip install -r dashboard/requirements.txt`
if running elsewhere).

## Structure

- `app.py` — the Streamlit app itself. A button-based sidebar nav (large
  descriptive labels — "Resistance gene profiling", not "Stage 1" — with
  the stage number as small secondary text, and a clear
  primary/secondary active-page highlight) opens on an **Overview**
  landing page (a file-verified 6×6 species×stage coverage matrix plus
  four headline validated results) by default. Stages 1, 5, and 6 show
  all 6 ESKAPE species as `st.tabs()` at once rather than behind a
  dropdown. One `render_stageN()` function per stage/page.
- `data.py` — all file-reading logic, kept separate from rendering so
  every loader can return `None` (or an explicit "not found" signal)
  instead of raising, letting the app layer decide how to display a
  genuine gap. Also has `compute_coverage_matrix()` for the Overview
  page — every cell is a real existence check against a file on disk,
  not a hardcoded assumption.
- `methods.py` — short, hand-condensed methods/limitations notes per
  stage. These are **condensed summaries, not an automated extraction**
  of `DECISIONS_AND_LIMITATIONS.md` — each bullet names the exact section
  of that file it summarizes, so a viewer who wants the full account
  (not just what fit in a bullet point) knows exactly where to look.

## Known display limitations (real ones, not bugs)

- **Stage 2, SHV-1/tazobactam**: the only docking target with no
  structured `validation_summary.json` (it predates that convention).
  Shown with its top-pose affinity parsed directly from the raw Vina
  log; distance-to-catalytic-residue is not machine-readable for this
  one target and is documented in prose in README instead.
- **Stage 4, P99/AmpC L293P**: only the corrected (cefepime) result has
  a surviving `result.json` on disk — the original, ambiguous cephalothin
  run's numbers were overwritten and exist only in
  `DECISIONS_AND_LIMITATIONS.md`/README prose. The dashboard shows only
  the file-backed cefepime result, with that history noted as a caveat.
- **Stage 6, K. pneumoniae**: has no dedicated population-scale CRISPR-Cas
  run (unlike the other 5 species) — shown from its original n=3
  proof-of-concept entry in the cross-species crossref table instead.
