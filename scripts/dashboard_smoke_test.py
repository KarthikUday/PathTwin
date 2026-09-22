#!/usr/bin/env python3
"""
Headless end-to-end smoke test for dashboard/app.py using Streamlit's
official AppTest framework (streamlit.testing.v1) -- actually executes the
app's Python code for every page and every selector value, rather than
just reading the source. Reports:
  - any uncaught exception per (page, selector-value)
  - every warning/error message rendered, flagged as EXPECTED (a known,
    documented honest-limitation caveat) or UNEXPECTED (a "no result file
    found" / missing-data box, or anything else not accounted for)

Run from the project root: python3 scripts/dashboard_smoke_test.py
"""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = PROJECT_ROOT / "dashboard" / "app.py"

# Substrings that mark a warning/error as a KNOWN, already-documented honest
# limitation rather than an unexpected gap. Hand-curated from README /
# DECISIONS_AND_LIMITATIONS.md -- anything NOT matching one of these is
# reported as UNEXPECTED for manual review.
EXPECTED_WARNING_SUBSTRINGS = [
    "UNRELIABLE", "dominant-group", "dominant clone", "dominant_st",
    "not read as", "not asserted", "flagged as", "mismatch", "MISMATCH",
    "off-target", "reference/positive-control", "reference compound",
    "organism mismatch", "not mechanistically", "uninformative",
    "does not involve", "no van cluster", "no acquired carbapenemase",
    "not statistical", "NOT A REAL STATISTIC", "NOT COMPUTABLE",
    "mathematically", "genuinely exploratory", "no correlation could be",
    "not validated", "high/relatively poor", "no MIC against",
    "isolated FROM", "produced BY", "less complete", "weaker", "confound",
    "Honest scale caveat", "Discrimination did not improve",
    "No result file found",  # the generic `missing()` box itself; classified
                              # separately below by page/selector context
]


def classify(msg: str, page: str, selector: str) -> str:
    if "No result file found" in msg:
        return "MISSING_FILE"
    for s in EXPECTED_WARNING_SUBSTRINGS:
        if s.lower() in msg.lower():
            return "EXPECTED"
    return "UNEXPECTED"


def get_selector_values(at, key):
    for sb in at.selectbox:
        if sb.key == key:
            return sb.options
    return None


def run_page(page_key, page_label, selector_key=None):
    results = []
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.session_state["page"] = page_key
    at.run()
    _collect(at, page_label, "<default>", results)

    if selector_key:
        options = get_selector_values(at, selector_key)
        if options is None:
            results.append((page_label, "<selector>", "ERROR",
                             f"selectbox key={selector_key} not found on page"))
            return results
        for opt in options:
            at2 = AppTest.from_file(str(APP_PATH), default_timeout=60)
            at2.session_state["page"] = page_key
            at2.run()
            sb = next(s for s in at2.selectbox if s.key == selector_key)
            sb.select(opt).run()
            _collect(at2, page_label, opt, results)
    return results


def _collect(at, page_label, selector_val, results):
    if at.exception:
        for exc in at.exception:
            results.append((page_label, selector_val, "EXCEPTION", str(exc.value) + "\n" + exc.stack_trace))
    for w in at.warning:
        results.append((page_label, selector_val, "warning", w.value))
    for e in at.error:
        results.append((page_label, selector_val, "error", e.value))


def main():
    pages = [
        ("overview", "Overview", None),
        ("stage1", "Stage 1", None),          # species via st.tabs -- all render every run
        ("stage2", "Stage 2", "s2_target"),
        ("stage3", "Stage 3", "s3_comparison"),
        ("stage4", "Stage 4", "s4_mut"),
        ("stage5", "Stage 5", None),           # species via st.tabs -- all render every run
        ("stage6", "Stage 6", None),           # species via st.tabs -- all render every run
        ("wetlab", "Wet-lab candidate validation", "wl_candidate"),
        ("mic_correlation", "Docking-vs-MIC correlation", None),
    ]

    all_results = []
    for key, label, sel in pages:
        print(f"=== {label} ===", flush=True)
        try:
            res = run_page(key, label, sel)
        except Exception as e:
            res = [(label, "<page load>", "FATAL", repr(e))]
        for r in res:
            print("  ", r[0], "|", r[1], "|", r[2], "|", r[3][:200].replace("\n", " "))
        all_results.extend(res)

    print("\n\n===================== SUMMARY =====================")
    exceptions = [r for r in all_results if r[2] in ("EXCEPTION", "FATAL", "ERROR")]
    missing = [r for r in all_results if r[2] == "warning" and "No result file found" in r[3]]
    other_warnings = [r for r in all_results if r[2] == "warning" and "No result file found" not in r[3]]
    errors_rendered = [r for r in all_results if r[2] == "error"]

    print(f"Total (page, selector) combinations exercised: {sum(1 for _ in all_results)}")
    print(f"Uncaught exceptions / fatal errors: {len(exceptions)}")
    for r in exceptions:
        print("  EXCEPTION:", r[0], "|", r[1], "|", r[3][:500])
    print(f"\n'No result file found' boxes rendered: {len(missing)}")
    for r in missing:
        cls = classify(r[3], r[0], r[1])
        print(f"  [{cls}] {r[0]} | {r[1]} | {r[3][:200]}")
    print(f"\nst.error() boxes rendered: {len(errors_rendered)}")
    for r in errors_rendered:
        print("  ERROR:", r[0], "|", r[1], "|", r[3][:200])
    print(f"\nOther st.warning() boxes rendered: {len(other_warnings)}")
    unexpected = [r for r in other_warnings if classify(r[3], r[0], r[1]) == "UNEXPECTED"]
    print(f"  of which classified UNEXPECTED: {len(unexpected)}")
    for r in unexpected:
        print("  UNEXPECTED WARNING:", r[0], "|", r[1], "|", r[3][:300])

    sys.exit(1 if (exceptions or unexpected) else 0)


if __name__ == "__main__":
    main()
