#!/usr/bin/env python3
"""
Sinhala Q&A benchmark - per-model result builder.

Scans a folder of annotation sheets (one CSV per model) and writes:

    <output-dir>/<model>/<model>_results.csv   one row per metric
    <output-dir>/all_models_results.csv        one row per model

Each subset is scored only with the metrics that are valid for it:

    short_answer   ->  exact_match, token_f1, chrf, chrf3
    unanswerable   ->  squad_v2_noans_em

chrF / EM / token-F1 are NOT reported for unanswerable items: there both the
reference and the output are the literal string "UNANSWERABLE", so those
metrics score that string against itself and mean nothing.

Drop new model sheets into the folder and re-run; nothing else needs changing.

Usage:
    python evaluate_models.py
    python evaluate_models.py --input-dir . --output-dir model_results
    python evaluate_models.py --bootstrap 0        # skip confidence intervals
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["id", "reference_answer_sin", "model_output_sin",
                    "qa_subtype", "hallucination_type"]

# Metric columns, and the subset each one is valid for.
SHORT_ANSWER_METRICS = [("exact_match", "Exact Match (%)"),
                        ("token_f1", "Token F1 (%)"),
                        ("chrf", "chrF"),
                        ("chrf3", "chrF3")]
UNANSWERABLE_METRIC = ("squad_v2_noans_em", "No-answer EM (%)")


BINARY_METRICS = {"exact_match", "squad_v2_noans_em"}

ST_SHORT = "short_answer"
ST_UNANS = "unanswerable"

LBL_SUPPORTED = "Supported"
LBL_UNSUPPORTED = "Unsupported"
LBL_CORRECT_REFUSAL = "Correct Refusal"
LBL_FALSE_REFUSAL = "False Refusal"


LBL_DEGENERATE = "Degenerated Repetition Output"
HUMAN_LABELS = [LBL_SUPPORTED, LBL_UNSUPPORTED, LBL_CORRECT_REFUSAL,
                LBL_FALSE_REFUSAL, LBL_DEGENERATE]

# Spelling variants seen in the sheets, mapped to the canonical label.
LABEL_ALIASES = {"Degenerated Repetitive Output": LBL_DEGENERATE}

# The no-answer token, plus misspellings seen in the sheets.
NOANS_TOKEN = "UNANSWERABLE"
NOANS_TOKEN_VARIANTS = {"UNANSWERABLE", "UNASWERABLE"}
RNG_SEED = 20250830


def read_sheet(path: Path) -> pd.DataFrame:
    """Read one annotation sheet, tolerating BOM / encoding variation."""
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as err:      # pragma: no cover - fallback path
            last_err = err
    raise RuntimeError(f"could not decode file: {last_err}")


def model_name_from(df: pd.DataFrame, path: Path) -> str:
    """Prefer the model_name column; fall back to the trailing part of the filename."""
    if "model_name" in df.columns:
        names = df["model_name"].dropna().astype(str).str.strip()
        names = names[names != ""]
        if not names.empty:
            if names.nunique() > 1:
                print(f"    ! multiple model_name values {sorted(names.unique())}"
                      f" - using the most frequent")
            return names.mode().iat[0]
    stem = path.stem
    return (stem.rsplit(" - ", 1)[-1] if " - " in stem else stem).strip()


def safe_dirname(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "unknown_model"


def to_percent(series: pd.Series) -> pd.Series:
    """Put a continuous metric column on a 0-100 scale (0-1 columns are scaled up)."""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().any() and s.max(skipna=True) <= 1.0:
        s = s * 100.0
    return s


def binary_to_percent(series: pd.Series) -> pd.Series:

    s = pd.to_numeric(series, errors="coerce")
    return s.map({0.0: 0.0, 1.0: 100.0, 100.0: 100.0})


def scale_warnings(df: pd.DataFrame) -> list[str]:
    """Encoding problems spotted while scaling, reported on the console."""
    out = []
    for col in BINARY_METRICS:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        odd = s[~s.isin([0.0, 1.0, 100.0])]
        if len(odd):
            out.append(f"{col}: {len(odd)} value(s) outside {{0, 1, 100}} "
                       f"are not pass/fail scores and were dropped "
                       f"(e.g. {', '.join(f'{v:g}' for v in odd.unique()[:3])})")
        if s.isin([1.0]).any() and s.isin([100.0]).any():
            out.append(f"{col}: mixed 0/1 and 0/100 encodings in one column "
                       f"({int(s.eq(1).sum())} as 1, {int(s.eq(100).sum())} as 100) "
                       f"- both read as correct")

    for col, _ in SHORT_ANSWER_METRICS:
        if col in BINARY_METRICS or col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) and s.max() > 1.0 and (s <= 1.0).mean() > 0.5:
            out.append(f"{col}: ambiguous scale - most values are <=1 but the "
                       f"column reaches {s.max():g}; check the source sheet")
    return out


def is_noans_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper().isin(NOANS_TOKEN_VARIANTS)


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required column(s): {missing}")

    for col in ("qa_subtype", "hallucination_type"):
        df[col] = df[col].fillna("").astype(str).str.strip()

    warnings = scale_warnings(df)

    for variant, canonical in LABEL_ALIASES.items():
        hits = int(df["hallucination_type"].eq(variant).sum())
        if hits:
            warnings.append(f'hallucination_type: {hits} item(s) spelled '
                            f'"{variant}" normalised to "{canonical}"')
            df["hallucination_type"] = df["hallucination_type"].replace(variant, canonical)

    misspelled = df["reference_answer_sin"].fillna("").astype(str).str.strip().str.upper()
    odd = int(misspelled.isin(NOANS_TOKEN_VARIANTS - {NOANS_TOKEN}).sum())
    if odd:
        warnings.append(f'reference_answer_sin: {odd} item(s) misspell the no-answer '
                        f'token (expected "{NOANS_TOKEN}") - accepted, fix the sheet')


    misplaced = (df["qa_subtype"].eq(ST_SHORT)
                 & df["hallucination_type"].eq(LBL_CORRECT_REFUSAL)
                 & is_noans_text(df["model_output_sin"]))
    if misplaced.any():
        ids = df.loc[misplaced, "id"].tolist()
        warnings.append(f'hallucination_type: answerable Correct Refusal item(s) '
                        f'{ids} normalised to False Refusal')
        df.loc[misplaced, "hallucination_type"] = LBL_FALSE_REFUSAL

    for col, _ in SHORT_ANSWER_METRICS + [UNANSWERABLE_METRIC]:
        if col not in df.columns:
            df[col] = np.nan
        elif col in BINARY_METRICS:
            df[col] = binary_to_percent(df[col])
        else:
            df[col] = to_percent(df[col])

    df.attrs["scale_warnings"] = warnings
    return df



def bootstrap_ci(values: np.ndarray, n_boot: int) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of the mean."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if n_boot <= 0 or len(values) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(RNG_SEED)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def build_results(df: pd.DataFrame, model: str, n_boot: int) -> pd.DataFrame:
    """The 13-row per-model result table."""
    short = df[df["qa_subtype"].eq(ST_SHORT)]
    unans = df[df["qa_subtype"].eq(ST_UNANS)]
    rows: list[dict] = []

    def add(metric: str, subset: str, value: float, n: int,
            ci: tuple[float, float] = (np.nan, np.nan)) -> None:
        lo, hi = ci
        rows.append({
            "model": model,
            "metric": metric,
            "subset": subset,
            "value": round(value, 2) if not np.isnan(value) else "",
            "n": n,
            "ci95_low": round(lo, 2) if not np.isnan(lo) else "",
            "ci95_high": round(hi, 2) if not np.isnan(hi) else "",
        })

    def add_mean(metric: str, subset: str, series: pd.Series) -> None:
        """Mean of a metric column, over the rows that actually carry a score."""
        s = pd.to_numeric(series, errors="coerce").dropna()
        value = float(s.mean()) if len(s) else np.nan
        add(metric, subset, value, len(s), bootstrap_ci(s.to_numpy(), n_boot))

    def add_rate(metric: str, subset: str, mask: pd.Series) -> None:
        """Percentage of rows in `mask`'s frame satisfying it, with a CI."""
        hits = mask.astype(float) * 100.0
        value = float(hits.mean()) if len(hits) else np.nan
        add(metric, subset, value, len(hits), bootstrap_ci(hits.to_numpy(), n_boot))

    # --- automatic metrics, each on its own subset -------------------------
    for col, label in SHORT_ANSWER_METRICS:
        add_mean(label, ST_SHORT, short[col])

    noans_col, noans_label = UNANSWERABLE_METRIC
    add_mean(noans_label, ST_UNANS, unans[noans_col])

    # --- human annotation counts ------------------------------------------
    for label, frame in ((LBL_SUPPORTED, short), (LBL_UNSUPPORTED, short),
                         (LBL_FALSE_REFUSAL, short), (LBL_CORRECT_REFUSAL, unans)):
        subset = ST_SHORT if frame is short else ST_UNANS
        add(f"{label} (n)", subset,
            float(frame["hallucination_type"].eq(label).sum()), len(frame))

    # Degenerate output can occur on either subset, so it is counted over all items.
    add(f"{LBL_DEGENERATE} (n)", "overall",
        float(df["hallucination_type"].eq(LBL_DEGENERATE).sum()), len(df))

    # --- human annotation rates -------------------------------------------
    add_rate("Hallucination rate (%)", ST_SHORT,
             short["hallucination_type"].eq(LBL_UNSUPPORTED))
    add_rate("False refusal rate (%)", ST_SHORT,
             short["hallucination_type"].eq(LBL_FALSE_REFUSAL))
    add_rate("Correct refusal rate (%)", ST_UNANS,
             unans["hallucination_type"].eq(LBL_CORRECT_REFUSAL))
    add_rate("Degenerate output rate (%)", "overall",
             df["hallucination_type"].eq(LBL_DEGENERATE))
    add_rate("Human accuracy (%)", "overall",
             df["hallucination_type"].isin([LBL_SUPPORTED, LBL_CORRECT_REFUSAL]))

    return pd.DataFrame(rows)


def headline_row(results: pd.DataFrame, df: pd.DataFrame, model: str) -> dict:
    """One flat row per model for the all-models sheet."""
    def get(metric: str) -> float:
        hit = results.loc[results["metric"].eq(metric), "value"]
        return hit.iat[0] if len(hit) and hit.iat[0] != "" else np.nan

    return {
        "model": model,
        "n_items": len(df),
        "n_short_answer": int(df["qa_subtype"].eq(ST_SHORT).sum()),
        "n_unanswerable": int(df["qa_subtype"].eq(ST_UNANS).sum()),
        "exact_match": get("Exact Match (%)"),
        "token_f1": get("Token F1 (%)"),
        "chrf": get("chrF"),
        "chrf3": get("chrF3"),
        "noans_em": get("No-answer EM (%)"),
        "hallucination_rate": get("Hallucination rate (%)"),
        "degenerate_rate": get("Degenerate output rate (%)"),
        "false_refusal_rate": get("False refusal rate (%)"),
        "correct_refusal_rate": get("Correct refusal rate (%)"),
        "human_accuracy": get("Human accuracy (%)"),
    }


def data_quality_warnings(df: pd.DataFrame) -> list[str]:
    """Rows that look like data-entry problems, reported on the console."""
    warnings: list[str] = list(df.attrs.get("scale_warnings", []))

    def check(mask: pd.Series, issue: str) -> None:
        ids = df.loc[mask, "id"].tolist()
        if ids:
            shown = ", ".join(str(i) for i in ids[:10])
            more = f" (+{len(ids) - 10} more)" if len(ids) > 10 else ""
            warnings.append(f"{issue}: id {shown}{more}")

    unans = df["qa_subtype"].eq(ST_UNANS)
    short = df["qa_subtype"].eq(ST_SHORT)
    noans_col, _ = UNANSWERABLE_METRIC

    check(unans & ~is_noans_text(df["reference_answer_sin"]),
          "unanswerable item whose reference answer is not the UNANSWERABLE token")
    check(unans & df[noans_col].isna(),
          "unanswerable item with no squad_v2_noans_em score (excluded from the mean)")
    check(short & df["token_f1"].isna(),
          "short_answer item with no token_f1 score (excluded from the mean)")
    check(short & df[noans_col].notna(),
          "short_answer item carrying a squad_v2_noans_em score (ignored; that "
          "metric applies to unanswerable items only)")
    check(df["hallucination_type"].isin([LBL_CORRECT_REFUSAL, LBL_FALSE_REFUSAL])
          & ~is_noans_text(df["model_output_sin"]),
          "labelled as a refusal but the model output is an actual answer")
    check(short & df["hallucination_type"].eq(LBL_CORRECT_REFUSAL),
          "short_answer item labelled Correct Refusal (should be False Refusal)")
    check(unans & df["hallucination_type"].isin([LBL_SUPPORTED, LBL_FALSE_REFUSAL]),
          "unanswerable item labelled Supported or False Refusal "
          "(contradicts its gold subtype)")
    check(~df["qa_subtype"].isin([ST_SHORT, ST_UNANS]), "unrecognised qa_subtype")
    check(~df["hallucination_type"].isin(HUMAN_LABELS), "unrecognised hallucination_type")
    check(df["id"].duplicated(keep=False), "duplicate id")

    return warnings


def print_table(model: str, results: pd.DataFrame) -> None:
    print(f"\n  {model}")
    width = max(26, results["metric"].str.len().max() + 2)
    print(f"  {'metric':<{width}}{'subset':<14}{'value':>8}{'n':>6}   95% CI")
    print("  " + "-" * (width + 40))
    for _, r in results.iterrows():
        ci = (f"   {r['ci95_low']:.2f} - {r['ci95_high']:.2f}"
              if r["ci95_low"] != "" else "")
        print(f"  {r['metric']:<{width}}{r['subset']:<14}{r['value']:>8.2f}"
              f"{r['n']:>6}{ci}")


def write_csv(df: pd.DataFrame, path: Path, root: Path) -> bool:
    try:
        shown = path.relative_to(root)
    except ValueError:                          # pragma: no cover
        shown = path
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    except PermissionError:
        # Excel and some editors hold an exclusive lock on an open CSV.
        print(f"    COULD NOT WRITE {shown} - the file is open in another "
              f"program. Close it and re-run.", file=sys.stderr)
        return False
    print(f"    wrote {shown}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=".", help="folder holding the annotation sheets")
    ap.add_argument("--output-dir", default="model_results", help="folder for the result sheets")
    ap.add_argument("--pattern", default="*.csv", help="glob for input sheets")
    ap.add_argument("--bootstrap", type=int, default=1000,
                    help="bootstrap resamples for the 95%% CIs (0 disables)")
    args = ap.parse_args()

    in_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sheets = sorted(p for p in in_dir.glob(args.pattern)
                    if p.is_file() and out_dir not in p.parents)
    if not sheets:
        print(f"No input sheets matched {args.pattern!r} in {in_dir}", file=sys.stderr)
        return 1

    print(f"Input : {in_dir}")
    print(f"Output: {out_dir}")
    print(f"Found {len(sheets)} sheet(s)")

    headlines = []
    for path in sheets:
        print(f"\n- {path.name}")
        try:
            raw = read_sheet(path)
            model = model_name_from(raw, path)
            df = normalise(raw)
        except Exception as err:
            print(f"    SKIPPED: {err}", file=sys.stderr)
            continue

        results = build_results(df, model, args.bootstrap)
        headlines.append(headline_row(results, df, model))

        slug = safe_dirname(model)
        model_dir = out_dir / slug
        model_dir.mkdir(parents=True, exist_ok=True)
        write_csv(results, model_dir / f"{slug}_results.csv", out_dir.parent)

        for warning in data_quality_warnings(df):
            print(f"    ! {warning}")
        print_table(model, results)

    ok = True
    if headlines:
        comparison = pd.DataFrame(headlines).sort_values("token_f1", ascending=False)
        print()
        ok = write_csv(comparison, out_dir / "all_models_results.csv", out_dir.parent)

    print(f"\nDone. {len(headlines)} model(s) written to {out_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
