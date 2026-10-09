from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import evaluate_models as em  # noqa: E402

OUT = PROJECT / "paper_artifacts"

MODEL_META = {
    "gemini-pro-agent":                  ("Gemini-3.1-Pro",      "Gemini",   "closed"),
    "gemini-3.6-flash-high":             ("Gemini-3.6-Flash",    "Gemini",   "closed"),
    "gpt-5.5":                           ("GPT-5.5",             "GPT",      "closed"),
    "claude-opus-4-6-thinking":          ("Claude-Opus-4.6",     "Claude",   "closed"),
    "claude-sonnet-4-6":                 ("Claude-Sonnet-4.6",   "Claude",   "closed"),
    "meta-llama/llama-4-maverick":       ("Llama-4-Maverick",    "Llama",    "open"),
    "meta-llama/llama-3.1-70b-instruct": ("Llama-3.1-70B",       "Llama",    "open"),
    "meta-llama/llama-3.1-8b-instruct":  ("Llama-3.1-8B",        "Llama",    "open"),
    "deepseek/deepseek-v4-pro-0813":     ("DeepSeek-V4-Pro",     "DeepSeek", "open"),
    "deepseek/deepseek-v4-flash-0731":   ("DeepSeek-V4-Flash",   "DeepSeek", "open"),
    "qwen/qwen3.5-27b":                  ("Qwen3.5-27B",         "Qwen",     "open"),
    "qwen/qwen3.5-9b":                   ("Qwen3.5-9B",          "Qwen",     "open"),
    "Qwen/Qwen3.5-4B":                   ("Qwen3.5-4B",          "Qwen",     "open"),
    "mistralai/mistral-large-2512":      ("Mistral-Large",       "Mistral",  "open"),
}

# Question-source subsets, keyed by the host in the `source` URL.
SUBSET_OF_HOST = {
    "huggingface.co":    "SiQuAD (span)",
    "en.wikivoyage.org": "Belebele (adapted)",
    "en.wikinews.org":   "Belebele (adapted)",
    "en.wikibooks.org":  "Unanswerable (constructed)",
}

SURFACE, INK, INK_2, MUTED = "#ffffff", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES_1 = "#2a78d6"
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"

OUTCOME_ORDER = [em.LBL_SUPPORTED, em.LBL_CORRECT_REFUSAL, em.LBL_FALSE_REFUSAL,
                 em.LBL_DEGENERATE, em.LBL_UNSUPPORTED]
OUTCOME_COLOR = {em.LBL_SUPPORTED: GOOD, em.LBL_CORRECT_REFUSAL: "#7fce7f",
                 em.LBL_FALSE_REFUSAL: WARNING, em.LBL_DEGENERATE: SERIOUS,
                 em.LBL_UNSUPPORTED: CRITICAL}
OUTCOME_SHORT = {em.LBL_SUPPORTED: "Supported", em.LBL_CORRECT_REFUSAL: "Correct refusal",
                 em.LBL_FALSE_REFUSAL: "False refusal",
                 em.LBL_DEGENERATE: "Degenerate output", em.LBL_UNSUPPORTED: "Unsupported"}

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,          # embed real fonts for camera-ready
    "font.family": ["DejaVu Sans"], "font.size": 8.5,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "legend.frameon": False,
})


def load_items() -> dict[str, pd.DataFrame]:
    items = {}
    for path in sorted(PROJECT.glob("*.csv")):
        sheet = em.read_sheet(path)
        try:
            model = em.model_name_from(sheet, path)
            data = em.normalise(sheet)
        except ValueError:
            continue

        misplaced = (data["qa_subtype"].eq(em.ST_SHORT)
                     & data["hallucination_type"].eq(em.LBL_CORRECT_REFUSAL))
        if misplaced.any():
            if not em.is_noans_text(data.loc[misplaced, "model_output_sin"]).all():
                ids = data.loc[misplaced, "id"].tolist()
                raise SystemExit(f"Correct Refusal on answerable item(s) with an "
                                 f"actual answer in {path.name}: {ids}")
            ids = data.loc[misplaced, "id"].tolist()
            print(f"  ! {path.name}: treating answerable Correct Refusal item(s) "
                  f"{ids} as False Refusal", file=sys.stderr)
            data.loc[misplaced, "hallucination_type"] = em.LBL_FALSE_REFUSAL

        items[model] = data
    missing = set(items) - set(MODEL_META)
    if missing:
        raise SystemExit(f"No display name for: {sorted(missing)} - add to MODEL_META")
    return items


def per_model_metrics(items: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for key, d in items.items():
        name, family, access = MODEL_META[key]
        sa = d[d["qa_subtype"].eq(em.ST_SHORT)]
        un = d[d["qa_subtype"].eq(em.ST_UNANS)]
        lab = sa["hallucination_type"]

        n_sa, n_un = len(sa), len(un)
        false_ref = int(lab.eq(em.LBL_FALSE_REFUSAL).sum())
        supported = int(lab.eq(em.LBL_SUPPORTED).sum())
        unsupported = int(lab.eq(em.LBL_UNSUPPORTED).sum())
        degenerate = int(d["hallucination_type"].eq(em.LBL_DEGENERATE).sum())
        correct_ref = int(un["hallucination_type"].eq(em.LBL_CORRECT_REFUSAL).sum())
        unsupported_un = int(un["hallucination_type"].eq(em.LBL_UNSUPPORTED).sum())

        ok = sa[sa["hallucination_type"].eq(em.LBL_SUPPORTED)]
        out_len = ok["model_output_sin"].fillna("").astype(str).str.split().str.len()
        ref_len = (ok["reference_answer_sin"].fillna("").astype(str)
                   .str.split().str.len().clip(lower=1))

        rows.append({
            "key": key, "model": name, "family": family, "access": access,
            "Verbosity": (out_len / ref_len).median(),
            # automatic, answerable
            "EM": sa["exact_match"].mean(),
            "F1": sa["token_f1"].mean(),
            "chrF": sa["chrf"].mean(),
            "chrF3": sa["chrf3"].mean(),
            # human labels, reported on the subset where each label applies
            "SupportedSA": 100 * supported / n_sa,
            "FRR": 100 * false_ref / n_sa,
            "UnsupportedSA": 100 * unsupported / n_sa,
            # unanswerable
            "NoAnsEM": pd.to_numeric(un["squad_v2_noans_em"], errors="coerce").mean(),
            "CorrectRefUn": 100 * correct_ref / n_un,
            "UnsupportedUn": 100 * unsupported_un / n_un,
            # overall
            "Degen": 100 * degenerate / len(d),
            "HumanAcc": 100 * (supported + correct_ref) / len(d),
            "n_supported": supported, "n_unsupported": unsupported,
            "n_false_ref": false_ref, "n_correct_ref": correct_ref,
            "n_unsupported_un": unsupported_un,
            "n_degen": degenerate, "n_sa": n_sa, "n_un": n_un,
        })
    order = {k: i for i, k in enumerate(MODEL_META)}
    return pd.DataFrame(rows).sort_values("key", key=lambda s: s.map(order))

def f(v, nd=1):
    return "--" if pd.isna(v) else f"{v:.{nd}f}"


def table_main(t: pd.DataFrame) -> str:
    n_models = len(t)
    n_sa = int(t["n_sa"].iloc[0])
    n_un = int(t["n_un"].iloc[0])
    n_all = n_sa + n_un
    L = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l rrrr rrr rrr rr}",
        r"\toprule",
        r" & \multicolumn{4}{c}{\textbf{Answerable: overlap metrics}}"
        r" & \multicolumn{3}{c}{\textbf{Answerable: human labels}}"
        r" & \multicolumn{3}{c}{\textbf{Unanswerable}} & & \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
        r"\textbf{Model} & EM & F1 & chrF & chrF3 & Sup.$\uparrow$ & Uns.$\downarrow$"
        r" & False Ref.$\downarrow$ & NoAns-EM$\uparrow$ & Corr. Ref.$\uparrow$"
        r" & Uns.$\downarrow$"
        r" & \textbf{Acc}$\uparrow$ & Len \\",
        r"\midrule",
        r"\multicolumn{13}{l}{\textit{Closed-source}} \\",
    ]
    for access, header in (("closed", None), ("open", r"\textit{Open-weight}")):
        if header:
            L += [r"\midrule", rf"\multicolumn{{13}}{{l}}{{{header}}} \\"]
        for _, r in t[t["access"].eq(access)].iterrows():
            L.append(
                f"{r['model']} & {f(r.EM)} & {f(r.F1)} & {f(r.chrF)} & {f(r.chrF3)} & "
                f"{f(r.SupportedSA)} & {f(r.UnsupportedSA)} & {f(r.FRR)} & "
                f"{f(r.NoAnsEM)} & {f(r.CorrectRefUn)} & {f(r.UnsupportedUn)} & "
                f"{f(r.HumanAcc)} & "
                rf"{f(r.Verbosity, 1)}$\times$ \\")
    L += [
        r"\bottomrule", r"\end{tabular}}",
        rf"\caption{{QA results for all {n_models} systems (\%). Human-label rates use "
        rf"all {n_sa} answerable or {n_un} unanswerable items, as appropriate. "
        r"Automatic metrics are averaged over the available non-missing scores. "
        r"\textbf{Sup.}: Supported. \textbf{Uns.}: Unsupported. "
        r"\textbf{False Ref.}: False Refusal. \textbf{Corr. Ref.}: Correct Refusal. "
        rf"\textbf{{Acc}}: human accuracy over all {n_all} items "
        r"(Supported $+$ Correct refusal). "
        r"\textbf{Len}: median per-item length ratio between answers judged Supported "
        r"and their reference spans.}",
        r"\label{tab:qa-main}", r"\end{table*}",
    ]
    return "\n".join(L)


def table_subset(items: dict[str, pd.DataFrame]) -> tuple[str, pd.DataFrame]:
    """Aggregate over models, broken down by question-source subset."""
    frames = []
    for key, d in items.items():
        d = d.copy()
        host = (d["source"].fillna("").astype(str)
                .str.extract(r"https?://([^/]+)", expand=False).fillna("(none)"))
        d["subset"] = host.map(SUBSET_OF_HOST).fillna("(other)")
        d["model"] = MODEL_META[key][0]
        frames.append(d)
    allm = pd.concat(frames, ignore_index=True)

    allm.loc[allm["qa_subtype"].eq(em.ST_UNANS), "subset"] = "Unanswerable (constructed)"

    rows = []
    for subset, g in allm.groupby("subset"):
        n_items = int(g.groupby("model").size().iloc[0])
        lab = g["hallucination_type"]
        n = len(g)
        answerable = bool(g["qa_subtype"].eq(em.ST_SHORT).all())
        rows.append({
            "subset": subset,
            "n": n_items,
            "answerable": answerable,
            # Overlap metrics need a reference answer, so they exist only here.
            "F1": g["token_f1"].mean() if answerable else np.nan,
            "chrF": g["chrf"].mean() if answerable else np.nan,

            "Correct": 100 * lab.eq(em.LBL_SUPPORTED if answerable
                                    else em.LBL_CORRECT_REFUSAL).sum() / n,
            "Unsupported": 100 * lab.eq(em.LBL_UNSUPPORTED).sum() / n,
        })
    tab = (pd.DataFrame(rows)
           .sort_values(["answerable", "n"], ascending=[False, False]))

    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\begin{tabular}{l r rr rr}", r"\toprule",
         r"\textbf{Source subset} & \textbf{n} & \textbf{F1} & \textbf{chrF}"
         r" & \textbf{Corr.} & \textbf{Uns.} \\", r"\midrule"]
    for _, r in tab.iterrows():
        f1 = "--" if pd.isna(r.F1) else f(r.F1)
        cf = "--" if pd.isna(r.chrF) else f(r.chrF)
        L.append(f"{r.subset} & {int(r.n)} & {f1} & {cf} & "
                 f"{f(r.Correct)} & {f(r.Unsupported)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          rf"\caption{{QA results by question-source subset, pooled over all {len(items)} systems "
          r"(\%). \textbf{n} is items per system. \textbf{Corr.} is Supported for the "
          r"answerable subsets and Correct refusal for the unanswerable subset; "
          r"\textbf{Uns.} is the Unsupported (hallucination) share. Overlap metrics "
          r"require a reference answer and are undefined for unanswerable items.}",
          r"\label{tab:qa-subset}", r"\end{table}"]
    return "\n".join(L), tab


#Figure
def fig_behaviour(items: dict[str, pd.DataFrame], t: pd.DataFrame) -> None:
    order = t.sort_values("HumanAcc")["key"].tolist()
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    y = np.arange(len(order))
    for i, key in enumerate(order):
        d = items[key]
        left = 0.0
        for lbl in OUTCOME_ORDER:
            n = int(d["hallucination_type"].eq(lbl).sum())
            if not n:
                continue
            pct = 100 * n / len(d)
            ax.barh(i, pct, left=left, height=0.66, color=OUTCOME_COLOR[lbl],
                    edgecolor=SURFACE, linewidth=1.2, zorder=3,
                    label=OUTCOME_SHORT[lbl] if i == 0 else None)
            left += pct

    handles = [mpl.patches.Patch(facecolor=OUTCOME_COLOR[l], label=OUTCOME_SHORT[l])
               for l in OUTCOME_ORDER]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.16),
              ncol=5, fontsize=7.4, handlelength=1.1, columnspacing=1.1)
    ax.set_yticks(y, [MODEL_META[k][0] for k in order], fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of QA outputs (%)", fontsize=8.5)
    ax.grid(axis="x")
    fig.savefig(OUT / "fig_qa_behaviour.pdf")
    fig.savefig(OUT / "fig_qa_behaviour.png", dpi=200)
    plt.close(fig)


def _label_points(ax, xs, ys, labels, fontsize=7.0, manual_offsets=None):
    from matplotlib.transforms import Bbox
    fig = ax.figure
    fig.canvas.draw()
    frame = ax.get_window_extent()

    placed = []
    for px, py in zip(xs, ys):
        mx, my = ax.transData.transform((px, py))
        placed.append(Bbox.from_bounds(mx - 5, my - 5, 10, 10))

    manual_offsets = manual_offsets or {}
    offsets = [(0, 8), (0, -12), (9, 3), (-9, 3), (0, 17), (0, -21),
                (11, -10), (-11, -10), (0, 26), (0, -30), (20, 3), (-20, 3),
                (16, 14), (-16, 14), (16, -18), (-16, -18),
                (28, 8), (-28, 8), (28, -12), (-28, -12), (36, 0), (-36, 0)]
    for i in np.argsort(-np.asarray(ys)):
        label_offsets = []
        if labels[i] in manual_offsets:
            label_offsets.append(manual_offsets[labels[i]])
        label_offsets.extend(offsets)
        last_ann = None
        for dx, dy in label_offsets:
            ha = "center" if dx == 0 else ("left" if dx > 0 else "right")
            ann = ax.annotate(labels[i], (xs[i], ys[i]), xytext=(dx, dy),
                              textcoords="offset points", ha=ha, va="center",
                              fontsize=fontsize, color=INK_2, zorder=5)
            fig.canvas.draw()
            box = ann.get_window_extent()
            inside = (box.x0 >= frame.x0 and box.x1 <= frame.x1
                      and box.y0 >= frame.y0 and box.y1 <= frame.y1)
            if inside and not any(box.overlaps(b) for b in placed):
                placed.append(box.expanded(1.04, 1.15))
                if abs(dy) > 15 or abs(dx) > 15:
                    ax.annotate("", (xs[i], ys[i]), xytext=(dx * 0.55, dy * 0.55),
                                textcoords="offset points",
                                arrowprops=dict(arrowstyle="-", color=AXIS,
                                                linewidth=0.6, shrinkA=0, shrinkB=3),
                                zorder=2)
                break
            last_ann = (dx, dy)
            ann.remove()
        else:
            dx, dy = last_ann or (8, 8)
            ha = "center" if dx == 0 else ("left" if dx > 0 else "right")
            ax.annotate(labels[i], (xs[i], ys[i]), xytext=(dx, dy),
                        textcoords="offset points", ha=ha, va="center",
                        fontsize=fontsize, color=INK_2, zorder=5,
                        arrowprops=dict(arrowstyle="-", color=AXIS,
                                        linewidth=0.5, shrinkA=0, shrinkB=3))


def _indexed_labels(ax, frame: pd.DataFrame, offsets: dict[str, tuple[int, int]],
                    fontsize=6.8) -> None:
    """Use compact point indices plus an in-panel legend for dense regions."""
    letters = list("abcdefghijklmnopqrstuvwxyz")
    legend_lines = []
    for letter, (_, row) in zip(letters, frame.iterrows()):
        name = row["model"]
        dx, dy = offsets.get(name, (0, 0))
        ax.annotate(
            letter, xy=(row["FRR"], row["UnsupportedUn"]), xytext=(dx, dy),
            textcoords="offset points", ha="center", va="center",
            fontsize=fontsize + 0.8, fontweight="bold", color=SERIES_1,
            zorder=7,
        )
        legend_lines.append(f"{letter} - {name}")

    ax.text(
        0.64, 0.97, "\n".join(legend_lines), transform=ax.transAxes,
        ha="left", va="top", fontsize=6.35, color=INK_2,
        bbox=dict(boxstyle="round,pad=0.28", facecolor=SURFACE,
                  edgecolor=GRID, linewidth=0.45, alpha=0.96),
        zorder=6,
    )


ZOOM_FRR, ZOOM_UNSUPPORTED = 18.0, 30.0  # the well-performing window, panel (b)


def fig_calibration(t: pd.DataFrame) -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.8, 4.2))

    def draw(ax, sub, xlim, ylim, label_all, manual_offsets=None, fontsize=6.8):
        closed = sub["access"].eq("closed")
        ax.scatter(sub.loc[closed, "FRR"], sub.loc[closed, "UnsupportedUn"], s=38,
                   color=SERIES_1, edgecolor=SURFACE, linewidth=1.1, zorder=4,
                   marker="o", label="closed")
        ax.scatter(sub.loc[~closed, "FRR"], sub.loc[~closed, "UnsupportedUn"], s=42,
                   facecolor="none", edgecolor=SERIES_1, linewidth=1.2, zorder=4,
                   marker="s", label="open")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        lab = sub if label_all else sub[sub["UnsupportedUn"] > ZOOM_UNSUPPORTED]
        _label_points(ax, lab["FRR"].to_numpy(), lab["UnsupportedUn"].to_numpy(),
                      lab["model"].tolist(), fontsize=fontsize,
                      manual_offsets=manual_offsets)
        ax.set_xlabel("False refusal rate (answerable) %", fontsize=8)
        ax.grid(axis="both")

    # (a) full range - only the outliers are labelled; the rest are the box.
    draw(ax_a, t, (0, t["FRR"].max() * 1.30 + 4), (0, 112), label_all=False,
         manual_offsets={"Llama-3.1-70B": (10, 0), "Llama-3.1-8B": (10, 0),
                         "Mistral-Large": (10, 0)})
    ax_a.add_patch(mpl.patches.Rectangle(
        (0, 0), ZOOM_FRR, ZOOM_UNSUPPORTED, fill=False,
        edgecolor=MUTED, linewidth=0.8, zorder=2))
    ax_a.annotate("(b)", (ZOOM_FRR, ZOOM_UNSUPPORTED), xytext=(4, 4),
                  textcoords="offset points", fontsize=7.6, color=MUTED)
    ax_a.set_ylabel("Unsupported rate (unanswerable) %", fontsize=8)
    ax_a.set_title("(a) all systems", fontsize=8.5, loc="left", pad=6)
    ax_a.legend(loc="center right", fontsize=7.2, handletextpad=0.3)

    # (b) the well-calibrated corner, every system in it labelled.
    zoom = t[(t["FRR"] <= ZOOM_FRR) & (t["UnsupportedUn"] <= ZOOM_UNSUPPORTED)]
    draw(ax_b, zoom, (0, ZOOM_FRR + 10.0),
         (0, ZOOM_UNSUPPORTED + 8.5), label_all=False, fontsize=6.1)
    zoom_for_legend = zoom.sort_values(["UnsupportedUn", "FRR"], ascending=[False, True])
    _indexed_labels(ax_b, zoom_for_legend, {
        "Llama-4-Maverick": (0, 7),
        "Qwen3.5-4B": (8, 4),
        "Qwen3.5-9B": (0, 8),
        "GPT-5.5": (0, 7),
        "DeepSeek-V4-Pro": (8, 3),
        "Claude-Opus-4.6": (-7, 7),
        "Claude-Sonnet-4.6": (8, -7),
        "DeepSeek-V4-Flash": (8, 7),
        "Gemini-3.1-Pro": (7, -7),
        "Gemini-3.6-Flash": (0, 7),
        "Qwen3.5-27B": (0, 8),
    })
    ax_b.annotate("ideal", (0, 0), xytext=(1.0, 2.0), textcoords="data",
                  fontsize=7.8, fontweight="bold", color=GOOD,
                  bbox=dict(boxstyle="round,pad=0.12", facecolor=SURFACE,
                            edgecolor="none", alpha=0.92))
    ax_b.set_title(f"(b) detail: False Ref. $\\leq$ {ZOOM_FRR:.0f}, "
                   f"Unsupported $\\leq$ {ZOOM_UNSUPPORTED:.0f}",
                   fontsize=8.5, loc="left", pad=6)

    fig.tight_layout()
    fig.savefig(OUT / "fig_qa_calibration.pdf")
    fig.savefig(OUT / "fig_qa_calibration.png", dpi=200)
    plt.close(fig)

    fig_a, ax = plt.subplots(figsize=(4.2, 4.0))
    draw(ax, t, (0, t["FRR"].max() * 1.30 + 4), (0, 112), label_all=False,
         manual_offsets={"Llama-3.1-70B": (10, 0), "Llama-3.1-8B": (10, 0),
                         "Mistral-Large": (10, 0)})
    ax.add_patch(mpl.patches.Rectangle(
        (0, 0), ZOOM_FRR, ZOOM_UNSUPPORTED, fill=False,
        edgecolor=MUTED, linewidth=0.8, zorder=2))
    ax.annotate("detail", (ZOOM_FRR, ZOOM_UNSUPPORTED), xytext=(4, 4),
                textcoords="offset points", fontsize=7.6, color=MUTED)
    ax.set_ylabel("Unsupported rate (unanswerable) %", fontsize=8)
    ax.set_title("All systems", fontsize=8.5, loc="left", pad=6)
    ax.legend(loc="center right", fontsize=7.2, handletextpad=0.3)
    fig_a.tight_layout()
    fig_a.savefig(OUT / "fig_qa_calibration_all.pdf")
    fig_a.savefig(OUT / "fig_qa_calibration_all.png", dpi=200)
    plt.close(fig_a)

    fig_b, ax = plt.subplots(figsize=(4.2, 4.0))
    draw(ax, zoom, (0, ZOOM_FRR + 10.0),
         (0, ZOOM_UNSUPPORTED + 8.5), label_all=False, fontsize=6.1)
    _indexed_labels(ax, zoom_for_legend, {
        "Llama-4-Maverick": (0, 7),
        "Qwen3.5-4B": (8, 4),
        "Qwen3.5-9B": (0, 8),
        "GPT-5.5": (0, 7),
        "DeepSeek-V4-Pro": (8, 3),
        "Claude-Opus-4.6": (-7, 7),
        "Claude-Sonnet-4.6": (8, -7),
        "DeepSeek-V4-Flash": (8, 7),
        "Gemini-3.1-Pro": (7, -7),
        "Gemini-3.6-Flash": (0, 7),
        "Qwen3.5-27B": (0, 8),
    })
    ax.annotate("ideal", (0, 0), xytext=(1.0, 2.0), textcoords="data",
                fontsize=7.8, fontweight="bold", color=GOOD,
                bbox=dict(boxstyle="round,pad=0.12", facecolor=SURFACE,
                          edgecolor="none", alpha=0.92))
    ax.set_title(f"Detail: False Ref. $\\leq$ {ZOOM_FRR:.0f}, "
                 f"Unsupported $\\leq$ {ZOOM_UNSUPPORTED:.0f}",
                 fontsize=8.5, loc="left", pad=6)
    fig_b.tight_layout()
    fig_b.savefig(OUT / "fig_qa_calibration_zoom.pdf")
    fig_b.savefig(OUT / "fig_qa_calibration_zoom.png", dpi=200)
    plt.close(fig_b)


MOVER_THRESHOLD = 4          # rank places moved before a system is highlighted


def fig_metric_gap(t: pd.DataFrame) -> float:

    rho = float(t[["F1", "SupportedSA"]].corr(method="spearman").iat[0, 1])

    # Unique ranks: a tie on the primary key is broken by the other metric, so
    # two systems never land on the same slot and overprint each other.
    r = t.copy()
    r["rank_f1"] = (r.sort_values(["F1", "SupportedSA"], ascending=False)
                     .assign(k=range(1, len(r) + 1))["k"])
    r["rank_car"] = (r.sort_values(["SupportedSA", "F1"], ascending=False)
                      .assign(k=range(1, len(r) + 1))["k"])
    r["move"] = (r["rank_f1"] - r["rank_car"]).abs()

    fig, ax = plt.subplots(figsize=(3.35, 4.3))
    for _, m in r.iterrows():
        big = m["move"] >= MOVER_THRESHOLD
        ax.plot([0, 1], [m.rank_f1, m.rank_car], zorder=4 if big else 3,
                color=SERIES_1 if big else "#cfcec7",
                linewidth=1.5 if big else 1.0)
        ax.scatter([0, 1], [m.rank_f1, m.rank_car], s=16, zorder=5,
                   color=SERIES_1 if big else "#cfcec7")
        ax.annotate(m.model, (0, m.rank_f1), xytext=(-6, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=6.8, color=INK if big else MUTED)
        ax.annotate(m.model, (1, m.rank_car), xytext=(6, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=6.8, color=INK if big else MUTED)

    ax.set_xlim(-1.02, 2.02)
    ax.set_ylim(len(r) + 0.6, 0.4)          # rank 1 at the top
    ax.set_xticks([0, 1], ["by Token F1", "by human\nSupported rate"], fontsize=8)
    ax.set_yticks([])
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"Spearman $\\rho$ = {rho:.2f}", fontsize=8.5, loc="left", pad=6)

    fig.savefig(OUT / "fig_qa_metric_gap.pdf")
    fig.savefig(OUT / "fig_qa_metric_gap.png", dpi=200)
    plt.close(fig)
    return rho

def write_numbers(t: pd.DataFrame, rho: float, sub: pd.DataFrame) -> None:
    def cmd(name, val, nd=1):
        return (rf"\newcommand{{\{name}}}{{{val:.{nd}f}}}" if isinstance(val, float)
                else rf"\newcommand{{\{name}}}{{{val}}}")

    best_supported = t.loc[t["SupportedSA"].idxmax()]
    worst_supported = t.loc[t["SupportedSA"].idxmin()]
    best_unsupported_un = t.loc[t["UnsupportedUn"].idxmin()]
    worst_unsupported_un = t.loc[t["UnsupportedUn"].idxmax()]
    best_f1 = t.loc[t["F1"].idxmax()]
    gap = (t["NoAnsEM"] - t["CorrectRefUn"]).abs()

    n_sa = int(t["n_sa"].iloc[0])
    n_un = int(t["n_un"].iloc[0])
    L = ["% Auto-generated by make_paper_artifacts.py - do not edit by hand.",
         cmd("qaNModels", len(t), 0),
         cmd("qaNItems", n_sa + n_un, 0), cmd("qaNAnswerable", n_sa, 0),
         cmd("qaNUnanswerable", n_un, 0),
         cmd("qaSupportedMax", best_supported.SupportedSA),
         cmd("qaSupportedMaxModel", best_supported.model),
         cmd("qaSupportedMin", worst_supported.SupportedSA),
         cmd("qaSupportedMinModel", worst_supported.model),
         cmd("qaUnsupportedMin", t["UnsupportedSA"].min()),
         cmd("qaUnsupportedMax", t["UnsupportedSA"].max()),
         cmd("qaFrrMin", t["FRR"].min()), cmd("qaFrrMax", t["FRR"].max()),
         cmd("qaUnsupportedUnMin", best_unsupported_un.UnsupportedUn),
         cmd("qaUnsupportedUnMinModel", best_unsupported_un.model),
         cmd("qaUnsupportedUnMax", worst_unsupported_un.UnsupportedUn),
         cmd("qaUnsupportedUnMaxModel", worst_unsupported_un.model),
         cmd("qaUnsupportedUnMedian", float(t["UnsupportedUn"].median())),
         cmd("qaF1Max", best_f1.F1), cmd("qaF1MaxModel", best_f1.model),
         cmd("qaEmMax", t["EM"].max()), cmd("qaEmMin", t["EM"].min()),
         cmd("qaSpearman", rho, 2),
         cmd("qaNoAnsGapMax", float(gap.max())),
         cmd("qaNoAnsGapModel", t.loc[gap.idxmax(), "model"]),
         cmd("qaNGapFive", int((gap >= 5).sum()), 0),
         cmd("qaDegenMax", t["Degen"].max()),
         cmd("qaDegenModel", t.loc[t["Degen"].idxmax(), "model"]),
         cmd("qaAccMax", t["HumanAcc"].max()),
         cmd("qaAccMaxModel", t.loc[t["HumanAcc"].idxmax(), "model"]),
         cmd("qaAccMin", t["HumanAcc"].min()),
         cmd("qaAccMinModel", t.loc[t["HumanAcc"].idxmin(), "model"]),
         # Verbosity: why the overlap metrics and the human labels disagree.
         cmd("qaVerbRhoFOne", float(t[["Verbosity", "F1"]]
                                   .corr(method="spearman").iat[0, 1]), 2),
         cmd("qaVerbRhoEM", float(t[["Verbosity", "EM"]]
                                  .corr(method="spearman").iat[0, 1]), 2),
         cmd("qaVerbMax", t["Verbosity"].max()),
         cmd("qaVerbMaxModel", t.loc[t["Verbosity"].idxmax(), "model"]),
         cmd("qaVerbMin", t["Verbosity"].min()),
         ]
    (OUT / "qa_numbers.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------

def main() -> int:
    OUT.mkdir(exist_ok=True)
    items = load_items()
    t = per_model_metrics(items)

    (OUT / "tab_qa_main.tex").write_text(table_main(t), encoding="utf-8")
    subset_tex, sub = table_subset(items)
    (OUT / "tab_qa_subset.tex").write_text(subset_tex, encoding="utf-8")

    fig_behaviour(items, t)
    fig_calibration(t)
    rho = fig_metric_gap(t)
    write_numbers(t, rho, sub)

    try:
        t.round(2).to_csv(OUT / "qa_metrics.csv", index=False, encoding="utf-8-sig")
    except PermissionError:      # Excel holds an exclusive lock on an open CSV
        print("  ! qa_metrics.csv is open in another program - not rewritten",
              file=sys.stderr)

    pd.set_option("display.width", 220)
    print(t[["model", "access", "EM", "F1", "chrF", "SupportedSA", "UnsupportedSA",
             "FRR", "NoAnsEM", "CorrectRefUn", "UnsupportedUn", "HumanAcc"]]
          .round(2).to_string(index=False))
    print(f"\nSpearman(F1, Supported) = {rho:.2f}")
    print("\nBy source subset:")
    print(sub.round(2).to_string(index=False))
    print(f"\nWrote {len(list(OUT.iterdir()))} files to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
