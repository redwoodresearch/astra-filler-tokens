"""Accuracy-by-release-date and METR-style time-horizon analysis of the no-CoT suite.

usage: uv run -m nf.horizon suite suite_deep sweep_openai sweep_or

Per model: fit P(correct | task, depth) = chance + (1 - chance) * sigmoid(a + b * log2(human_minutes)) by maximum
likelihood over all instances (depths <= MAX_DEPTH, tasks in FIT_TASKS), where human_minutes comes from
data/human_time_ratings.json (independent subagent estimates). TH50 is the human time at which the
chance-corrected success probability is 0.5 (TH80: 0.8). Instance-level bootstrap gives CIs. Release dates come
from the OpenRouter catalog `created` field, overridden by official dates for OpenAI-direct models."""

import datetime as dt
import json
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from .analyze import DATA, load, tag_dir

matplotlib.use("Agg")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 12})

FIT_TASKS = ("perm", "lookup", "modchain", "reach", "stack", "xor")  # ca excluded: degenerate (see README)
MAX_DEPTH = 16  # common support across all models
RELEASE = {  # OpenAI-direct ids -> official release dates
    "gpt-4o": "2024-05-13",
    "gpt-4.1": "2025-04-14",
    "gpt-5": "2025-08-07",
    "gpt-5.1": "2025-11-13",
    "gpt-5.2": "2025-12-11",
    "gpt-5.4": "2026-03-05",
    "gpt-5.5": "2026-04-23",
    "gpt-5.6-sol": "2026-07-09",
    "gpt-5.6-terra": "2026-07-09",
    "gpt-5.6-luna": "2026-07-09",
    "gpt-6-astra": "2026-09-03",
    "claude-fable-5-1": "2026-08-28",
    "claude-fable-5": "2026-06-07",
    "claude-opus-5": "2026-07-24",
    "claude-sonnet-5": "2026-06-29",
    "claude-opus-4-8": "2026-05-28",
    "claude-sonnet-4-6": "2026-02-17",
    "claude-haiku-4-5": "2025-10-15",
}
EXCLUDE = {"gpt-5.6-sol:low", "claude-opus-5:low"}  # matched-effort duplicates kept for comparison, not for trends
FAMILY_COLORS = {"openai": "#4878CF", "anthropic": "#D65F5F", "google": "#6ACC65", "x-ai": "#B47CC7", "open": "#C4AD66"}


def family(model: str) -> str:
    for f in ("anthropic", "google", "x-ai"):
        if model.startswith(f + "/"):
            return f
    if model.startswith("claude"):
        return "anthropic"
    return "openai" if model.startswith("gpt") else "open"


def release_dates() -> dict[str, dt.date]:
    cat = json.loads((DATA / "openrouter_models_catalog.json").read_text())
    dates = {m["id"]: dt.datetime.fromtimestamp(m["created"], tz=dt.UTC).date() for m in cat}
    dates.update({k: dt.date.fromisoformat(v) for k, v in RELEASE.items()})
    return dates


def human_minutes() -> dict[tuple[str, int], float]:
    h = json.loads((DATA / "human_time_ratings.json").read_text())["tasks"]
    return {(t, int(d)): s / 60 for t, v in h.items() for d, s in v["seconds"].items()}


def fit_horizon(x: np.ndarray, y: np.ndarray, c: np.ndarray, reg: float = 0.05):
    """x = log2 human minutes, y in {0,1}, c = chance. Returns (a, b) of chance-corrected logistic."""

    def nll(p):
        a, b = p
        q = c + (1 - c) * expit(a + b * x)
        q = np.clip(q, 1e-6, 1 - 1e-6)
        return (
            -(y * np.log(q) + (1 - y) * np.log(1 - q)).sum() + reg * (b + 1) ** 2
        )  # weak prior: slope ~ -1 per doubling

    best = min(
        (
            minimize(nll, [a0, b0], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 2000})
            for a0 in (0.0, 2.0, -2.0)
            for b0 in (-1.0, -3.0)
        ),
        key=lambda r: r.fun,
    )
    return best.x


def horizon_from_params(a, b, p=0.5):
    return float(2 ** ((np.log(p / (1 - p)) - a) / b)) if b < 0 else np.inf


def per_model(df: pd.DataFrame, n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    hm = human_minutes()
    d = df[df.task.isin(FIT_TASKS) & (df.depth <= MAX_DEPTH)].copy()
    d["x"] = np.log2([hm[(t, int(k))] for t, k in zip(d.task, d.depth)])
    rng = np.random.default_rng(seed)
    rows = []
    for me, g in d.groupby("model_eff"):
        x, y, c = g.x.values, g.correct.astype(float).values, g.chance.values.astype(float)
        a, b = fit_horizon(x, y, c)
        boots = []
        for _ in range(n_boot):
            i = rng.integers(0, len(g), len(g))
            ab = fit_horizon(x[i], y[i], c[i])
            boots.append((horizon_from_params(*ab), horizon_from_params(*ab, 0.8)))
        boots = np.array(boots)
        rows.append(
            {
                "model_eff": me,
                "n": len(g),
                "th50_min": horizon_from_params(a, b),
                "th50_lo": np.nanpercentile(boots[:, 0], 2.5),
                "th50_hi": np.nanpercentile(boots[:, 0], 97.5),
                "th80_min": horizon_from_params(a, b, 0.8),
                "th80_lo": np.nanpercentile(boots[:, 1], 2.5),
                "th80_hi": np.nanpercentile(boots[:, 1], 97.5),
                "slope": b,
                "intercept": a,
                "norm_acc": float(((g.correct - g.chance) / (1 - g.chance)).mean()),
            }
        )
    return pd.DataFrame(rows)


def attach_dates(r: pd.DataFrame) -> pd.DataFrame:
    dates = release_dates()
    r = r.copy()
    r["model"] = r.model_eff.str.rsplit(":", n=1).str[0]
    r["release"] = r.model.map(dates)
    r["family"] = r.model.map(family)
    r["label"] = r.model.str.split("/").str[-1]
    return r


def doubling_days(r: pd.DataFrame, col: str) -> tuple[float, float]:
    """OLS of log2(horizon) on date -> doubling time in days and R^2."""
    t = np.array([(x - dt.date(2024, 1, 1)).days for x in r.release], float)
    y = np.log2(r[col].values.astype(float))
    ok = np.isfinite(y)
    b, a = np.polyfit(t[ok], y[ok], 1)
    pred = a + b * t[ok]
    r2 = 1 - ((y[ok] - pred) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum()
    return 1 / b, r2


def fig_accuracy_by_date(r: pd.DataFrame, out):
    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    for fam, g in r.groupby("family"):
        ax.scatter(g.release, g.norm_acc, s=70, color=FAMILY_COLORS[fam], label=fam, zorder=3, edgecolor="white")
    for _, row in r.iterrows():
        ax.annotate(row.label, (row.release, row.norm_acc), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set_ylabel("chance-corrected accuracy, no CoT\n(mean over 6 tasks x depths 1-16)", fontsize=12)
    ax.set_xlabel("release date", fontsize=12)
    ax.set_title("No-CoT serial-depth suite: accuracy by model release date", fontsize=14)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_horizon_by_date(r: pd.DataFrame, metr: pd.DataFrame, out):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True)
    ax = axes[0]
    pre = r[r.model != "gpt-6-astra"]
    dbl, r2 = doubling_days(pre, "th50_min")
    for fam, g in r.groupby("family"):
        ax.errorbar(
            g.release,
            g.th50_min,
            yerr=[g.th50_min - g.th50_lo, g.th50_hi - g.th50_min],
            fmt="o",
            color=FAMILY_COLORS[fam],
            label=fam,
            capsize=2,
            ms=7,
            mec="white",
        )
    for _, row in r.iterrows():
        ax.annotate(row.label, (row.release, row.th50_min), fontsize=8, xytext=(4, 3), textcoords="offset points")
    t = np.array([(x - dt.date(2024, 1, 1)).days for x in pre.release], float)
    b, a = np.polyfit(t, np.log2(pre.th50_min.values.astype(float)), 1)
    xs = np.linspace(t.min(), (dt.date(2026, 10, 1) - dt.date(2024, 1, 1)).days, 50)
    ax.plot(
        [dt.date(2024, 1, 1) + dt.timedelta(days=float(v)) for v in xs],
        2 ** (a + b * xs),
        "k--",
        lw=1,
        label=f"trend excl. Astra: doubling every {dbl:.0f} d (R²={r2:.2f})",
    )
    ax.set_yscale("log")
    ax.set_ylabel("50% time horizon on this suite (human minutes, no CoT)", fontsize=12)
    ax.set_xlabel("release date", fontsize=12)
    ax.set_title("This suite: no-CoT 50% time horizon vs release date", fontsize=13)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    m = metr[metr.release_date >= "2024-01-01"].copy()
    m["release"] = pd.to_datetime(m.release_date).dt.date
    ax.errorbar(
        m.release,
        m.p50_min,
        yerr=[m.p50_min - m.p50_lo, m.p50_hi - m.p50_min],
        fmt="s",
        color="gray",
        capsize=2,
        ms=6,
        label="METR HCAST/RE-Bench/SWAA p50 (agentic, with CoT)",
    )
    for _, row in m.iterrows():
        ax.annotate(
            row.metr_id.replace("_inspect", ""),
            (row.release, row.p50_min),
            fontsize=7,
            xytext=(4, 3),
            textcoords="offset points",
        )
    mdbl, mr2 = doubling_days(m.rename(columns={"p50_min": "th50_min"}), "th50_min")
    ax.set_yscale("log")
    ax.set_ylabel("METR 50% time horizon (human minutes)", fontsize=12)
    ax.set_xlabel("release date", fontsize=12)
    ax.set_title(f"METR time horizons (v1.1, 2024+): doubling every {mdbl:.0f} d (R²={mr2:.2f})", fontsize=13)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return dbl, r2, mdbl, mr2


if __name__ == "__main__":
    tags = sys.argv[1:]
    df = pd.concat([load(t) for t in tags])
    df = df[(df.status != "error") & (df.arm == "B") & (df.k == 0) & ~df.model_eff.isin(EXCLUDE)]
    out = tag_dir("horizon")
    out.mkdir(exist_ok=True)
    r = attach_dates(per_model(df))
    r = r.dropna(subset=["release"]).sort_values("release")
    pd.set_option("display.width", 250)
    print(
        r[["label", "model_eff", "release", "n", "norm_acc", "th50_min", "th50_lo", "th50_hi", "th80_min", "slope"]]
        .round(3)
        .to_string(index=False)
    )
    r.to_csv(out / "horizons_by_model.csv", index=False)
    metr = pd.read_csv(DATA / "metr_horizons.csv")
    fig_accuracy_by_date(r, out / "accuracy_by_release_date.png")
    dbl, r2, mdbl, mr2 = fig_horizon_by_date(r, metr, out / "horizon_by_release_date.png")
    print(
        f"\nthis suite (excl. Astra): TH50 doubling every {dbl:.0f} days (R²={r2:.2f}); METR 2024+: {mdbl:.0f} days (R²={mr2:.2f})"
    )
    astra = r[r.model == "gpt-6-astra"]
    if len(astra):
        pre = r[r.model != "gpt-6-astra"]
        t = np.array([(x - dt.date(2024, 1, 1)).days for x in pre.release], float)
        b, a = np.polyfit(t, np.log2(pre.th50_min.values.astype(float)), 1)
        ta = (astra.release.iloc[0] - dt.date(2024, 1, 1)).days
        pred = 2 ** (a + b * ta)
        print(
            f"Astra TH50 = {astra.th50_min.iloc[0]:.2f} min vs trend prediction {pred:.2f} min at its release date "
            f"(x{astra.th50_min.iloc[0] / pred:.1f}; equivalent to {np.log2(astra.th50_min.iloc[0] / pred) / b:.0f} days ahead of trend)"
        )
    print("figs ->", out)
