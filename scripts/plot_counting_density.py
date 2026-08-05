import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from geo_vlms.analysis import load_records, score_records
from geo_vlms.tasks import Counting

BIN_EDGES = [1, 2, 3, 6, 11, 21, float("inf")]
BIN_LABELS = ["1", "2", "3-5", "6-10", "11-20", "21+"]
MARK_COLOR = "#2a78d6"
INK = "#1a1a1a"
MUTED = "#767676"

STYLE = {
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.edgecolor": "#c8c8c8",
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
    "axes.labelcolor": MUTED,
    "axes.labelsize": 10,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "grid.color": "#e8e8e8",
    "grid.linewidth": 0.8,
}


def parse_args() -> argparse.Namespace:
    """Parse CLI for density plots."""
    parser = argparse.ArgumentParser(
        description="Plot counting error against ground-truth object count.",
    )
    parser.add_argument(
        "-r",
        "--records",
        type=str,
        required=True,
        help="Path to counting records JSONL file.",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        required=False,
        default=None,
        help="Directory to write figures into. Shows them interactively when omitted.",
    )
    return parser.parse_args()


def prepare_positives(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Filter to valid positive records and derive per-record error columns."""
    pos = metrics_df[
        (metrics_df["expected"] >= 1) & (metrics_df["valid"] == 1.0)
    ].copy()
    pos["predicted"] = pos["expected"] + pos["signed_error"]
    pos["relative_error"] = pos["absolute_error"] / pos["expected"]
    pos["expected_bin"] = pd.cut(
        pos["expected"], bins=BIN_EDGES, labels=BIN_LABELS, right=False
    )
    return pos


def bin_by_expected(pos: pd.DataFrame) -> pd.DataFrame:
    """Aggregate error metrics per expected-count bin."""
    return pos.groupby("expected_bin", observed=False).agg(
        n=("expected", "size"),
        mean_absolute_error=("absolute_error", "mean"),
        mean_relative_error=("relative_error", "mean"),
        exact_match=("exact_match", "mean"),
    )


def plot_scatter(pos: pd.DataFrame, model_name: str) -> Figure:
    """Scatter predicted against expected counts, dot area = record count."""
    pairs = pos.groupby(["expected", "predicted"]).size().reset_index(name="n")

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.grid(True)

    lim = 1.08 * max(pairs["expected"].max(), pairs["predicted"].max())
    ax.plot([0, lim], [0, lim], color="#c8c8c8", lw=1, zorder=1)
    ax.text(
        0.80 * lim,
        0.84 * lim,
        "perfect count",
        rotation=45,
        fontsize=9,
        color=MUTED,
        ha="center",
    )
    sns.scatterplot(
        data=pairs,
        x="expected",
        y="predicted",
        s=25 * pairs["n"],
        color=MARK_COLOR,
        alpha=0.6,
        edgecolor="white",
        linewidth=0.5,
        zorder=2,
        legend=False,
        ax=ax,
    )
    sns.despine(ax=ax)

    worst = pos.loc[pos["absolute_error"].idxmax()]
    ax.annotate(
        f"expected {int(worst['expected'])}, answered {int(worst['predicted'])}",
        xy=(worst["expected"], worst["predicted"]),
        xytext=(-14, -6),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=MUTED,
        arrowprops={"arrowstyle": "-", "color": "#b0b0b0", "lw": 0.8},
    )

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("expected count")
    ax.set_ylabel("predicted count")
    ax.set_title("Predicted vs. expected object count", fontsize=13, pad=28)
    ax.text(
        0,
        1.02,
        f"{model_name} on VHR10 positives · dot area = number of records",
        transform=ax.transAxes,
        fontsize=9.5,
        color=MUTED,
    )
    fig.tight_layout()
    return fig


def plot_bins(by_bin: pd.DataFrame, model_name: str) -> Figure:
    """Bar panels of error and accuracy per expected-count bin."""
    ticks = [
        f"{label}\nn={n}" for label, n in zip(by_bin.index, by_bin["n"], strict=True)
    ]
    panels = [
        ("mean_absolute_error", "Mean absolute error", ".2f"),
        ("mean_relative_error", "Mean relative error (MAE / expected)", ".0%"),
        ("exact_match", "Exact-match rate", ".0%"),
    ]

    data = by_bin.reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4), sharex=True)
    for ax, (column, title, spec) in zip(axes, panels, strict=True):
        sns.barplot(
            data=data, x="expected_bin", y=column, color=MARK_COLOR, width=0.65, ax=ax
        )
        ax.bar_label(
            ax.containers[0],
            labels=[f"{v:{spec}}" for v in data[column]],
            padding=3,
            fontsize=9.5,
            color=INK,
        )
        ax.set_xticks(range(len(data)), ticks)
        ax.set_title(title, fontsize=10.5, color=MUTED, pad=10)
        ax.set_xlabel("")
        ax.set_ylim(0, 1.18 * max(data[column].max(), 1e-9))
        ax.yaxis.set_visible(False)
        sns.despine(ax=ax, left=True)
    axes[2].set_ylim(0, 1.12)

    fig.suptitle(
        "Counting error vs. ground-truth object count",
        x=0.02,
        y=0.985,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.905,
        f"{model_name} on VHR10 positives, grouped by expected count",
        fontsize=9.5,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return fig


def main():
    args = parse_args()
    task = Counting()

    records_path = Path(args.records)
    if not records_path.exists():
        raise FileNotFoundError(f"Records file {records_path} does not exist.")

    records_df = load_records(records_path)
    records_df = records_df.join(pd.json_normalize(records_df["metadata"])).drop(
        columns=["metadata"]
    )

    metrics_df = score_records(
        records_df=records_df, parse=task.parse_response, score=task.score
    )

    pos = prepare_positives(metrics_df)
    by_bin = bin_by_expected(pos)
    print(by_bin)

    model_name = records_df["model_name"].iloc[0]
    sns.set_theme(style="white", rc=STYLE)
    figures = {
        "scatter": plot_scatter(pos, model_name),
        "bins": plot_bins(by_bin, model_name),
    }

    if args.out is None:
        plt.show()
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        out_path = out_dir / f"{records_path.stem}-{name}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
