#!/usr/bin/env python3
"""Run country-level performance style analysis for VNL Men 2023."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Constrain thread counts for reproducible behavior across local environments
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SRC_DIR = PROJECT_ROOT / "src"
# Enable local package imports when running from repository root
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vnl_men_2023.player_clustering import load_player_data

STAT_COLUMNS = ["Attack", "Block", "Serve", "Set", "Dig", "Receive"]
POSITION_ORDER = ["OH", "OP", "MB", "S", "L"]


def parse_args() -> argparse.Namespace:
    # CLI parameters control input location and country inclusion thresholds
    parser = argparse.ArgumentParser(description="Country-level style analysis for VNL Men 2023.")
    parser.add_argument("--input", default="data/raw/VNL2023.csv", help="Input CSV path.")
    parser.add_argument(
        "--min-players",
        type=int,
        default=6,
        help="Minimum players per country to include in country-level comparisons.",
    )
    parser.add_argument(
        "--top-countries",
        type=int,
        default=8,
        help="Number of countries to include in radar/boxplot views.",
    )
    return parser.parse_args()


def _zscore_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Standard score transform used for cross-country comparability
    std = df.std(ddof=0).replace(0, 1.0)
    return (df - df.mean()) / std


def build_country_summary(df: pd.DataFrame, min_players: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Player counts determine which countries have enough representation for stable comparisons
    country_counts = (
        df.groupby("Country", as_index=False)
        .size()
        .rename(columns={"size": "players"})
        .sort_values("players", ascending=False)
    )

    eligible = set(country_counts[country_counts["players"] >= min_players]["Country"])
    if not eligible:
        raise ValueError("No countries meet --min-players threshold. Lower --min-players and retry.")

    # Core country profile table built from mean stat values
    country_means = (
        df[df["Country"].isin(eligible)]
        .groupby("Country", as_index=False)[STAT_COLUMNS]
        .mean()
        .merge(country_counts, on="Country", how="left")
    )

    # Composite style indices used for attack/defense comparisons
    country_means["attack_index"] = 0.8 * country_means["Attack"] + 0.2 * country_means["Serve"]
    country_means["defense_index"] = (country_means["Dig"] + country_means["Receive"] + country_means["Block"]) / 3.0

    # Z-score columns make country-level indices and raw stats directly comparable on one scale
    z = _zscore_frame(country_means[["attack_index", "defense_index"]])
    country_means["attack_index_z"] = z["attack_index"]
    country_means["defense_index_z"] = z["defense_index"]
    country_means["defense_minus_attack_z"] = country_means["defense_index_z"] - country_means["attack_index_z"]

    stat_z = _zscore_frame(country_means[STAT_COLUMNS])
    stat_z.columns = [f"{c}_z" for c in stat_z.columns]
    country_summary = pd.concat([country_means, stat_z], axis=1).sort_values("attack_index", ascending=False)

    return country_summary, country_counts


def build_position_mix(df: pd.DataFrame, eligible_countries: set[str]) -> pd.DataFrame:
    # Country-position distribution supports dominance analysis by role share
    mix = (
        df[df["Country"].isin(eligible_countries)]
        .groupby(["Country", "Position"], as_index=False)
        .size()
        .rename(columns={"size": "players"})
    )

    totals = mix.groupby("Country", as_index=False)["players"].sum().rename(columns={"players": "country_players"})
    mix = mix.merge(totals, on="Country", how="left")
    mix["share"] = mix["players"] / mix["country_players"]

    mix["Position"] = pd.Categorical(mix["Position"], categories=POSITION_ORDER, ordered=True)
    mix = mix.sort_values(["Country", "share", "Position"], ascending=[True, False, True]).reset_index(drop=True)

    dominant = (
        mix.sort_values(["Country", "share"], ascending=[True, False])
        .groupby("Country", as_index=False)
        .first()[["Country", "Position", "share", "country_players"]]
        .rename(columns={"Position": "dominant_position", "share": "dominant_share"})
    )

    return mix.merge(dominant, on=["Country", "country_players"], how="left")


def save_radar(
    country_summary: pd.DataFrame,
    selected_countries: list[str],
    out_path: Path,
    title: str,
) -> None:
    # Radar charts are rendered as small multiples (one country per subplot) for readability
    radar_stats = ["Attack", "Block", "Serve", "Set", "Dig", "Receive"]
    plot_df = country_summary[country_summary["Country"].isin(selected_countries)].copy()
    if plot_df.empty:
        return

    labels = radar_stats
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    # Global min-max normalization keeps panels comparable across split radar files
    min_vals = country_summary[radar_stats].min()
    max_vals = country_summary[radar_stats].max()
    denom = (max_vals - min_vals).replace(0, 1.0)
    norm = (plot_df[radar_stats] - min_vals) / denom

    n = len(plot_df)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.2 * ncols, 4.0 * nrows),
        subplot_kw={"projection": "polar"},
    )
    axes_arr = np.atleast_1d(axes).reshape(-1)
    palette = sns.color_palette("tab20", n_colors=n)

    for i, (_, row) in enumerate(plot_df.reset_index(drop=True).iterrows()):
        ax = axes_arr[i]
        values = norm.iloc[i].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2.2, color=palette[i])
        ax.fill(angles, values, alpha=0.16, color=palette[i])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.35)
        ax.set_title(row["Country"], fontsize=10, pad=12)

    for j in range(n, len(axes_arr)):
        fig.delaxes(axes_arr[j])

    fig.suptitle(title, y=1.01, fontsize=14)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_boxplots(df: pd.DataFrame, out_path: Path) -> None:
    # Boxplots expose player-level distribution spread by country and stat
    subset = df.copy()
    if subset.empty:
        return

    plot_df = subset.melt(
        id_vars=["Country"],
        value_vars=["Attack", "Dig", "Receive"],
        var_name="Stat",
        value_name="Value",
    )
    order = (
        subset.groupby("Country", as_index=False)["Attack"]
        .median()
        .sort_values("Attack", ascending=False)["Country"]
        .tolist()
    )

    plt.figure(figsize=(16, 8))
    ax = sns.boxplot(data=plot_df, x="Country", y="Value", hue="Stat", order=order)

    # Mean markers make central tendency explicit on top of distribution boxes
    mean_df = (
        subset.groupby("Country", as_index=False)[["Attack", "Dig", "Receive"]]
        .mean()
        .melt(id_vars=["Country"], var_name="Stat", value_name="Mean")
    )
    sns.pointplot(
        data=mean_df,
        x="Country",
        y="Mean",
        hue="Stat",
        order=order,
        dodge=0.4,
        markers="D",
        linestyle="none",
        markersize=4.5,
        errorbar=None,
        palette=["#111111", "#111111", "#111111"],
        ax=ax,
    )
    plt.title("Attack vs Defense Distribution by Country (All Countries)")
    plt.xlabel("Country")
    plt.ylabel("Per-Match Stat Value")
    plt.xticks(rotation=35, ha="right")
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) >= 6:
        box_handles = handles[:3]
        box_labels = labels[:3]
        mean_handles = handles[3:6]
        mean_labels = [f"{lbl} mean" for lbl in labels[3:6]]
        ax.legend(
            box_handles + mean_handles,
            box_labels + mean_labels,
            title="Stat",
            ncol=3,
            loc="upper right",
        )
    else:
        ax.legend(title="Stat", ncol=3, loc="upper right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_heatmap(country_summary: pd.DataFrame, out_path: Path) -> None:
    # Heatmap summarizes relative strengths via country z-scores
    z_cols = ["Attack_z", "Block_z", "Serve_z", "Set_z", "Dig_z", "Receive_z"]
    hm = (
        country_summary[["Country", *z_cols]]
        .set_index("Country")
        .sort_values("Attack_z", ascending=False)
        .rename(columns={c: c.replace("_z", "") for c in z_cols})
    )

    plt.figure(figsize=(10, max(6, 0.45 * len(hm))))
    sns.heatmap(hm, cmap="RdBu_r", center=0, annot=True, fmt=".2f", linewidths=0.3)
    plt.title("Country Stat Heatmap (Z-score Across Countries)")
    plt.xlabel("Stat")
    plt.ylabel("Country")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def write_summary(
    country_summary: pd.DataFrame,
    position_mix: pd.DataFrame,
    out_path: Path,
    radar_path: Path,
    all_radar_paths: list[Path],
    boxplot_path: Path,
    heatmap_path: Path,
) -> None:
    # Markdown report captures headline findings and links generated visual files
    strongest = country_summary.sort_values("attack_index", ascending=False).head(5)
    defensive = country_summary.sort_values("defense_minus_attack_z", ascending=False).head(5)

    dominant = (
        position_mix[["Country", "dominant_position", "dominant_share", "country_players"]]
        .drop_duplicates()
        .sort_values(["dominant_share", "country_players"], ascending=[False, False])
    )

    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Country Performance Analysis\n\n")
        f.write("## Strongest Attackers\n\n")
        f.write("Ranked by `attack_index = 0.8*Attack + 0.2*Serve`.\n\n")
        for i, row in enumerate(strongest.itertuples(index=False), start=1):
            f.write(
                f"{i}. {row.Country}: attack_index={row.attack_index:.2f} "
                f"(Attack={row.Attack:.2f}, Serve={row.Serve:.2f}, players={int(row.players)})\n"
            )

        f.write("\n## Defense-Reliant Countries\n\n")
        f.write("Ranked by `defense_minus_attack_z` (higher means relatively more defense-oriented style).\n\n")
        for i, row in enumerate(defensive.itertuples(index=False), start=1):
            f.write(
                f"{i}. {row.Country}: defense_minus_attack_z={row.defense_minus_attack_z:.2f} "
                f"(Dig={row.Dig:.2f}, Receive={row.Receive:.2f}, Block={row.Block:.2f})\n"
            )

        f.write("\n## Position Dominance by Team\n\n")
        for row in dominant.itertuples(index=False):
            f.write(
                f"- {row.Country}: {row.dominant_position} dominates "
                f"({row.dominant_share:.1%} of {int(row.country_players)} tracked players)\n"
            )

        f.write("\n## Visualizations\n\n")
        f.write(f"- Radar chart (selected countries): `{radar_path}`\n")
        for p in all_radar_paths:
            f.write(f"- Radar chart (all countries): `{p}`\n")
        f.write(f"- Boxplots: `{boxplot_path}`\n")
        f.write(f"- Heatmap: `{heatmap_path}`\n")


def main() -> None:
    # Pipeline flow: load, compute summaries, render visuals, export tables/report
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = load_player_data(str(input_path))
    country_summary, country_counts = build_country_summary(df, min_players=args.min_players)

    eligible_countries = set(country_summary["Country"])
    position_mix = build_position_mix(df, eligible_countries)

    # Preferred countries are force-included when available for consistency with example narratives
    preferred = ["Japan", "Italy", "Brazil"]
    selected = country_counts[country_counts["Country"].isin(eligible_countries)].copy()
    selected = selected.sort_values("players", ascending=False)
    selected_countries = selected.head(args.top_countries)["Country"].tolist()

    for country in preferred:
        if country in eligible_countries and country not in selected_countries:
            selected_countries.append(country)

    selected_countries = selected_countries[: max(args.top_countries, len(preferred))]

    processed_dir = Path("data/processed")
    figures_dir = Path("reports/figures")
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = processed_dir / "country_performance_summary.csv"
    counts_csv = processed_dir / "country_player_counts.csv"
    position_mix_csv = processed_dir / "country_position_mix.csv"
    insight_md = processed_dir / "country_performance_insights.md"

    radar_path = figures_dir / "country_style_radar.png"
    all_radar_paths: list[Path] = []
    boxplot_path = figures_dir / "country_stat_boxplots.png"
    heatmap_path = figures_dir / "country_style_heatmap.png"

    country_summary.sort_values("attack_index", ascending=False).to_csv(summary_csv, index=False)
    country_counts.to_csv(counts_csv, index=False)
    position_mix.sort_values(["Country", "share"], ascending=[True, False]).to_csv(position_mix_csv, index=False)

    save_radar(
        country_summary,
        selected_countries,
        radar_path,
        title="Country Style Radar (Selected Countries, Small Multiples)",
    )

    # All-country radar is split into parts so subplot labels remain legible
    all_countries = country_summary.sort_values("attack_index", ascending=False)["Country"].tolist()
    per_plot = 8
    total_parts = int(np.ceil(len(all_countries) / per_plot))
    for part_idx, start in enumerate(range(0, len(all_countries), per_plot), start=1):
        chunk = all_countries[start : start + per_plot]
        out = figures_dir / f"country_style_radar_all_part{part_idx}.png"
        save_radar(
            country_summary,
            chunk,
            out,
            title=f"Country Style Radar (All Countries, Part {part_idx}/{total_parts})",
        )
        all_radar_paths.append(out)
    save_boxplots(df, boxplot_path)
    save_heatmap(country_summary, heatmap_path)

    write_summary(
        country_summary=country_summary,
        position_mix=position_mix,
        out_path=insight_md,
        radar_path=radar_path,
        all_radar_paths=all_radar_paths,
        boxplot_path=boxplot_path,
        heatmap_path=heatmap_path,
    )

    print(f"Saved: {summary_csv}")
    print(f"Saved: {counts_csv}")
    print(f"Saved: {position_mix_csv}")
    print(f"Saved: {insight_md}")
    print(f"Saved: {radar_path}")
    for p in all_radar_paths:
        print(f"Saved: {p}")
    print(f"Saved: {boxplot_path}")
    print(f"Saved: {heatmap_path}")


if __name__ == "__main__":
    main()
