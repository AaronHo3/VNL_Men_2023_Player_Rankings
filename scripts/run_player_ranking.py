#!/usr/bin/env python3
"""Run custom player ranking model for VNL Men 2023."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vnl_men_2023.player_clustering import load_player_data
from vnl_men_2023.player_ranking import DEFAULT_WEIGHTS, build_player_ranking

POSITION_ORDER = ["OH", "OP", "MB", "S", "L"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank VNL 2023 players with a custom weighted score.")
    parser.add_argument("--input", default="data/raw/VNL2023.csv", help="Input CSV path.")
    parser.add_argument("--top-n", type=int, default=20, help="Top N players for summary outputs.")
    parser.add_argument("--top-n-per-position", type=int, default=5, help="Top N players per position.")
    parser.add_argument("--w-attack", type=float, default=DEFAULT_WEIGHTS["Attack"], help="Weight for Attack.")
    parser.add_argument("--w-block", type=float, default=DEFAULT_WEIGHTS["Block"], help="Weight for Block.")
    parser.add_argument("--w-serve", type=float, default=DEFAULT_WEIGHTS["Serve"], help="Weight for Serve.")
    parser.add_argument("--w-dig", type=float, default=DEFAULT_WEIGHTS["Dig"], help="Weight for Dig.")
    parser.add_argument("--w-receive", type=float, default=DEFAULT_WEIGHTS["Receive"], help="Weight for Receive.")
    return parser.parse_args()


def save_position_plots(position_ranking: pd.DataFrame, figures_dir: Path) -> list[Path]:
    saved_paths: list[Path] = []
    for pos in POSITION_ORDER:
        pos_df = position_ranking[position_ranking["Position"] == pos].copy()
        if pos_df.empty:
            continue

        pos_df = pos_df.sort_values("rank_within_position").copy()
        pos_df["player_country"] = pos_df["Player"] + " (" + pos_df["Country"] + ")"

        fig_height = max(6, min(18, 0.34 * len(pos_df)))
        plt.figure(figsize=(11, fig_height))
        ax = sns.barplot(
            data=pos_df,
            y="player_country",
            x="fair_combined_score",
            color="#2a9d8f",
        )
        plt.title(f"All {len(pos_df)} {pos} Players by Position-Fair Score")
        plt.xlabel("Fair Combined Score")
        plt.ylabel("Player (Country)")

        for i, row in enumerate(pos_df.itertuples(index=False)):
            ax.text(
                float(row.fair_combined_score) + 0.03,
                i,
                f"#{int(row.rank_within_position)} | {float(row.fair_combined_score):.2f}",
                va="center",
                fontsize=9,
            )

        plt.tight_layout()
        out_path = figures_dir / f"all_players_{pos}.png"
        plt.savefig(out_path, dpi=220)
        plt.close()
        saved_paths.append(out_path)
    return saved_paths


def save_combined_position_plot(top_by_position: pd.DataFrame, out_path: Path) -> None:
    plot_df = top_by_position.copy()
    plot_df["player_country"] = plot_df["Player"] + " (" + plot_df["Country"] + ")"
    plot_df["Position"] = pd.Categorical(plot_df["Position"], categories=POSITION_ORDER, ordered=True)
    plot_df = plot_df.sort_values(["Position", "rank_within_position"])

    g = sns.FacetGrid(
        plot_df,
        col="Position",
        col_order=POSITION_ORDER,
        col_wrap=3,
        sharex=False,
        sharey=False,
        height=4.0,
        aspect=1.2,
    )
    g.map_dataframe(
        sns.barplot,
        y="player_country",
        x="fair_combined_score",
        color="#457b9d",
    )

    for ax, pos in zip(g.axes.flatten(), POSITION_ORDER):
        sub = plot_df[plot_df["Position"] == pos].sort_values("rank_within_position")
        if sub.empty:
            continue
        ax.set_title(f"{pos}")
        ax.set_ylabel("Player (Country)")
        ax.set_xlabel("Fair Score")
        for i, row in enumerate(sub.itertuples(index=False)):
            ax.text(
                float(row.fair_combined_score) + 0.03,
                i,
                f"#{int(row.rank_within_position)} | {float(row.fair_combined_score):.2f}",
                va="center",
                fontsize=8,
            )

    g.fig.suptitle("Top Players by Position (Rank + Score)", y=1.02, fontsize=14)
    g.tight_layout()
    g.savefig(out_path, dpi=220)
    plt.close(g.fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    weights = {
        "Attack": args.w_attack,
        "Block": args.w_block,
        "Serve": args.w_serve,
        "Dig": args.w_dig,
        "Receive": args.w_receive,
    }

    df = load_player_data(str(input_path))
    result = build_player_ranking(
        df,
        weights=weights,
        top_n=args.top_n,
        top_n_per_position=args.top_n_per_position,
    )

    processed_dir = Path("data/processed")
    figures_dir = Path("reports/figures")
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    ranking_out = processed_dir / "player_ranking.csv"
    top_out = processed_dir / "top_player_ranking.csv"
    position_ranking_out = processed_dir / "player_ranking_by_position.csv"
    top_by_position_out = processed_dir / "top_players_by_position.csv"
    compare_out = processed_dir / "ranking_vs_attack_leaders.csv"
    position_mix_out = processed_dir / "ranking_top_position_mix.csv"
    weights_out = processed_dir / "ranking_weights.json"
    summary_out = processed_dir / "ranking_summary.md"
    fig_out = figures_dir / "top_player_ranking.png"
    combined_fig_out = figures_dir / "top_players_by_position_combined.png"

    result.ranking.to_csv(ranking_out, index=False)
    result.top_n.to_csv(top_out, index=False)
    (
        result.ranking.sort_values(["Position", "rank_within_position", "fair_combined_score"], ascending=[True, True, False])
        .to_csv(position_ranking_out, index=False)
    )
    result.top_by_position.to_csv(top_by_position_out, index=False)
    result.attack_leader_compare.to_csv(compare_out, index=False)
    result.position_mix_top_n.to_csv(position_mix_out, index=False)

    with weights_out.open("w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)

    with summary_out.open("w", encoding="utf-8") as f:
        f.write("# Player Ranking Summary\n\n")
        f.write("## Baseline Formula\n\n")
        f.write(
            "- score = "
            f"{weights['Attack']:.3f}*Attack + "
            f"{weights['Block']:.3f}*Block + "
            f"{weights['Serve']:.3f}*Serve + "
            f"{weights['Dig']:.3f}*Dig + "
            f"{weights['Receive']:.3f}*Receive\n\n"
        )
        f.write("## Fair Ranking Logic\n\n")
        f.write("- `role_adjusted_score`: z-scored stats within each position + role-specific weights\n")
        f.write("- `position_percentile`: percentile rank within same position\n")
        f.write("- `fair_combined_score = 0.40*custom_score + 0.60*(position_percentile*10)`\n\n")
        f.write("## Position-First Leaderboard (Recommended)\n\n")
        f.write(f"- Full table by role rank: `{position_ranking_out}`\n")
        f.write(f"- Top {args.top_n_per_position} each role: `{top_by_position_out}`\n\n")
        f.write("### Example Top Entries by Position\n\n")
        for row in result.top_by_position.head(10).itertuples(index=False):
            f.write(
                f"- {row.Position} #{row.rank_within_position} {row.Player} ({row.Country}) "
                f"fair={row.fair_combined_score:.3f}, custom={row.custom_score:.3f}, attack={row.Attack:.2f}\n"
            )
        f.write("\n## Global Top-N Mix (Secondary View)\n\n")
        for row in result.position_mix_top_n.itertuples(index=False):
            f.write(f"- {row.Position}: {row.players}\n")
        f.write("\n## Comparison with Attack Leaders\n\n")
        f.write(
            f"- Overlap of top-{result.summary['top_n']} custom ranking with top-{result.summary['top_n']} "
            f"attack-only leaders: {result.summary['overlap_top_n_with_attack_leaders']} "
            f"({result.summary['overlap_ratio']:.1%})\n"
        )

    top_plot = result.top_n.head(min(args.top_n, 20)).copy()
    plt.figure(figsize=(11, 7))
    sns.barplot(data=top_plot, y="Player", x="fair_combined_score", hue="Position")
    plt.title("Top Players by Fair Combined Score")
    plt.xlabel("Fair Combined Score")
    plt.ylabel("Player")
    plt.tight_layout()
    plt.savefig(fig_out, dpi=220)
    plt.close()
    position_plot_paths = save_position_plots(result.ranking, figures_dir)
    save_combined_position_plot(result.top_by_position, combined_fig_out)

    print(f"Saved: {ranking_out}")
    print(f"Saved: {top_out}")
    print(f"Saved: {position_ranking_out}")
    print(f"Saved: {top_by_position_out}")
    print(f"Saved: {compare_out}")
    print(f"Saved: {position_mix_out}")
    print(f"Saved: {weights_out}")
    print(f"Saved: {summary_out}")
    print(f"Saved: {fig_out}")
    print(f"Saved: {combined_fig_out}")
    for p in position_plot_paths:
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
