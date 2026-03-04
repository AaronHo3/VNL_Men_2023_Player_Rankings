#!/usr/bin/env python3
"""Run player archetype clustering on VNL Men 2023 data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# Keep numerical libs compatible with restricted environments.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Allow running script directly from project root without package install.
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vnl_men_2023.player_clustering import load_player_data, run_clustering


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster VNL 2023 players by performance stats.")
    parser.add_argument(
        "--input",
        default="data/raw/VNL2023.csv",
        help="Path to input CSV (default: data/raw/VNL2023.csv)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Optional fixed number of clusters. If omitted, silhouette chooses k.",
    )
    parser.add_argument(
        "--with-tsne",
        action="store_true",
        help="Also compute t-SNE projection (off by default for portability).",
    )
    return parser.parse_args()


def save_scatter(df: pd.DataFrame, x: str, y: str, out_path: Path, title: str) -> None:
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        hue="cluster_label",
        style="Position",
        s=90,
        alpha=0.85,
    )
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.03, 1), loc="upper left")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_interactive_pca(df: pd.DataFrame, out_path: Path) -> bool:
    interactive_df = df[
        [
            "Player",
            "Country",
            "Position",
            "cluster_id",
            "cluster_label",
            "Attack",
            "Block",
            "Serve",
            "Set",
            "Dig",
            "Receive",
            "pca_1",
            "pca_2",
        ]
    ].copy()

    for col in ["Attack", "Block", "Serve", "Set", "Dig", "Receive", "pca_1", "pca_2"]:
        interactive_df[col] = interactive_df[col].round(3)

    data_json = json.dumps(interactive_df.to_dict(orient="records"))
    labels = sorted(interactive_df["cluster_label"].unique().tolist())
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    color_map = {label: palette[i % len(palette)] for i, label in enumerate(labels)}
    color_json = json.dumps(color_map)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VNL Men 2023 Player Clusters (Interactive PCA)</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 20px;
      color: #222;
    }}
    #chart-wrap {{
      position: relative;
      width: 980px;
      max-width: 100%;
    }}
    svg {{
      width: 100%;
      height: auto;
      border: 1px solid #ddd;
      background: #fff;
    }}
    .dot {{
      opacity: 0.9;
      cursor: pointer;
    }}
    .dot:hover {{
      stroke: #111;
      stroke-width: 1.5;
      opacity: 1;
    }}
    #tooltip {{
      position: absolute;
      pointer-events: none;
      background: rgba(28, 28, 28, 0.95);
      color: #fff;
      padding: 8px 10px;
      border-radius: 6px;
      font-size: 12px;
      line-height: 1.35;
      min-width: 220px;
      display: none;
      z-index: 10;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }}
    .legend {{
      margin-top: 10px;
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      font-size: 13px;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .swatch {{
      width: 12px;
      height: 12px;
      border-radius: 50%;
      display: inline-block;
    }}
    .title {{
      font-size: 20px;
      font-weight: 600;
      margin: 0 0 12px 0;
    }}
  </style>
</head>
<body>
  <h1 class="title">VNL Men 2023 Player Clusters (PCA, Interactive)</h1>
  <div id="chart-wrap">
    <svg id="plot" viewBox="0 0 980 700"></svg>
    <div id="tooltip"></div>
  </div>
  <div id="legend" class="legend"></div>
  <script>
    const data = {data_json};
    const colors = {color_json};
    const width = 980, height = 700;
    const margin = {{ top: 30, right: 30, bottom: 50, left: 60 }};
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;
    const xVals = data.map(d => d.pca_1);
    const yVals = data.map(d => d.pca_2);
    const xMin = Math.min(...xVals), xMax = Math.max(...xVals);
    const yMin = Math.min(...yVals), yMax = Math.max(...yVals);
    const padX = (xMax - xMin) * 0.08 || 1;
    const padY = (yMax - yMin) * 0.08 || 1;

    const xScale = x => margin.left + ((x - (xMin - padX)) / ((xMax + padX) - (xMin - padX))) * innerW;
    const yScale = y => margin.top + innerH - ((y - (yMin - padY)) / ((yMax + padY) - (yMin - padY))) * innerH;

    const svg = document.getElementById("plot");
    const tooltip = document.getElementById("tooltip");

    const axisStyle = "stroke:#666;stroke-width:1";
    const textStyle = "font-size:12px;fill:#444";

    function line(x1, y1, x2, y2, style) {{
      const el = document.createElementNS("http://www.w3.org/2000/svg", "line");
      el.setAttribute("x1", x1); el.setAttribute("y1", y1);
      el.setAttribute("x2", x2); el.setAttribute("y2", y2);
      el.setAttribute("style", style);
      return el;
    }}
    function text(x, y, content, style, anchor = "middle") {{
      const el = document.createElementNS("http://www.w3.org/2000/svg", "text");
      el.setAttribute("x", x); el.setAttribute("y", y);
      el.setAttribute("style", style);
      el.setAttribute("text-anchor", anchor);
      el.textContent = content;
      return el;
    }}

    svg.appendChild(line(margin.left, margin.top + innerH, margin.left + innerW, margin.top + innerH, axisStyle));
    svg.appendChild(line(margin.left, margin.top, margin.left, margin.top + innerH, axisStyle));
    svg.appendChild(text(margin.left + innerW / 2, height - 14, "PCA 1", textStyle));
    svg.appendChild(text(16, margin.top + innerH / 2, "PCA 2", textStyle, "start"));

    data.forEach(d => {{
      const cx = xScale(d.pca_1);
      const cy = yScale(d.pca_2);
      const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      dot.setAttribute("class", "dot");
      dot.setAttribute("cx", cx);
      dot.setAttribute("cy", cy);
      dot.setAttribute("r", 5);
      dot.setAttribute("fill", colors[d.cluster_label] || "#555");
      dot.addEventListener("mousemove", (evt) => {{
        tooltip.style.display = "block";
        tooltip.style.left = `${{evt.offsetX + 14}}px`;
        tooltip.style.top = `${{evt.offsetY + 14}}px`;
        tooltip.innerHTML = `
          <strong>${{d.Player}}</strong><br/>
          ${{d.Country}} | Pos: ${{d.Position}}<br/>
          Cluster: ${{d.cluster_label}} (#${{d.cluster_id}})<br/>
          Attack ${{d.Attack}}, Block ${{d.Block}}, Serve ${{d.Serve}}<br/>
          Set ${{d.Set}}, Dig ${{d.Dig}}, Receive ${{d.Receive}}<br/>
          PCA: (${{d.pca_1}}, ${{d.pca_2}})
        `;
      }});
      dot.addEventListener("mouseleave", () => {{
        tooltip.style.display = "none";
      }});
      svg.appendChild(dot);
    }});

    const legend = document.getElementById("legend");
    Object.keys(colors).forEach(label => {{
      const item = document.createElement("div");
      item.className = "legend-item";
      item.innerHTML = `<span class="swatch" style="background:${{colors[label]}}"></span>${{label}}`;
      legend.appendChild(item);
    }});
  </script>
</body>
</html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return True


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    data = load_player_data(str(input_path))
    result = run_clustering(data, k=args.k, with_tsne=args.with_tsne)

    processed_dir = Path("data/processed")
    figures_dir = Path("reports/figures")
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    players_out = processed_dir / "player_clusters.csv"
    profile_out = processed_dir / "cluster_profiles.csv"
    z_profile_out = processed_dir / "cluster_profiles_zscore.csv"
    counts_out = processed_dir / "cluster_counts.csv"
    silhouette_out = processed_dir / "silhouette_by_k.csv"
    rationale_out = processed_dir / "cluster_label_rationale.csv"
    summary_out = processed_dir / "cluster_interpretation.md"

    result.players.to_csv(players_out, index=False)
    result.cluster_profile_raw.to_csv(profile_out)
    result.cluster_profile_z.to_csv(z_profile_out)
    result.silhouette_by_k.to_csv(silhouette_out, index=False)
    result.cluster_label_rationale.to_csv(rationale_out, index=False)
    (
        result.players.groupby(["cluster_id", "cluster_label"], as_index=False)
        .size()
        .rename(columns={"size": "players"})
        .sort_values("cluster_id")
        .to_csv(counts_out, index=False)
    )
    counts_df = pd.read_csv(counts_out)
    with summary_out.open("w", encoding="utf-8") as f:
        f.write("# Cluster Interpretation Summary\n\n")
        f.write(f"- Selected k: **{result.best_k}**\n")
        f.write(f"- Silhouette score at selected k: **{result.silhouette:.4f}**\n\n")
        if not result.silhouette_by_k.empty:
            best_row = result.silhouette_by_k.sort_values("silhouette", ascending=False).iloc[0]
            f.write(
                f"- Best tested k by silhouette: **{int(best_row['k'])}** "
                f"(score **{float(best_row['silhouette']):.4f}**)\n\n"
            )
        f.write("## Cluster Sizes\n\n")
        for row in counts_df.itertuples(index=False):
            f.write(f"- Cluster {row.cluster_id}: {row.cluster_label} ({row.players} players)\n")
        f.write("\n## Label Rules Applied\n\n")
        for row in result.cluster_label_rationale.itertuples(index=False):
            f.write(f"- Cluster {row.cluster_id} -> {row.cluster_label}: {row.rule_reason}\n")

    save_scatter(
        result.players,
        x="pca_1",
        y="pca_2",
        out_path=figures_dir / "clusters_pca.png",
        title="VNL Men 2023 Player Clusters (PCA)",
    )
    interactive_pca_path = figures_dir / "clusters_pca_interactive.html"
    interactive_ok = save_interactive_pca(result.players, interactive_pca_path)
    if args.with_tsne:
        save_scatter(
            result.players,
            x="tsne_1",
            y="tsne_2",
            out_path=figures_dir / "clusters_tsne.png",
            title="VNL Men 2023 Player Clusters (t-SNE)",
        )

    print(f"Done. k={result.best_k}, silhouette={result.silhouette:.4f}")
    print(f"Saved: {players_out}")
    print(f"Saved: {profile_out}")
    print(f"Saved: {z_profile_out}")
    print(f"Saved: {counts_out}")
    print(f"Saved: {silhouette_out}")
    print(f"Saved: {rationale_out}")
    print(f"Saved: {summary_out}")
    print(f"Saved: {figures_dir / 'clusters_pca.png'}")
    if interactive_ok:
        print(f"Saved: {interactive_pca_path}")
    if args.with_tsne:
        print(f"Saved: {figures_dir / 'clusters_tsne.png'}")


if __name__ == "__main__":
    main()
