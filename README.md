# VNL Men 2023 Analysis

Project for exploring the Volleyball Nations League (Men, 2023) Kaggle dataset.

## Project structure

```
vnl_men_2023/
├── data/
│   ├── raw/         # Original dataset files from Kaggle (do not edit)
│   ├── interim/     # Temporary cleaned/merged data
│   └── processed/   # Final modeling/analysis tables
├── notebooks/       # Jupyter notebooks for exploratory analysis
├── reports/
│   └── figures/     # Charts and exported visuals
├── scripts/         # Runnable project scripts
├── src/
│   └── vnl_men_2023/
│       ├── __init__.py
│       └── config.py
├── tests/           # Unit/integration tests
├── .gitignore
├── requirements.txt
└── README.md
```

## Quick start

1. Create env + install dependencies:
   - `make setup`
2. Download dataset into `data/raw/`:
   - `make download-data`
3. Run clustering pipeline:
   - `make cluster`
4. Start Jupyter (optional):
   - `make notebook`
5. Run custom ranking model:
   - `make rank`

The download uses this Kaggle dataset id: `yeganehbavafa/vnl-men-2023`.

If you prefer, run the script directly:
- `bash scripts/download_kaggle_data.sh`

## Project idea: Player Performance Clustering

This repository now includes an end-to-end clustering pipeline to discover player archetypes:

- Standardizes player stats (`Attack`, `Block`, `Serve`, `Set`, `Dig`, `Receive`)
- Picks `k` with silhouette score (or you can pass a fixed `--k`)
- Runs K-means clustering
- Visualizes clusters with both PCA and t-SNE
- Generates cluster labels using explicit rules (offensive/defensive/balanced/serving/setting specialists)

Run:

- `make cluster`
- `make cluster-tsne` (optional t-SNE view)
- `make notebook` (open notebooks)

Outputs:

- `data/processed/player_clusters.csv`
- `data/processed/cluster_profiles.csv`
- `data/processed/cluster_profiles_zscore.csv`
- `data/processed/cluster_counts.csv`
- `data/processed/silhouette_by_k.csv`
- `data/processed/cluster_label_rationale.csv`
- `data/processed/cluster_interpretation.md`
- `reports/figures/clusters_pca.png`
- `reports/figures/clusters_pca_interactive.html` (hover for player details)
- `reports/figures/clusters_tsne.png` (only when running `cluster-tsne`)

## How clusters are determined

1. Feature selection:
   - Uses `Attack`, `Block`, `Serve`, `Set`, `Dig`, `Receive`.
2. Standardization:
   - Each feature is transformed to z-score scale so no single stat dominates due to units/magnitude.
3. Choosing `k`:
   - Tests `k=2..8`, computes silhouette score for each, and picks the best.
   - See `data/processed/silhouette_by_k.csv`.
4. K-means assignment:
   - Players are clustered in full 6-feature standardized space.
5. PCA visualization:
   - The 2D PCA chart is only a projection for visualization; clustering is not done in 2D PCA space.
6. Label assignment (rule-based):
   - `Setting Specialists`: `Set z-score >= 1.0`
   - `Offensive Specialists`: `Attack z + Serve z >= 1.5`
   - `Defensive Specialists`: `Dig z + Receive z + Block z >= 1.5` and offense `< 0.6`
   - `Serving Specialists`: `Serve z >= 1.0` and `Attack z < 0.2`
   - otherwise `Balanced All-Around`
   - Exact cluster-level rule outcomes are exported to `data/processed/cluster_label_rationale.csv`.

## Interpretable analysis checklist

- Open `data/processed/silhouette_by_k.csv` to verify `k` selection quality.
- Open `data/processed/cluster_profiles.csv` for real average stats by cluster.
- Open `data/processed/cluster_profiles_zscore.csv` to see relative strengths/weaknesses.
- Open `data/processed/cluster_label_rationale.csv` to see why each title was assigned.
- Open `data/processed/cluster_interpretation.md` for a plain-language summary.

## Notebook

- `notebooks/01_player_clustering_interpretable.ipynb` provides:
  - silhouette-by-k inspection
  - cluster profile interpretation
  - explicit label rationale table
  - PCA visualization with context

## Project idea: Player Ranking Model

This repository includes a custom ranking pipeline based on a weighted formula.

Default formula:

- `score = 0.40*Attack + 0.20*Block + 0.10*Serve + 0.15*Dig + 0.15*Receive`

Run:

- `make rank`

Recommended interpretation order:

- `data/processed/player_ranking_by_position.csv` (primary, fair role-vs-role ranking)
- `data/processed/top_players_by_position.csv` (quick top list per role)
- `data/processed/top_player_ranking.csv` (secondary global mixed leaderboard)

Optional custom weights:

- `.venv/bin/python scripts/run_player_ranking.py --w-attack 0.45 --w-block 0.20 --w-serve 0.10 --w-dig 0.10 --w-receive 0.15`
- `.venv/bin/python scripts/run_player_ranking.py --top-n 20 --top-n-per-position 5`

Outputs:

- `data/processed/player_ranking.csv` (full ranking)
- `data/processed/player_ranking_by_position.csv` (full table sorted by role rank)
- `data/processed/top_player_ranking.csv` (top N players by fair combined score)
- `data/processed/top_players_by_position.csv` (top players within each role)
- `data/processed/ranking_vs_attack_leaders.csv` (custom rank vs attack-only rank)
- `data/processed/ranking_top_position_mix.csv` (position mix in top N)
- `data/processed/ranking_weights.json` (weights used)
- `data/processed/ranking_summary.md` (plain-language summary)
- `reports/figures/top_player_ranking.png` (top players chart)
- `reports/figures/top_players_by_position_combined.png` (all positions in one faceted chart)
- `reports/figures/all_players_OH.png` (all outside hitters, rank + score + country)
- `reports/figures/all_players_OP.png` (all opposites, rank + score + country)
- `reports/figures/all_players_MB.png` (all middle blockers, rank + score + country)
- `reports/figures/all_players_S.png` (all setters, rank + score + country)
- `reports/figures/all_players_L.png` (all liberos, rank + score + country)

Interpretation notes:

- Why OH/OP dominated before:
  - Attack had the largest weight and many non-attacking roles (S/L) naturally score low in attack.
- Better ranking now:
  - `custom_score` keeps your weighted formula.
- `role_adjusted_score` z-scores players within the same position and applies role-specific weights.
- `fair_combined_score = 0.40*custom_score + 0.60*(position_percentile*10)` is used for final ranking.
- This keeps attack value while giving fair visibility to MB/S/L roles.
- Use `top_players_by_position.csv` when you want a clean, role-by-role leaderboard.
- `ranking_vs_attack_leaders.csv` helps compare your model with attack-only leaderboards (a proxy for official scoring leaders).
- If you later add official VNL leaders CSV, you can compare external leaderboards directly by player name.
