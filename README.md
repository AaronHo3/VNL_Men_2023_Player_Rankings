# VNL Men 2023 Analysis

Project for exploring the Men's 2023 Volleyball Nations League Kaggle dataset.

## Project structure

```
vnl_men_2023/
├── data/
│   ├── raw/         # Original dataset files from Kaggle
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
6. Run country performance analysis:
   - `make country`
7. Launch interactive dashboard:
   - `make dashboard`

The download uses this Kaggle dataset id: `yeganehbavafa/vnl-men-2023`.

If preferred, run the script directly:
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
- `make notebook`

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
  - `custom_score` keeps weighted formula.
- `role_adjusted_score` z-scores players within the same position and applies role-specific weights.
- `fair_combined_score = 0.40*custom_score + 0.60*(position_percentile*10)` is used for final ranking.
- This keeps attack value while giving fair visibility to MB/S/L roles.
- Use `top_players_by_position.csv` for a clean, role-by-role leaderboard.
- `ranking_vs_attack_leaders.csv` helps compare model with attack-only leaderboards (a proxy for official scoring leaders).
- If official VNL leaders CSV is added, it can compare external leaderboards directly by player name.

## Project idea: Country Performance Analysis

This repository includes a country-level style comparison to answer:

- Which country has the strongest attackers?
- Which countries rely more on defense?
- Which positions dominate certain teams?

Run:

- `make country`

Optional:

- `.venv/bin/python scripts/run_country_performance.py --min-players 6 --top-countries 8`

Outputs:

- `data/processed/country_performance_summary.csv` (country averages + attack/defense indices + z-scores)
- `data/processed/country_player_counts.csv` (players tracked per country)
- `data/processed/country_position_mix.csv` (position share by country + dominant role)
- `data/processed/country_performance_insights.md` (plain-language answers to attack/defense/position questions)
- `reports/figures/country_style_radar.png` (style radar across selected countries)
- `reports/figures/country_style_radar_all_part1.png` (all-country radar, part 1)
- `reports/figures/country_style_radar_all_part2.png` (all-country radar, part 2; auto-split for readability)
- `reports/figures/country_stat_boxplots.png` (attack/dig/receive distribution by country, with country mean markers)
- `reports/figures/country_style_heatmap.png` (country x stat z-score heatmap)

Interpretation notes:

- Radar values are normalized to `[0, 1]` by stat using min-max scaling across eligible countries.
- A radar value of `1.00` means that country has the highest country-average value for that stat in the compared set.
- Boxplots show player-level distributions, not country totals:
  - y-axis is each player's dataset-provided per-match stat value (`Attack`, `Dig`, `Receive`).
  - boxes summarize spread (quartiles/median) across players from a country.
  - black diamond markers indicate country means for each plotted stat.
- A country looking strong in radar but less extreme in boxplots is possible:
  - radar uses country means after normalization,
  - boxplots expose distribution, variance, and outliers.
- Higher boxplot values mean players from that country tend to contribute more per player for that stat.
- This does not by itself prove higher team-level totals, which also depend on lineup usage, opportunities, and number of tracked players.

## Interactive Sports Analytics Dashboard

The project includes an interactive dashboard built with **Python + Plotly + Streamlit** and Tableau-ready exports.

Run:

- `make dashboard`

Dashboard features:

- Player search:
  - search by name and filter by country/position
  - select players from labeled dropdown entries (`Player (Country | Position)`)
  - browse players quickly with `Previous Player` and `Next Player` buttons
  - inspect player profile radar vs position average
- Country comparison:
  - compare selected countries across all core stats
  - attack vs defense index comparison
- Position analytics:
  - stat distribution by position
  - position mean heatmap
  - country-to-position composition treemap
- Top players leaderboard:
  - live ranking with adjustable weights
  - top-N leaderboard table + chart
- Tableau tab:
  - download curated CSV extracts (`players`, `country_summary`, `ranking`) for Tableau

Dashboard interpretation notes:

- Every chart in the app includes an on-screen interpretation caption directly below the visual.
- Player profile radar:
  - selected player is shown in **green** and position average in **red** for contrast.
  - axis categories are fixed to `Attack`, `Block`, `Serve`, `Set`, `Dig`, `Receive`.
  - radial tick values are shown to read magnitude differences.
- Leaderboard rankings use `fair_combined_score` from the ranking pipeline
  (`0.40*custom_score + 0.60*(position_percentile*10)`).

First launch note:

- Streamlit may prompt for optional onboarding email in terminal.
- Press `Enter` to skip, then open the printed local URL (typically `http://localhost:8501`).

Dashboard app file:

- `scripts/dashboard_app.py`
