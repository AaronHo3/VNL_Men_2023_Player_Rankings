#!/usr/bin/env python3
"""Interactive Sports Analytics Dashboard for VNL Men 2023."""

from __future__ import annotations

import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Constrain numeric library threading for stable local execution
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

SRC_DIR = PROJECT_ROOT / "src"
# Allow direct imports from the local package when running as a script
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vnl_men_2023.player_clustering import load_player_data
from vnl_men_2023.player_ranking import DEFAULT_WEIGHTS, build_player_ranking

STAT_COLUMNS = ["Attack", "Block", "Serve", "Set", "Dig", "Receive"]
POSITION_ORDER = ["OH", "OP", "MB", "S", "L"]


@st.cache_data(show_spinner=False)
def load_base_data(input_path: str) -> pd.DataFrame:
    # Cache raw data loading to avoid repeated disk reads during widget updates
    return load_player_data(input_path)


@st.cache_data(show_spinner=False)
def compute_ranking(df: pd.DataFrame, w_attack: float, w_block: float, w_serve: float, w_dig: float, w_receive: float):
    # Convert slider values into model weights, then run ranking pipeline once per weight set
    weights = {
        "Attack": w_attack,
        "Block": w_block,
        "Serve": w_serve,
        "Dig": w_dig,
        "Receive": w_receive,
    }
    return build_player_ranking(df, weights=weights, top_n=30, top_n_per_position=10)


def to_csv_download(df: pd.DataFrame) -> bytes:
    # Standard helper used by download buttons
    return df.to_csv(index=False).encode("utf-8")


def render_player_search(df: pd.DataFrame) -> None:
    st.subheader("Player Search")
    st.caption("Filter first, then select a player from a labeled list. Use Previous/Next for fast browsing.")

    # Top-row filters narrow the candidate list before profile selection
    c1, c2, c3 = st.columns([2, 1, 1])
    query = c1.text_input("Search player name", placeholder="e.g., Ishikawa, Giannelli, Kurek")
    country_filter = c2.selectbox("Country", ["All"] + sorted(df["Country"].unique().tolist()))
    pos_filter = c3.selectbox("Position", ["All"] + POSITION_ORDER)

    # Apply each filter independently and keep the resulting player subset
    filtered = df.copy()
    if query.strip():
        filtered = filtered[filtered["Player"].str.contains(query.strip(), case=False, na=False)]
    if country_filter != "All":
        filtered = filtered[filtered["Country"] == country_filter]
    if pos_filter != "All":
        filtered = filtered[filtered["Position"] == pos_filter]

    st.caption(f"Matches: {len(filtered)} players")
    st.dataframe(filtered[["Player", "Country", "Position", *STAT_COLUMNS]], use_container_width=True, hide_index=True)

    if filtered.empty:
        return

    # Build stable labeled options so similarly named players remain distinguishable
    selection_df = filtered[["Player", "Country", "Position"]].drop_duplicates().sort_values(["Player", "Country"])
    selection_options = selection_df.apply(
        lambda r: f"{r['Player']} ({r['Country']} | {r['Position']})",
        axis=1,
    ).tolist()
    option_to_row = dict(zip(selection_options, selection_df.to_dict(orient="records")))

    # Store selection index in session state so buttons and dropdown stay synchronized
    if "player_option_idx" not in st.session_state:
        st.session_state["player_option_idx"] = 0
    st.session_state["player_option_idx"] = min(st.session_state["player_option_idx"], len(selection_options) - 1)

    # Navigation controls update index first, then trigger a rerun
    nav_prev, nav_next = st.columns([1, 1])
    if nav_prev.button("Previous Player", use_container_width=True):
        st.session_state["player_option_idx"] = (st.session_state["player_option_idx"] - 1) % len(selection_options)
        st.rerun()
    if nav_next.button("Next Player", use_container_width=True):
        st.session_state["player_option_idx"] = (st.session_state["player_option_idx"] + 1) % len(selection_options)
        st.rerun()

    selected_option = st.selectbox(
        "Select player for profile",
        options=selection_options,
        index=st.session_state["player_option_idx"],
    )
    st.session_state["player_option_idx"] = selection_options.index(selected_option)

    selected_row = option_to_row[selected_option]
    selected_name = str(selected_row["Player"])
    player_row = filtered[
        (filtered["Player"] == selected_row["Player"])
        & (filtered["Country"] == selected_row["Country"])
        & (filtered["Position"] == selected_row["Position"])
    ].iloc[0]

    # Compare selected player against the average profile of the same position
    role_df = df[df["Position"] == player_row["Position"]]
    role_means = role_df[STAT_COLUMNS].mean()

    profile_df = pd.DataFrame(
        {
            "Stat": STAT_COLUMNS,
            "Player": [float(player_row[s]) for s in STAT_COLUMNS],
            "Position Avg": [float(role_means[s]) for s in STAT_COLUMNS],
        }
    )

    # Radar chart overlays selected player and position-average baseline
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=profile_df["Player"],
            theta=profile_df["Stat"],
            fill="toself",
            mode="lines+markers",
            name=f"{selected_name}",
            line=dict(color="#22c55e", width=3),
            marker=dict(color="#22c55e", size=7),
            fillcolor="rgba(34, 197, 94, 0.28)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=profile_df["Position Avg"],
            theta=profile_df["Stat"],
            fill="toself",
            mode="lines+markers",
            name=f"{player_row['Position']} Avg",
            line=dict(color="#ef4444", width=3),
            marker=dict(color="#ef4444", size=7),
            fillcolor="rgba(239, 68, 68, 0.22)",
        )
    )
    # Dynamic radial ticks scale to the selected profile range
    r_max = max(float(profile_df["Player"].max()), float(profile_df["Position Avg"].max()))
    tick_step = max(1.0, round(r_max / 5.0, 1))
    tick_vals = [round(i * tick_step, 1) for i in range(0, 6)]
    axis_max = tick_vals[-1] if tick_vals[-1] > 0 else 1.0

    fig.update_layout(
        title=f"Player Profile vs Position Average: {selected_name}",
        height=760,
        polar=dict(
            bgcolor="#f8fafc",
            radialaxis=dict(
                visible=True,
                range=[0, axis_max],
                tickvals=tick_vals,
                ticktext=[f"{v:.1f}" for v in tick_vals],
                tickfont=dict(size=16, color="#000000"),
                tickangle=0,
                gridcolor="#6b7280",
                gridwidth=1.4,
                linecolor="#e5e7eb",
                linewidth=2.0,
            ),
            angularaxis=dict(
                categoryorder="array",
                categoryarray=STAT_COLUMNS,
                tickmode="array",
                tickvals=STAT_COLUMNS,
                ticktext=STAT_COLUMNS,
                tickfont=dict(size=18, color="#ffffff"),
                gridcolor="#9ca3af",
                gridwidth=1.1,
                linecolor="#e5e7eb",
                linewidth=1.8,
            ),
        ),
        title_font=dict(size=22, color="#ffffff"),
        legend=dict(font=dict(size=14, color="#ffffff")),
        margin=dict(l=70, r=70, t=90, b=70),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Interpretation: each spoke is a stat. The player polygon is compared against the average polygon for that "
        "position. Larger outward distance indicates a higher value for that metric."
    )


def render_country_comparison(df: pd.DataFrame) -> None:
    st.subheader("Country Comparison")

    # Country summary table is reused across visuals in this tab
    country_summary = df.groupby("Country", as_index=False)[STAT_COLUMNS].mean()
    country_summary["attack_index"] = 0.8 * country_summary["Attack"] + 0.2 * country_summary["Serve"]
    country_summary["defense_index"] = (
        country_summary["Dig"] + country_summary["Receive"] + country_summary["Block"]
    ) / 3.0

    selected = st.multiselect(
        "Compare countries",
        options=country_summary.sort_values("attack_index", ascending=False)["Country"].tolist(),
        default=country_summary.sort_values("attack_index", ascending=False)["Country"].head(4).tolist(),
    )

    if not selected:
        st.info("Select at least one country.")
        return

    comp = country_summary[country_summary["Country"].isin(selected)].copy()

    # Long format enables grouped bar charts by stat and country
    long_df = comp.melt(
        id_vars=["Country"],
        value_vars=STAT_COLUMNS,
        var_name="Stat",
        value_name="Value",
    )

    bar = px.bar(
        long_df,
        x="Stat",
        y="Value",
        color="Country",
        barmode="group",
        title="Country Average Stats",
    )
    st.plotly_chart(bar, use_container_width=True)
    st.caption(
        "Interpretation: grouped bars compare country-average values for each stat. Compare countries within the same "
        "stat column, not across different stat types."
    )

    # Index chart separates offense-oriented and defense-oriented country profiles
    idx_long = comp.melt(
        id_vars=["Country"],
        value_vars=["attack_index", "defense_index"],
        var_name="Index",
        value_name="Value",
    )
    idx_fig = px.bar(idx_long, x="Country", y="Value", color="Index", barmode="group", title="Attack vs Defense Index")
    st.plotly_chart(idx_fig, use_container_width=True)
    st.caption(
        "Interpretation: attack index = 0.8*Attack + 0.2*Serve. Defense index is the mean of Dig, Receive, and Block. "
        "Higher bars indicate stronger average profile on that index."
    )

    st.dataframe(comp.sort_values("attack_index", ascending=False), use_container_width=True, hide_index=True)


def render_position_analytics(df: pd.DataFrame) -> None:
    st.subheader("Position Analytics")

    stat = st.selectbox("Select stat", STAT_COLUMNS, index=0)
    pos_order_present = [p for p in POSITION_ORDER if p in df["Position"].unique()]

    box = px.box(
        df,
        x="Position",
        y=stat,
        color="Position",
        category_orders={"Position": pos_order_present},
        points="outliers",
        title=f"{stat} Distribution by Position",
    )
    st.plotly_chart(box, use_container_width=True)
    st.caption(
        "Interpretation: each box shows median and quartiles for the selected stat by position. Points indicate outliers."
    )

    # Position-level means feed the heatmap for cross-stat comparison
    pos_means = (
        df.groupby("Position", as_index=False)[STAT_COLUMNS]
        .mean()
        .set_index("Position")
        .reindex(pos_order_present)
        .reset_index()
    )
    hm = px.imshow(
        pos_means.set_index("Position")[STAT_COLUMNS],
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Average Stats by Position",
    )
    st.plotly_chart(hm, use_container_width=True)
    st.caption(
        "Interpretation: heatmap cells are position-level mean values. Warmer colors indicate higher averages."
    )

    # Country-position composition supports roster-shape exploration
    country_pos = (
        df.groupby(["Country", "Position"], as_index=False)
        .size()
        .rename(columns={"size": "players"})
    )
    totals = country_pos.groupby("Country", as_index=False)["players"].sum().rename(columns={"players": "country_players"})
    country_pos = country_pos.merge(totals, on="Country", how="left")
    country_pos["share"] = country_pos["players"] / country_pos["country_players"]

    treemap = px.treemap(
        country_pos,
        path=["Country", "Position"],
        values="players",
        color="share",
        color_continuous_scale="Blues",
        title="Country -> Position Composition",
    )
    st.plotly_chart(treemap, use_container_width=True)
    st.caption(
        "Interpretation: each rectangle size is player count. Color intensity represents within-country position share."
    )


def render_leaderboard(df: pd.DataFrame) -> None:
    st.subheader("Top Players Leaderboard")

    # Slider controls expose ranking sensitivity to weight changes
    st.caption("Adjust weights to recalculate the fair-combined ranking live.")
    c1, c2, c3, c4, c5 = st.columns(5)
    w_attack = c1.slider("Attack", 0.0, 1.0, float(DEFAULT_WEIGHTS["Attack"]), 0.05)
    w_block = c2.slider("Block", 0.0, 1.0, float(DEFAULT_WEIGHTS["Block"]), 0.05)
    w_serve = c3.slider("Serve", 0.0, 1.0, float(DEFAULT_WEIGHTS["Serve"]), 0.05)
    w_dig = c4.slider("Dig", 0.0, 1.0, float(DEFAULT_WEIGHTS["Dig"]), 0.05)
    w_receive = c5.slider("Receive", 0.0, 1.0, float(DEFAULT_WEIGHTS["Receive"]), 0.05)

    result = compute_ranking(df, w_attack, w_block, w_serve, w_dig, w_receive)

    top_n = st.slider("Top N players", min_value=5, max_value=50, value=20, step=5)
    table_cols = [
        "rank_fair_combined",
        "Player",
        "Country",
        "Position",
        "fair_combined_score",
        "custom_score",
        "Attack",
        "Block",
        "Serve",
        "Dig",
        "Receive",
    ]
    top_df = result.ranking.head(top_n)[table_cols].copy()

    st.dataframe(top_df, use_container_width=True, hide_index=True)
    st.caption(
        "Interpretation: ranks are based on fair_combined_score, which blends weighted custom score with "
        "position-fair adjustment."
    )

    bar = px.bar(
        top_df.sort_values("fair_combined_score", ascending=True),
        x="fair_combined_score",
        y="Player",
        color="Position",
        orientation="h",
        title=f"Top {top_n} by Fair Combined Score",
    )
    st.plotly_chart(bar, use_container_width=True)
    st.caption(
        "Interpretation: longer bars indicate higher fair_combined_score. Colors show positional representation in the top-N."
    )


def render_tableau_exports(df: pd.DataFrame) -> None:
    st.subheader("Tableau Exports")
    st.caption("Download curated CSV extracts and connect them directly in Tableau.")

    # Export tables are computed from the same base dataset used in interactive tabs
    country_summary = df.groupby("Country", as_index=False)[STAT_COLUMNS].mean()
    country_summary["attack_index"] = 0.8 * country_summary["Attack"] + 0.2 * country_summary["Serve"]
    country_summary["defense_index"] = (
        country_summary["Dig"] + country_summary["Receive"] + country_summary["Block"]
    ) / 3.0

    result = compute_ranking(
        df,
        float(DEFAULT_WEIGHTS["Attack"]),
        float(DEFAULT_WEIGHTS["Block"]),
        float(DEFAULT_WEIGHTS["Serve"]),
        float(DEFAULT_WEIGHTS["Dig"]),
        float(DEFAULT_WEIGHTS["Receive"]),
    )

    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "Download players CSV",
        data=to_csv_download(df),
        file_name="dashboard_players.csv",
        mime="text/csv",
    )
    c2.download_button(
        "Download country summary CSV",
        data=to_csv_download(country_summary),
        file_name="dashboard_country_summary.csv",
        mime="text/csv",
    )
    c3.download_button(
        "Download ranking CSV",
        data=to_csv_download(result.ranking),
        file_name="dashboard_player_ranking.csv",
        mime="text/csv",
    )


def main() -> None:
    # Page scaffold: global settings, input selection, and tab routing
    st.set_page_config(page_title="VNL Men 2023 Dashboard", layout="wide")
    st.title("VNL Men 2023 Sports Analytics Dashboard")
    st.markdown("Interactive analysis with **Streamlit + Plotly**. Use the tabs to explore players, countries, positions, and rankings.")

    default_input = str(PROJECT_ROOT / "data/raw/VNL2023.csv")
    input_path = st.sidebar.text_input("Input CSV path", value=default_input)

    if not Path(input_path).exists():
        st.error(f"Input CSV not found: {input_path}")
        st.stop()

    df = load_base_data(input_path)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Player Search",
            "Country Comparison",
            "Position Analytics",
            "Leaderboard",
            "Tableau",
        ]
    )

    with tab1:
        render_player_search(df)
    with tab2:
        render_country_comparison(df)
    with tab3:
        render_position_analytics(df)
    with tab4:
        render_leaderboard(df)
    with tab5:
        render_tableau_exports(df)


if __name__ == "__main__":
    main()
