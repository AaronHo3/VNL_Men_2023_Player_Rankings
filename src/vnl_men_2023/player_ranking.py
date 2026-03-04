"""Custom player ranking models for VNL Men 2023."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


RANK_FEATURES = ["Attack", "Block", "Serve", "Dig", "Receive"]
ALL_STAT_FEATURES = ["Attack", "Block", "Serve", "Set", "Dig", "Receive"]
DEFAULT_WEIGHTS = {
    "Attack": 0.40,
    "Block": 0.20,
    "Serve": 0.10,
    "Dig": 0.15,
    "Receive": 0.15,
}
POSITION_ROLE_WEIGHTS = {
    "OH": {"Attack": 0.35, "Block": 0.15, "Serve": 0.15, "Set": 0.00, "Dig": 0.15, "Receive": 0.20},
    "OP": {"Attack": 0.50, "Block": 0.20, "Serve": 0.15, "Set": 0.00, "Dig": 0.10, "Receive": 0.05},
    "MB": {"Attack": 0.20, "Block": 0.50, "Serve": 0.10, "Set": 0.00, "Dig": 0.10, "Receive": 0.10},
    "S": {"Attack": 0.05, "Block": 0.05, "Serve": 0.10, "Set": 0.60, "Dig": 0.10, "Receive": 0.10},
    "L": {"Attack": 0.00, "Block": 0.00, "Serve": 0.05, "Set": 0.05, "Dig": 0.45, "Receive": 0.45},
}


@dataclass
class RankingResult:
    ranking: pd.DataFrame
    top_n: pd.DataFrame
    top_by_position: pd.DataFrame
    attack_leader_compare: pd.DataFrame
    position_mix_top_n: pd.DataFrame
    summary: dict[str, float | int]


def _validate_weights(weights: dict[str, float]) -> None:
    missing = [f for f in RANK_FEATURES if f not in weights]
    if missing:
        raise ValueError(f"Missing weight(s): {missing}")
    extra = [k for k in weights if k not in RANK_FEATURES]
    if extra:
        raise ValueError(f"Unexpected weight key(s): {extra}")


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("Sum of weights must be > 0.")
    return {k: float(v) / total for k, v in weights.items()}


def _zscore_within_group(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - float(series.mean())) / std


def build_player_ranking(
    df: pd.DataFrame, weights: dict[str, float] | None = None, top_n: int = 20, top_n_per_position: int = 5
) -> RankingResult:
    use_weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    _validate_weights(use_weights)
    use_weights = _normalize_weights(use_weights)

    required = ["Player", "Country", "Position", *ALL_STAT_FEATURES]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    out = df.copy()
    out["score_attack"] = out["Attack"] * use_weights["Attack"]
    out["score_block"] = out["Block"] * use_weights["Block"]
    out["score_serve"] = out["Serve"] * use_weights["Serve"]
    out["score_dig"] = out["Dig"] * use_weights["Dig"]
    out["score_receive"] = out["Receive"] * use_weights["Receive"]
    out["custom_score"] = (
        out["score_attack"]
        + out["score_block"]
        + out["score_serve"]
        + out["score_dig"]
        + out["score_receive"]
    )

    # Position-fair score:
    # 1) z-score each stat within each position group
    # 2) apply role-specific weights (includes Set for setters)
    # 3) rank by weighted role score to compare across all players more fairly
    for stat in ALL_STAT_FEATURES:
        out[f"{stat.lower()}_z_pos"] = out.groupby("Position")[stat].transform(_zscore_within_group)

    role_score = pd.Series(0.0, index=out.index, dtype=float)
    for pos, pos_weights in POSITION_ROLE_WEIGHTS.items():
        mask = out["Position"] == pos
        if not mask.any():
            continue
        pos_part = pd.Series(0.0, index=out.index[mask], dtype=float)
        for stat, w in pos_weights.items():
            pos_part += out.loc[mask, f"{stat.lower()}_z_pos"] * float(w)
        role_score.loc[mask] = pos_part
    out["role_adjusted_score"] = role_score
    out["position_percentile"] = out.groupby("Position")["role_adjusted_score"].rank(pct=True, method="average")

    # Combined score keeps your custom formula but adds role fairness so all positions can surface.
    out["fair_combined_score"] = (0.40 * out["custom_score"]) + (0.60 * (out["position_percentile"] * 10.0))

    ranking = out.sort_values("fair_combined_score", ascending=False).reset_index(drop=True)
    ranking["rank_fair_combined"] = ranking.index + 1
    ranking["rank_custom_score"] = ranking["custom_score"].rank(method="min", ascending=False).astype(int)
    ranking["rank_attack_only"] = ranking["Attack"].rank(method="min", ascending=False).astype(int)
    ranking["rank_role_adjusted"] = ranking["role_adjusted_score"].rank(method="min", ascending=False).astype(int)
    ranking["rank_within_position"] = ranking.groupby("Position")["fair_combined_score"].rank(
        method="min", ascending=False
    ).astype(int)

    # Compare with "official-leader-like" perspective: attack-only leaders.
    compare = (
        ranking.loc[
            :,
            [
                "Player",
                "Country",
                "Position",
                "Attack",
                "custom_score",
                "role_adjusted_score",
                "fair_combined_score",
                "rank_fair_combined",
                "rank_custom_score",
                "rank_attack_only",
            ],
        ]
        .sort_values(["rank_attack_only", "rank_fair_combined"])
        .reset_index(drop=True)
    )

    top_df = ranking.head(top_n).copy()
    top_position_mix = top_df["Position"].value_counts().rename_axis("Position").reset_index(name="players")
    top_by_position = (
        ranking.sort_values(["Position", "rank_within_position"])
        .groupby("Position", as_index=False, group_keys=False)
        .head(top_n_per_position)
        .reset_index(drop=True)
    )

    top_attack_players = set(ranking.sort_values("Attack", ascending=False).head(top_n)["Player"])
    top_custom_players = set(top_df["Player"])
    overlap_n = len(top_attack_players & top_custom_players)

    summary = {
        "players_total": int(len(ranking)),
        "top_n": int(top_n),
        "overlap_top_n_with_attack_leaders": int(overlap_n),
        "overlap_ratio": float(overlap_n / max(1, top_n)),
        "positions_in_top_n": int(top_position_mix["Position"].nunique()),
        "top_n_per_position": int(top_n_per_position),
        "top_by_position_rows": int(len(top_by_position)),
    }

    return RankingResult(
        ranking=ranking,
        top_n=top_df,
        top_by_position=top_by_position,
        attack_leader_compare=compare,
        position_mix_top_n=top_position_mix,
        summary=summary,
    )
