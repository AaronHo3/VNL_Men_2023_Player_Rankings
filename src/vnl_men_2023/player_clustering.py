"""Player clustering pipeline for VNL Men 2023."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


STAT_COLUMNS = ["Attack", "Block", "Serve", "Set", "Dig", "Receive"]


@dataclass
class ClusteringResult:
    players: pd.DataFrame
    cluster_profile_raw: pd.DataFrame
    cluster_profile_z: pd.DataFrame
    silhouette_by_k: pd.DataFrame
    cluster_label_rationale: pd.DataFrame
    best_k: int
    silhouette: float


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_player_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    _validate_columns(df, ["Player", "Country", "Position", *STAT_COLUMNS])
    return df


def standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=0)
    std = np.where(std == 0, 1.0, std)
    return (x - mean) / std, mean, std


def _assign_labels(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    dist = np.linalg.norm(x[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(dist, axis=1)


def _fit_kmeans(
    x: np.ndarray, k: int, n_init: int = 20, max_iter: int = 200, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(random_state)
    best_labels = None
    best_centroids = None
    best_inertia = np.inf

    for _ in range(n_init):
        indices = rng.choice(len(x), size=k, replace=False)
        centroids = x[indices].copy()

        for _ in range(max_iter):
            labels = _assign_labels(x, centroids)
            new_centroids = centroids.copy()
            for cluster in range(k):
                points = x[labels == cluster]
                if len(points) == 0:
                    new_centroids[cluster] = x[rng.integers(0, len(x))]
                else:
                    new_centroids[cluster] = points.mean(axis=0)
            if np.allclose(new_centroids, centroids):
                centroids = new_centroids
                break
            centroids = new_centroids

        labels = _assign_labels(x, centroids)
        inertia = float(np.sum((x - centroids[labels]) ** 2))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centroids = centroids.copy()

    if best_labels is None or best_centroids is None:
        raise RuntimeError("K-means failed to converge.")
    return best_labels, best_centroids, best_inertia


def _silhouette_score(x: np.ndarray, labels: np.ndarray) -> float:
    n = len(x)
    unique = np.unique(labels)
    if len(unique) < 2:
        return -1.0

    dist = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    scores: list[float] = []

    for i in range(n):
        own = labels[i]
        own_mask = labels == own
        own_count = np.sum(own_mask) - 1

        if own_count <= 0:
            continue

        a = float(dist[i, own_mask].sum() / own_count)
        b = np.inf

        for other in unique:
            if other == own:
                continue
            other_mask = labels == other
            if np.any(other_mask):
                b = min(b, float(dist[i, other_mask].mean()))

        denom = max(a, b)
        if denom > 0:
            scores.append((b - a) / denom)

    return float(np.mean(scores)) if scores else -1.0


def choose_k_by_silhouette(x_scaled: np.ndarray, k_min: int = 2, k_max: int = 8) -> tuple[int, float, pd.DataFrame]:
    n_samples = x_scaled.shape[0]
    k_upper = min(k_max, n_samples - 1)
    if k_upper < k_min:
        raise ValueError("Not enough rows to evaluate clustering.")

    best_k = k_min
    best_score = -1.0
    rows: list[dict[str, float | int]] = []

    for k in range(k_min, k_upper + 1):
        labels, _, _ = _fit_kmeans(x_scaled, k=k, n_init=20, random_state=42)
        score = _silhouette_score(x_scaled, labels)
        rows.append({"k": k, "silhouette": score})
        if score > best_score:
            best_score = score
            best_k = k

    silhouette_by_k = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    return best_k, best_score, silhouette_by_k


def compute_pca_2d(x_scaled: np.ndarray) -> np.ndarray:
    x_centered = x_scaled - x_scaled.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x_centered, full_matrices=False)
    components = vt[:2].T
    return x_centered @ components


def label_cluster(cluster_z_row: pd.Series) -> str:
    offense = float(cluster_z_row["Attack"] + cluster_z_row["Serve"])
    defense = float(cluster_z_row["Dig"] + cluster_z_row["Receive"] + cluster_z_row["Block"])
    setting = float(cluster_z_row["Set"])

    if setting >= 1.0:
        return "Setting Specialists"
    if offense >= 1.5:
        return "Offensive Specialists"
    if defense >= 1.5 and offense < 0.6:
        return "Defensive Specialists"
    if cluster_z_row["Serve"] >= 1.0 and cluster_z_row["Attack"] < 0.2:
        return "Serving Specialists"
    return "Balanced All-Around"


def label_reason(cluster_z_row: pd.Series) -> str:
    offense = float(cluster_z_row["Attack"] + cluster_z_row["Serve"])
    defense = float(cluster_z_row["Dig"] + cluster_z_row["Receive"] + cluster_z_row["Block"])
    setting = float(cluster_z_row["Set"])

    if setting >= 1.0:
        return "High Set z-score (>= 1.0)."
    if offense >= 1.5:
        return "High Attack+Serve combined z-score (>= 1.5)."
    if defense >= 1.5 and offense < 0.6:
        return "High defensive z-score and low offense."
    if float(cluster_z_row["Serve"]) >= 1.0 and float(cluster_z_row["Attack"]) < 0.2:
        return "High Serve z-score with relatively low Attack."
    return "No specialist threshold hit; treated as balanced profile."


def run_clustering(df: pd.DataFrame, k: int | None = None, with_tsne: bool = False) -> ClusteringResult:
    stats = df[STAT_COLUMNS].copy()
    x = stats.to_numpy(dtype=float)
    x_scaled, mean, std = standardize(x)

    if k is None:
        best_k, best_score, silhouette_by_k = choose_k_by_silhouette(x_scaled)
    else:
        best_k = k
        best_score = float("nan")
        silhouette_by_k = pd.DataFrame(columns=["k", "silhouette"])

    cluster_id, _, _ = _fit_kmeans(x_scaled, k=best_k, n_init=25, random_state=42)
    pca_2d = compute_pca_2d(x_scaled)

    out = df.copy()
    out["cluster_id"] = cluster_id
    out["pca_1"] = pca_2d[:, 0]
    out["pca_2"] = pca_2d[:, 1]
    out["tsne_1"] = np.nan
    out["tsne_2"] = np.nan

    if with_tsne:
        # Optional because some environments block shared-memory operations used by compiled libs.
        from sklearn.manifold import TSNE  # type: ignore

        perplexity = max(5, min(30, len(df) // 4))
        tsne_2d = TSNE(
            n_components=2,
            random_state=42,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
        ).fit_transform(x_scaled)
        out["tsne_1"] = tsne_2d[:, 0]
        out["tsne_2"] = tsne_2d[:, 1]

    cluster_profile_raw = out.groupby("cluster_id")[STAT_COLUMNS].mean().round(3)
    cluster_profile_z_vals = (cluster_profile_raw.to_numpy() - mean) / std
    cluster_profile_z = pd.DataFrame(cluster_profile_z_vals, index=cluster_profile_raw.index, columns=STAT_COLUMNS).round(3)

    label_map: dict[int, str] = {}
    rationale_rows: list[dict[str, str | int]] = []
    for cluster in cluster_profile_z.index:
        c_row = cluster_profile_z.loc[cluster]
        label = label_cluster(c_row)
        reason = label_reason(c_row)
        label_map[int(cluster)] = label
        rationale_rows.append(
            {
                "cluster_id": int(cluster),
                "cluster_label": label,
                "rule_reason": reason,
            }
        )
    out["cluster_label"] = out["cluster_id"].map(label_map)
    cluster_label_rationale = pd.DataFrame(rationale_rows).sort_values("cluster_id").reset_index(drop=True)

    return ClusteringResult(
        players=out.sort_values(["cluster_id", "Player"]).reset_index(drop=True),
        cluster_profile_raw=cluster_profile_raw,
        cluster_profile_z=cluster_profile_z,
        silhouette_by_k=silhouette_by_k,
        cluster_label_rationale=cluster_label_rationale,
        best_k=best_k,
        silhouette=best_score,
    )
