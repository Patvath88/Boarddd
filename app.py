# app.py

"""
NBA Prop Predictor — Pro Tier (Stability Patch + TARGET Fix + Metric Card UI)
Performance-optimized: single-pass feature engineering, tiny-key caching, parallel per-stat training, fast trend.
"""

from __future__ import annotations

import os
import datetime
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import data_fetching as dfetch
from models import ModelManager


# =============================================================================
# CONFIG / CONSTANTS
# =============================================================================

PROP_MAP = {
    "Points": "PTS",
    "Rebounds": "REB",
    "Assists": "AST",
    "PRA": ["PTS", "REB", "AST"],
    "PR": ["PTS", "REB"],
    "PA": ["PTS", "AST"],
    "RA": ["REB", "AST"],
    "3PM": "FG3M",
    "Steals": "STL",
    "Blocks": "BLK",
    "Turnovers": "TOV",
    "Minutes": "MIN",
}

STAT_COLUMNS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "MIN"]

BASE_COLS = [
    "IS_HOME",
    "REST_DAYS",
    "BACK_TO_BACK",
    "OPP_ALLOW_PTS",
    "OPP_ALLOW_REB",
    "OPP_ALLOW_AST",
]

IGNORE_COLS = {
    "GAME_DATE",
    "MATCHUP",
    "SEASON_ID",
    "TEAM_ABBREVIATION",
    "WL",
    "VIDEO_AVAILABLE",
    "OPP_TEAM",
}

N_TRAIN = 60  # why: training on last 60 games is enough and much faster
MAX_WORKERS = max(2, min(8, os.cpu_count() or 4))  # why: parallel speed-up


# =============================================================================
# FAST UTILS
# =============================================================================

def _rolling_slope(values: np.ndarray, window: int) -> np.ndarray:
    """
    O(n) rolling slope using a closed form of linear regression over a sliding window.
    Avoids pandas .rolling(...).apply(polyfit) which is very slow.

    y indices i = 0..w-1
    slope = (w*Σ(i*y_i) - Σi * Σy) / (w*Σ(i^2) - (Σi)^2)
    """
    x = np.asarray(values, dtype=float)
    n = window
    if x.size == 0 or n <= 1:
        return np.full_like(x, np.nan, dtype=float)

    idx = np.arange(n, dtype=float)
    sum_i = idx.sum()
    sum_i2 = (idx * idx).sum()
    denom = n * sum_i2 - (sum_i ** 2)
    if denom == 0:
        return np.full_like(x, np.nan, dtype=float)

    sum_y = np.convolve(x, np.ones(n, dtype=float), mode="valid")
    sum_iy = np.convolve(x, idx, mode="valid")
    slope_valid = (n * sum_iy - sum_i * sum_y) / denom

    out = np.full(x.shape, np.nan, dtype=float)
    out[n - 1:] = slope_valid
    return out


def _hash_frame_small(X: pd.DataFrame, y: np.ndarray, player_id: int, season: str, stat: str) -> str:
    """
    Tiny-key content hash. Avoid hashing entire DF bytes (slow).
    Uses shapes, column names, and rolling checksum of sampled values.
    """
    h = hashlib.sha1()
    h.update(str(player_id).encode())
    h.update(season.encode())
    h.update(stat.encode())
    h.update(str(X.shape).encode())
    h.update(",".join(map(str, X.columns)).encode())

    if len(X) > 0:
        # sample 128 rows evenly for checksum
        idx = np.linspace(0, len(X) - 1, num=min(128, len(X)), dtype=int)
        sample = X.iloc[idx].to_numpy()
        h.update(np.nan_to_num(sample, nan=0.0).tobytes())
        y_sample = y[idx if len(y) == len(X) else np.clip(idx, 0, len(y) - 1)]
        h.update(np.nan_to_num(y_sample, nan=0.0).tobytes())

    return h.hexdigest()


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def compute_opponent_strength(df: pd.DataFrame) -> pd.DataFrame:
    opp = (
        df.groupby("OPP_TEAM")[["PTS", "REB", "AST"]]
        .mean()
        .rename(
            columns={
                "PTS": "OPP_ALLOW_PTS",
                "REB": "OPP_ALLOW_REB",
                "AST": "OPP_ALLOW_AST",
            }
        )
    )
    return df.join(opp, on="OPP_TEAM")


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["IS_HOME"] = df["MATCHUP"].apply(lambda x: 1 if "vs" in x else 0)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE")
    df["REST_DAYS"] = df["GAME_DATE"].diff().dt.days.fillna(2)
    df["BACK_TO_BACK"] = (df["REST_DAYS"] == 1).astype(int)
    return df


@st.cache_data(show_spinner=False)
def load_logs(player_id: int, season: str) -> pd.DataFrame:
    # why: network IO cached; avoids repeated API hits
    logs = dfetch.get_player_game_logs_nba(player_id, season)
    return logs.copy()


def _ensure_training_base(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["OPP_TEAM"] = df["MATCHUP"].str.extract(r"(?:vs\.|@)\s(.+)$")
    df = compute_opponent_strength(df)
    df = add_context_features(df)
    df = df.dropna(subset=["PTS", "REB", "AST"])
    df = df.reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def build_all_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Single-pass feature builder for all STAT_COLUMNS.
    Produces L1/L3/L5, AVG5/AVG10, TREND5 per stat + BASE_COLS.
    """
    df = _ensure_training_base(df_in)

    out = df.copy()
    # numeric cast only once
    for c in STAT_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)

    # Lags and rolling means
    for c in STAT_COLUMNS:
        s = out[c].to_numpy(dtype=float, copy=False)
        # lags
        out[f"{c}_L1"] = np.roll(s, 1); out.loc[out.index[0], f"{c}_L1"] = np.nan
        out[f"{c}_L3"] = out[c].shift(3)
        out[f"{c}_L5"] = out[c].shift(5)

        # rolling means
        out[f"{c}_AVG5"] = pd.Series(s).rolling(5, min_periods=5).mean().to_numpy()
        out[f"{c}_AVG10"] = pd.Series(s).rolling(10, min_periods=10).mean().to_numpy()

        # vectorized trend (slope) over window=5
        out[f"{c}_TREND"] = _rolling_slope(s, 5)

    # Fill base cols (no leak)
    for bc in BASE_COLS:
        if bc in out.columns:
            med = float(out[bc].median()) if out[bc].notna().any() else 0.0
            out[bc] = out[bc].fillna(med)

    return out


def select_X_for_stat(features: pd.DataFrame, stat: str) -> pd.DataFrame:
    """
    Choose only relevant engineered columns for the given stat, + base context.
    """
    cols = PROP_MAP[stat] if isinstance(PROP_MAP[stat], list) else [PROP_MAP[stat]]
    pattern_bits = []
    for c in cols:
        pattern_bits.extend([f"{c}_L1", f"{c}_L3", f"{c}_L5", f"{c}_AVG5", f"{c}_AVG10", f"{c}_TREND"])

    keep = set(BASE_COLS) | set(pattern_bits)
    keep = [c for c in keep if c in features.columns]
    X = features[keep].copy()
    return X


def build_target(df: pd.DataFrame, stat: str) -> pd.Series:
    if isinstance(PROP_MAP[stat], list):
        tgt = df[PROP_MAP[stat]].sum(axis=1)
    else:
        tgt = df[PROP_MAP[stat]]
    tgt = pd.to_numeric(tgt, errors="coerce").astype(float)
    return tgt


# =============================================================================
# MODEL CACHING (SMALL KEY) + TRAIN/PREDICT
# =============================================================================

def get_or_train_model_cached(
    player_id: int,
    season: str,
    stat: str,
    X: pd.DataFrame,
    y: np.ndarray,
) -> ModelManager:
    """
    In-memory cache keyed by a tiny string; avoids Streamlit hashing huge DataFrames.
    """
    key = _hash_frame_small(X, y, player_id, season, stat)
    ss: Dict[str, ModelManager] = st.session_state.setdefault("model_cache", {})
    if key in ss:
        return ss[key]

    manager = ModelManager(random_state=42)  # ensure ModelManager uses n_jobs=-1 internally if possible
    manager.train(X, y)
    ss[key] = manager
    return manager


def train_predict_for_stat(
    player_id: int,
    season: str,
    stat: str,
    features: pd.DataFrame,
    fast_mode: bool,
) -> Dict[str, float]:
    """
    Single stat train/predict with fallback to fast baseline if data is small or fast_mode is enabled.
    """
    # Prepare TARGET
    y_all = build_target(features, stat).to_numpy()
    X_all = select_X_for_stat(features, stat)

    # Align and drop NaNs together
    df_final = pd.concat([pd.Series(y_all, name="TARGET"), X_all], axis=1).dropna()
    if df_final.empty or len(df_final) < 12:
        # fast baseline when too few samples
        pred = float(np.nanmean(y_all[-10:])) if np.isfinite(y_all[-10:]).any() else float("nan")
        return {
            "Stat": stat,
            "Prediction": pred,
            "Best Model": "Baseline(10G Mean)",
            "MAE": float("nan"),
            "MSE": float("nan"),
        }

    y_final = df_final["TARGET"].to_numpy(dtype=float)
    X_final = df_final.drop(columns=["TARGET"])

    # Limit training rows for speed
    if len(X_final) > N_TRAIN:
        X_final = X_final.iloc[-N_TRAIN:].copy()
        y_final = y_final[-N_TRAIN:].copy()

    # Next row to predict = last row's features
    X_next = X_final.tail(1)

    if fast_mode:
        # Fast baseline without modeling
        pred = float(np.nanmean(y_final[-10:])) if np.isfinite(y_final[-10:]).any() else float("nan")
        return {
            "Stat": stat,
            "Prediction": pred,
            "Best Model": "FastMode(10G Mean)",
            "MAE": float("nan"),
            "MSE": float("nan"),
        }

    # Train or fetch cached model
    manager = get_or_train_model_cached(player_id, season, stat, X_final, y_final)

    _ = manager.predict(X_next)  # ensure internal state computes best.prediction
    best = manager.best_model()

    return {
        "Stat": stat,
        "Prediction": float(best.prediction),
        "Best Model": best.name,
        "MAE": float(best.mae),
        "MSE": float(best.mse),
    }


# =============================================================================
# STREAMLIT APP
# =============================================================================

@st.cache_data(show_spinner=False)
def load_player_list():
    try:
        p = dfetch.get_active_players_balldontlie()
        p["full_name"] = p["first_name"] + " " + p["last_name"]
        return p.sort_values("full_name")[["id", "full_name", "team_id"]]
    except Exception:
        fallback = dfetch.get_player_list_nba()
        fallback["full_name"] = fallback["full_name"]
        fallback["team_id"] = None
        return fallback[["id", "full_name", "team_id"]]


def main():
    st.set_page_config(page_title="NBA Prop Predictor Pro", page_icon="🏀", layout="wide")
    st.title("NBA Prop Predictor — Full Auto Mode (Faster)")

    players = load_player_list()

    with st.sidebar:
        name = st.selectbox("Select Player", players["full_name"])
        row = players[players["full_name"] == name].iloc[0]
        player_id = int(row["id"])
        fast_mode = st.toggle("Fast mode (no training, 10-game mean)", value=False)
        run = st.button("Get Predictions Now")

    if not run:
        st.info("Choose a player and click 'Get Predictions Now'")
        return

    # Preferred season
    season = "2025-26"
    logs = load_logs(player_id, season)
    if logs.empty:
        year = datetime.date.today().year
        fallback = f"{year-1}-{str(year)[-2:]}"
        logs = load_logs(player_id, fallback)
        season = fallback

    if logs.empty:
        st.error("No game logs found.")
        return

    # One-time feature build
    with st.spinner("Building features…"):
        features = build_all_features(logs)

    # Parallel per-stat training/prediction
    st.subheader("Predicted Props (AI Model Ensemble)")
    results: List[Dict[str, float]] = []

    with st.spinner("Training models & predicting…"):
        futures = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for stat in PROP_MAP.keys():
                futures[ex.submit(train_predict_for_stat, player_id, season, stat, features, fast_mode)] = stat
            for fut in as_completed(futures):
                results.append(fut.result())

    # Sort by display order of PROP_MAP
    order_index = {k: i for i, k in enumerate(PROP_MAP.keys())}
    results.sort(key=lambda r: order_index[r["Stat"]])

    # Metric cards
    for r in results:
        with st.container():
            st.metric(label=f"{r['Stat']}", value=(round(r["Prediction"], 2) if np.isfinite(r["Prediction"]) else "—"), delta=None)
            st.caption(f"Model: {r['Best Model']} | MAE: {r['MAE']:.2f} | MSE: {r['MSE']:.2f}")
            st.markdown("---")

    # Recent games
    st.subheader("Recent Games")
    st.dataframe(
        logs[
            [
                "GAME_DATE",
                "MATCHUP",
                "PTS",
                "REB",
                "AST",
                "FG3M",
                "STL",
                "BLK",
                "TOV",
                "MIN",
            ]
        ],
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
