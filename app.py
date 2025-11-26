# app.py
"""
NBA Prop Predictor — Pro Tier (Fast + Training Fix + Active-only Dropdown)
Restores model training by imputing early-season NaNs instead of dropping rows.
"""

from __future__ import annotations

import os
import datetime
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

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

N_TRAIN = 60                      # keep training window short for speed
MIN_ROWS_FOR_MODEL = 6            # lower bar so training actually happens
MAX_WORKERS = max(2, min(8, os.cpu_count() or 4))


# =============================================================================
# FAST UTILS
# =============================================================================

def _rolling_slope(values: np.ndarray, window: int) -> np.ndarray:
    """Vectorized rolling slope (O(n))."""
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
    """Tiny-key content hash (avoids hashing entire frames)."""
    h = hashlib.sha1()
    h.update(str(player_id).encode()); h.update(season.encode()); h.update(stat.encode())
    h.update(str(X.shape).encode()); h.update(",".join(map(str, X.columns)).encode())
    if len(X) > 0:
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
        .rename(columns={"PTS": "OPP_ALLOW_PTS", "REB": "OPP_ALLOW_REB", "AST": "OPP_ALLOW_AST"})
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
    logs = dfetch.get_player_game_logs_nba(player_id, season)
    return logs.copy()


def _ensure_training_base(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["OPP_TEAM"] = df["MATCHUP"].str.extract(r"(?:vs\.|@)\s(.+)$")
    df = compute_opponent_strength(df)
    df = add_context_features(df)
    df = df.dropna(subset=["PTS", "REB", "AST"])
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def build_all_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Single-pass feature builder for all STAT_COLUMNS.
    Uses min_periods=1 so we don't lose early-season rows.
    """
    df = _ensure_training_base(df_in)
    out = df.copy()

    for c in STAT_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)

    for c in STAT_COLUMNS:
        s = out[c].to_numpy(dtype=float, copy=False)

        # lags
        out[f"{c}_L1"] = np.roll(s, 1); out.loc[out.index[0], f"{c}_L1"] = np.nan
        out[f"{c}_L3"] = out[c].shift(3)
        out[f"{c}_L5"] = out[c].shift(5)

        # rolling means (min_periods=1 keeps rows)
        out[f"{c}_AVG5"] = pd.Series(s).rolling(5, min_periods=1).mean().to_numpy()
        out[f"{c}_AVG10"] = pd.Series(s).rolling(10, min_periods=1).mean().to_numpy()

        # vectorized trend (first window-1 are NaN; we'll impute later)
        out[f"{c}_TREND"] = _rolling_slope(s, 5)

    # fill base cols
    for bc in BASE_COLS:
        if bc in out.columns:
            med = float(out[bc].median()) if out[bc].notna().any() else 0.0
            out[bc] = out[bc].fillna(med)

    return out


def select_X_for_stat(features: pd.DataFrame, stat: str) -> pd.DataFrame:
    cols = PROP_MAP[stat] if isinstance(PROP_MAP[stat], list) else [PROP_MAP[stat]]
    bits = []
    for c in cols:
        bits.extend([f"{c}_L1", f"{c}_L3", f"{c}_L5", f"{c}_AVG5", f"{c}_AVG10", f"{c}_TREND"])
    keep = [c for c in set(BASE_COLS) | set(bits) if c in features.columns]
    return features[keep].copy()


def build_target(df: pd.DataFrame, stat: str) -> pd.Series:
    if isinstance(PROP_MAP[stat], list):
        tgt = df[PROP_MAP[stat]].sum(axis=1)
    else:
        tgt = df[PROP_MAP[stat]]
    return pd.to_numeric(tgt, errors="coerce").astype(float)


# =============================================================================
# MODEL CACHING + TRAIN/PREDICT
# =============================================================================

def get_or_train_model_cached(player_id: int, season: str, stat: str, X: pd.DataFrame, y: np.ndarray) -> ModelManager:
    key = _hash_frame_small(X, y, player_id, season, stat)
    ss: Dict[str, ModelManager] = st.session_state.setdefault("model_cache", {})
    if key in ss:
        return ss[key]
    manager = ModelManager(random_state=42)  # ensure internal estimator uses n_jobs=-1 if possible
    manager.train(X, y)
    ss[key] = manager
    return manager


def _impute_features(X: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill then median-impute (why: avoid dropping early rows)."""
    X = X.copy()
    X = X.fillna(method="ffill")
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)
    X = X.fillna(0.0)  # last resort
    return X


def train_predict_for_stat(player_id: int, season: str, stat: str, features: pd.DataFrame, fast_mode: bool) -> Dict[str, float]:
    # target + features
    y_all = build_target(features, stat).to_numpy()
    X_all = select_X_for_stat(features, stat)

    # align and impute (do not drop)
    df_join = pd.concat([pd.Series(y_all, name="TARGET", index=X_all.index), X_all], axis=1)
    df_join = df_join.loc[~df_join["TARGET"].isna()].copy()

    if df_join.empty:
        pred = float(np.nan)  # nothing we can do
        return {"Stat": stat, "Prediction": pred, "Best Model": "NoData", "MAE": float("nan"), "MSE": float("nan")}

    y_final = df_join["TARGET"].to_numpy(dtype=float)
    X_final = _impute_features(df_join.drop(columns=["TARGET"]))

    # limit training window
    if len(X_final) > N_TRAIN:
        X_final = X_final.iloc[-N_TRAIN:].copy()
        y_final = y_final[-N_TRAIN:].copy()

    X_next = X_final.tail(1)

    # fast mode bypass
    if fast_mode:
        pred = float(np.nanmean(y_final[-10:])) if np.isfinite(y_final[-10:]).any() else float("nan")
        return {"Stat": stat, "Prediction": pred, "Best Model": "FastMode(10G Mean)", "MAE": float("nan"), "MSE": float("nan")}

    # require minimal rows to avoid CV/fit errors; otherwise attempt and fallback
    if len(X_final) < MIN_ROWS_FOR_MODEL:
        pred = float(np.nanmean(y_final[-10:])) if np.isfinite(y_final[-10:]).any() else float("nan")
        return {"Stat": stat, "Prediction": pred, "Best Model": "Baseline(10G Mean)", "MAE": float("nan"), "MSE": float("nan")}

    try:
        manager = get_or_train_model_cached(player_id, season, stat, X_final, y_final)
        _ = manager.predict(X_next)  # sets internal best.prediction
        best = manager.best_model()
        return {"Stat": stat, "Prediction": float(best.prediction), "Best Model": best.name, "MAE": float(best.mae), "MSE": float(best.mse)}
    except Exception:
        # why: tiny/degenerate datasets or estimator errors
        pred = float(np.nanmean(y_final[-10:])) if np.isfinite(y_final[-10:]).any() else float("nan")
        return {"Stat": stat, "Prediction": pred, "Best Model": "Baseline(Fallback)", "MAE": float("nan"), "MSE": float("nan")}


# =============================================================================
# ACTIVE-ONLY PLAYER LIST
# =============================================================================

def _filter_active_players(df: pd.DataFrame, season_str: str) -> pd.DataFrame:
    if df.empty:
        return df
    try:
        parts = season_str.split("-")
        end_year = int("20" + parts[1]) if len(parts) == 2 and len(parts[1]) == 2 else int(parts[1])
    except Exception:
        end_year = datetime.date.today().year

    df = df.copy()
    if "full_name" not in df.columns:
        if {"first_name", "last_name"}.issubset(df.columns):
            df["full_name"] = (df["first_name"].astype(str).str.strip() + " " + df["last_name"].astype(str).str.strip()).str.strip()
        elif "PLAYER" in df.columns:
            df["full_name"] = df["PLAYER"].astype(str)
        else:
            df["full_name"] = df.get("full_name", "Unknown")

    active = pd.Series(False, index=df.index)
    for flag in ["is_active", "IS_ACTIVE", "active"]:
        if flag in df.columns: active = active | df[flag].astype(bool)
    for flag in ["STATUS", "status", "ROSTERSTATUS", "rosterstatus"]:
        if flag in df.columns: active = active | df[flag].astype(str).str.contains("active", case=False, na=False)
    for ty in ["TO_YEAR", "to_year", "to", "TO"]:
        if ty in df.columns:
            to_vals = pd.to_numeric(df[ty], errors="coerce")
            active = active | (to_vals >= end_year - 1)

    team_cols = [c for c in ["team_id", "TEAM_ID", "teamId"] if c in df.columns]
    if team_cols:
        active = active | (df[team_cols[0]].notna() & (df[team_cols[0]] != 0))
    if not active.any() and team_cols:
        active = df[team_cols[0]].notna() & (df[team_cols[0]] != 0)

    return df[active].copy()


@st.cache_data(show_spinner=False)
def load_player_list(season: str = "2025-26"):
    try:
        p = dfetch.get_active_players_balldontlie()
        if "id" not in p.columns and "player_id" in p.columns:
            p = p.rename(columns={"player_id": "id"})
        if "full_name" not in p.columns and {"first_name", "last_name"}.issubset(p.columns):
            p["full_name"] = p["first_name"] + " " + p["last_name"]
        if "team_id" not in p.columns:
            if "team" in p.columns and isinstance(p["team"].iloc[0], dict):
                p["team_id"] = p["team"].apply(lambda t: t.get("id") if isinstance(t, dict) else None)
            else:
                p["team_id"] = p.get("TEAM_ID", None)
        p = p[["id", "full_name", "team_id"]].dropna(subset=["id"]).drop_duplicates(subset=["id"])
        return p.sort_values("full_name")
    except Exception:
        fb = dfetch.get_player_list_nba()
        fb = _filter_active_players(fb, season)
        if "id" not in fb.columns:
            for c in ["PLAYER_ID", "player_id", "PersonId", "PERSON_ID"]:
                if c in fb.columns:
                    fb = fb.rename(columns={c: "id"}); break
        if "team_id" not in fb.columns:
            for c in ["TEAM_ID", "teamId"]:
                if c in fb.columns:
                    fb = fb.rename(columns={c: "team_id"}); break
        fb["team_id"] = fb.get("team_id", None)
        out = fb[["id", "full_name", "team_id"]].dropna(subset=["id"]).drop_duplicates(subset=["id"])
        return out.sort_values("full_name")


# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    st.set_page_config(page_title="NBA Prop Predictor Pro", page_icon="🏀", layout="wide")
    st.title("NBA Prop Predictor — Full Auto Mode (Faster + Training Fix)")

    season = "2025-26"
    players = load_player_list(season)

    with st.sidebar:
        name = st.selectbox("Select Player", players["full_name"])
        row = players[players["full_name"] == name].iloc[0]
        player_id = int(row["id"])
        fast_mode = st.toggle("Fast mode (no training, 10-game mean)", value=False)
        run = st.button("Get Predictions Now")

    if not run:
        st.info("Choose a player and click 'Get Predictions Now'")
        return

    logs = load_logs(player_id, season)
    if logs.empty:
        year = datetime.date.today().year
        fallback = f"{year-1}-{str(year)[-2:]}"
        logs = load_logs(player_id, fallback)
        season = fallback

    if logs.empty:
        st.error("No game logs found.")
        return

    with st.spinner("Building features…"):
        features = build_all_features(logs)

    st.subheader("Predicted Props (AI Model Ensemble)")
    results: List[Dict[str, float]] = []
    with st.spinner("Training models & predicting…"):
        futures = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for stat in PROP_MAP.keys():
                futures[ex.submit(train_predict_for_stat, player_id, season, stat, features, fast_mode)] = stat
            for fut in as_completed(futures):
                results.append(fut.result())

    order_index = {k: i for i, k in enumerate(PROP_MAP.keys())}
    results.sort(key=lambda r: order_index[r["Stat"]])

    for r in results:
        with st.container():
            val = r["Prediction"]
            st.metric(label=f"{r['Stat']}", value=(round(val, 2) if np.isfinite(val) else "—"), delta=None)
            st.caption(f"Model: {r['Best Model']} | MAE: {r['MAE']:.2f} | MSE: {r['MSE']:.2f}")
            st.markdown("---")

    st.subheader("Recent Games")
    st.dataframe(
        logs[["GAME_DATE", "MATCHUP", "PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV", "MIN"]],
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
