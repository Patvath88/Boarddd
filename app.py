# app.py
"""
NBA Prop Predictor — Elite Edition
- Faster training (single-pass features + imputation + caching)
- Active-only player dropdown
- Walk-forward backtesting (this season + last season)
- Save/Delete prediction runs
- Favorites page: save players and bulk-train/predict
- Elite UI styling
"""

from __future__ import annotations

import os
import uuid
import json
import math
import hashlib
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st

import data_fetching as dfetch
from models import ModelManager


# =============================================================================
# CONFIG
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

BASE_COLS = ["IS_HOME", "REST_DAYS", "BACK_TO_BACK", "OPP_ALLOW_PTS", "OPP_ALLOW_REB", "OPP_ALLOW_AST"]

N_TRAIN = 60                # why: keeps training light without losing recency
MIN_ROWS_FOR_MODEL = 6      # why: tiny bar so models still train early season
MAX_WORKERS = max(2, min(8, os.cpu_count() or 4))

DATA_DIR = Path("./data")   # local persistence
DATA_DIR.mkdir(parents=True, exist_ok=True)
PRED_FILE = DATA_DIR / "predictions.jsonl"
FAV_FILE = DATA_DIR / "favorites.json"


# =============================================================================
# STYLING
# =============================================================================

def inject_css() -> None:
    # why: consistent elite styling without changing Streamlit theme
    st.markdown(
        """
<style>
/* Global */
html, body { font-family: Inter, ui-sans-serif, system-ui; }
.block-container { padding-top: 2rem; max-width: 1200px; }
h1 { background: linear-gradient(90deg,#e2e8f0 0%, #60a5fa 40%, #34d399 100%);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.st-emotion-cache-16txtl3 { padding-top: 1rem; }

/* Cards */
.card {
  background: rgba(17, 24, 39, 0.6);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 18px 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
  backdrop-filter: blur(6px);
}

/* Metric */
div[data-testid="stMetric"] { padding: 8px 12px; }
div[data-testid="stMetric"] > label { font-weight: 600; opacity: .9; }
div[data-testid="stMetricValue"] { font-size: 2rem; }

/* Buttons */
.stButton>button {
  border-radius: 12px; padding: 10px 14px; font-weight: 600;
  background: linear-gradient(90deg,#0ea5e9,#22c55e);
  border: none;
}
.stButton>button:hover { filter: brightness(1.05); }
.tag {
  display:inline-block; padding:.2rem .55rem; border-radius:999px; font-size:.75rem;
  background:rgba(99,102,241,.15); border:1px solid rgba(99,102,241,.35);
}
</style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# UTILS: hashing, seasons, persistence
# =============================================================================

def _rolling_slope(values: np.ndarray, window: int) -> np.ndarray:
    # why: rolling polyfit is slow; use closed-form slope
    x = np.asarray(values, dtype=float); n = window
    if x.size == 0 or n <= 1: return np.full_like(x, np.nan, dtype=float)
    idx = np.arange(n, dtype=float)
    sum_i = idx.sum(); sum_i2 = (idx * idx).sum()
    denom = n * sum_i2 - (sum_i ** 2)
    if denom == 0: return np.full_like(x, np.nan, dtype=float)
    sum_y = np.convolve(x, np.ones(n), mode="valid")
    sum_iy = np.convolve(x, idx, mode="valid")
    slope_valid = (n * sum_iy - sum_i * sum_y) / denom
    out = np.full(x.shape, np.nan, dtype=float); out[n - 1:] = slope_valid
    return out


def _hash_frame_small(X: pd.DataFrame, y: np.ndarray, player_id: int, season: str, stat: str) -> str:
    h = hashlib.sha1()
    h.update(f"{player_id}|{season}|{stat}|{X.shape}|{','.join(map(str,X.columns))}".encode())
    if len(X) > 0:
        idx = np.linspace(0, len(X) - 1, num=min(128, len(X)), dtype=int)
        h.update(np.nan_to_num(X.iloc[idx].to_numpy(), nan=0.0).tobytes())
        ys = y[idx if len(y)==len(X) else np.clip(idx,0,len(y)-1)]
        h.update(np.nan_to_num(ys, nan=0.0).tobytes())
    return h.hexdigest()


def prev_season(season: str) -> str:
    try:
        s, e = season.split("-")
        s, e = int(s), int(("20"+e) if len(e)==2 else e)
        ps, pe = s-1, e-1
        return f"{ps}-{str(pe)[-2:]}"
    except Exception:
        year = dt.date.today().year
        return f"{year-2}-{str(year-1)[-2:]}"


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists(): return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _load_favorites() -> List[dict]:
    if not FAV_FILE.exists(): return []
    return json.loads(FAV_FILE.read_text("utf-8"))


def _save_favorites(rows: List[dict]) -> None:
    FAV_FILE.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


# =============================================================================
# DATA + FEATURES
# =============================================================================

@st.cache_data(show_spinner=False)
def load_player_list(season: str = "2025-26"):
    # why: active-only dropdown
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
        return p[["id", "full_name", "team_id"]].dropna(subset=["id"]).drop_duplicates(subset=["id"]).sort_values("full_name")
    except Exception:
        fb = dfetch.get_player_list_nba()
        fb = _filter_active_players(fb, season)
        if "id" not in fb.columns:
            for c in ["PLAYER_ID", "player_id", "PersonId", "PERSON_ID"]:
                if c in fb.columns: fb = fb.rename(columns={c: "id"}); break
        if "team_id" not in fb.columns:
            for c in ["TEAM_ID", "teamId"]:
                if c in fb.columns: fb = fb.rename(columns={c: "team_id"}); break
        fb["team_id"] = fb.get("team_id", None)
        return fb[["id","full_name","team_id"]].dropna(subset=["id"]).drop_duplicates(subset=["id"]).sort_values("full_name")


def _filter_active_players(df: pd.DataFrame, season_str: str) -> pd.DataFrame:
    if df.empty: return df
    try:
        parts = season_str.split("-")
        end_year = int("20"+parts[1]) if len(parts)==2 and len(parts[1])==2 else int(parts[1])
    except Exception:
        end_year = dt.date.today().year
    df = df.copy()
    if "full_name" not in df.columns:
        if {"first_name","last_name"}.issubset(df.columns):
            df["full_name"] = (df["first_name"].astype(str).str.strip() + " " + df["last_name"].astype(str).str.strip()).str.strip()
        elif "PLAYER" in df.columns:
            df["full_name"] = df["PLAYER"].astype(str)
        else:
            df["full_name"] = df.get("full_name","Unknown")
    active = pd.Series(False, index=df.index)
    for f in ["is_active","IS_ACTIVE","active"]:
        if f in df.columns: active = active | df[f].astype(bool)
    for f in ["STATUS","status","ROSTERSTATUS","rosterstatus"]:
        if f in df.columns: active = active | df[f].astype(str).str.contains("active", case=False, na=False)
    for ty in ["TO_YEAR","to_year","to","TO"]:
        if ty in df.columns:
            to_vals = pd.to_numeric(df[ty], errors="coerce")
            active = active | (to_vals >= end_year-1)
    team_cols = [c for c in ["team_id","TEAM_ID","teamId"] if c in df.columns]
    if team_cols:
        active = active | (df[team_cols[0]].notna() & (df[team_cols[0]] != 0))
    if not active.any() and team_cols:
        active = df[team_cols[0]].notna() & (df[team_cols[0]] != 0)
    return df[active].copy()


@st.cache_data(show_spinner=False)
def load_logs(player_id: int, season: str) -> pd.DataFrame:
    return dfetch.get_player_game_logs_nba(player_id, season).copy()


def compute_opponent_strength(df: pd.DataFrame) -> pd.DataFrame:
    opp = df.groupby("OPP_TEAM")[["PTS", "REB", "AST"]].mean().rename(columns={
        "PTS": "OPP_ALLOW_PTS", "REB": "OPP_ALLOW_REB", "AST": "OPP_ALLOW_AST"
    })
    return df.join(opp, on="OPP_TEAM")


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["IS_HOME"] = df["MATCHUP"].apply(lambda x: 1 if "vs" in x else 0)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE")
    df["REST_DAYS"] = df["GAME_DATE"].diff().dt.days.fillna(2)
    df["BACK_TO_BACK"] = (df["REST_DAYS"] == 1).astype(int)
    return df


def _ensure_training_base(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["OPP_TEAM"] = df["MATCHUP"].str.extract(r"(?:vs\.|@)\s(.+)$")
    df = compute_opponent_strength(df)
    df = add_context_features(df)
    df = df.dropna(subset=["PTS","REB","AST"])
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def build_all_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_training_base(df_in)
    out = df.copy()
    for c in STAT_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    for c in STAT_COLUMNS:
        s = out[c].to_numpy(dtype=float, copy=False)
        out[f"{c}_L1"] = np.roll(s, 1); out.loc[out.index[0], f"{c}_L1"] = np.nan
        out[f"{c}_L3"] = out[c].shift(3); out[f"{c}_L5"] = out[c].shift(5)
        out[f"{c}_AVG5"] = pd.Series(s).rolling(5, min_periods=1).mean().to_numpy()
        out[f"{c}_AVG10"] = pd.Series(s).rolling(10, min_periods=1).mean().to_numpy()
        out[f"{c}_TREND"] = _rolling_slope(s, 5)
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


def _impute_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X = X.fillna(method="ffill")
    med = X.median(numeric_only=True)
    X = X.fillna(med)
    X = X.fillna(0.0)
    return X


# =============================================================================
# MODEL CACHE + TRAIN/PREDICT
# =============================================================================

def get_or_train_model_cached(player_id: int, season: str, stat: str, X: pd.DataFrame, y: np.ndarray) -> ModelManager:
    key = _hash_frame_small(X, y, player_id, season, stat)
    ss: Dict[str, ModelManager] = st.session_state.setdefault("model_cache", {})
    if key in ss: return ss[key]
    manager = ModelManager(random_state=42)   # ensure estimators inside use scaling + TS CV if possible
    manager.train(X, y)
    ss[key] = manager
    return manager


def train_predict_for_stat(player_id: int, season: str, stat: str, features: pd.DataFrame, fast_mode: bool) -> Dict[str, float]:
    y_all = build_target(features, stat).to_numpy()
    X_all = select_X_for_stat(features, stat)
    df_join = pd.concat([pd.Series(y_all, name="TARGET", index=X_all.index), X_all], axis=1)
    df_join = df_join.loc[~df_join["TARGET"].isna()].copy()
    if df_join.empty:
        return {"Stat": stat, "Prediction": float("nan"), "Best Model": "NoData", "MAE": float("nan"), "MSE": float("nan")}
    y_final = df_join["TARGET"].to_numpy(dtype=float)
    X_final = _impute_features(df_join.drop(columns=["TARGET"]))
    if len(X_final) > N_TRAIN:
        X_final = X_final.iloc[-N_TRAIN:].copy()
        y_final = y_final[-N_TRAIN:].copy()
    X_next = X_final.tail(1)
    if fast_mode or len(X_final) < MIN_ROWS_FOR_MODEL:
        pred = float(np.nanmean(y_final[-10:])) if np.isfinite(y_final[-10:]).any() else float("nan")
        return {"Stat": stat, "Prediction": pred, "Best Model": "Baseline(10G Mean)" if fast_mode else "Baseline(SmallSample)", "MAE": float("nan"), "MSE": float("nan")}
    try:
        manager = get_or_train_model_cached(player_id, season, stat, X_final, y_final)
        _ = manager.predict(X_next)
        best = manager.best_model()
        return {"Stat": stat, "Prediction": float(best.prediction), "Best Model": best.name, "MAE": float(best.mae), "MSE": float(best.mse)}
    except Exception:
        pred = float(np.nanmean(y_final[-10:])) if np.isfinite(y_final[-10:]).any() else float("nan")
        return {"Stat": stat, "Prediction": pred, "Best Model": "Baseline(Fallback)", "MAE": float("nan"), "MSE": float("nan")}


# =============================================================================
# BACKTESTING
# =============================================================================

def walk_forward_backtest(player_id: int, season: str, stat: str, features: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    # why: proper expanding-window evaluation
    y_all = build_target(features, stat)
    X_all = select_X_for_stat(features, stat)
    df = pd.concat([features[["GAME_DATE"]], y_all.rename("TARGET"), X_all], axis=1)
    df = df.dropna(subset=["TARGET"]).reset_index(drop=True)

    preds, truth, dates = [], [], []
    start_idx = max(MIN_ROWS_FOR_MODEL + 5, 8)  # leave room for early NaNs
    start_idx = min(start_idx, max(len(df) - 1, 1))

    progress = st.progress(0, text=f"Backtesting {stat} — {season}")
    steps = max(1, len(df) - start_idx)

    for i, t in enumerate(range(start_idx, len(df))):
        train = df.iloc[max(0, t - N_TRAIN): t]   # cap by N_TRAIN
        test = df.iloc[[t]]

        y_tr = train["TARGET"].to_numpy()
        X_tr = _impute_features(train.drop(columns=["TARGET", "GAME_DATE"]))
        X_te = _impute_features(test.drop(columns=["TARGET", "GAME_DATE"]))

        if len(X_tr) < MIN_ROWS_FOR_MODEL:
            yhat = float(np.nanmean(y_tr[-10:])) if np.isfinite(y_tr[-10:]).any() else float("nan")
        else:
            try:
                manager = ModelManager(random_state=42)
                manager.train(X_tr, y_tr)
                yhat = float(manager.predict(X_te))
            except Exception:
                yhat = float(np.nanmean(y_tr[-10:])) if np.isfinite(y_tr[-10:]).any() else float("nan")

        preds.append(yhat)
        truth.append(float(test["TARGET"].iloc[0]))
        dates.append(pd.to_datetime(test["GAME_DATE"].iloc[0]))
        if steps: progress.progress((i + 1) / steps, text=f"Backtesting {stat} — {season}")

    progress.empty()

    out = pd.DataFrame({"GAME_DATE": dates, "y_true": truth, "y_pred": preds})
    out["abs_err"] = (out["y_true"] - out["y_pred"]).abs()
    out["sq_err"] = (out["y_true"] - out["y_pred"]) ** 2
    mae = float(out["abs_err"].mean()) if len(out) else float("nan")
    rmse = float(np.sqrt(out["sq_err"].mean())) if len(out) else float("nan")
    return out, {"MAE": mae, "RMSE": rmse, "N": int(len(out))}


# =============================================================================
# PAGES
# =============================================================================

def page_predict(players: pd.DataFrame):
    st.header("NBA Prop Predictor — Elite")
    st.caption("Auto-train per stat, ensemble model selection, mobile-friendly metrics.")

    name = st.selectbox("Select Player", players["full_name"], key="predict_player")
    player_row = players[players["full_name"] == name].iloc[0]
    player_id = int(player_row["id"])

    fast_mode = st.toggle("Fast mode (no training, 10-game mean)", value=False, key="fast_toggle")
    season = "2025-26"
    run = st.button("Get Predictions Now")

    if not run:
        st.info("Choose a player and click **Get Predictions Now**")
        return

    logs = load_logs(player_id, season)
    if logs.empty:
        year = dt.date.today().year
        fallback = f"{year-1}-{str(year)[-2:]}"
        logs = load_logs(player_id, fallback)
        season = fallback

    if logs.empty:
        st.error("No game logs found."); return

    with st.spinner("Building features…"):
        features = build_all_features(logs)

    st.subheader("Predicted Props")
    results: List[Dict[str, float]] = []
    with st.spinner("Training models & predicting…"):
        futures = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for stat in PROP_MAP.keys():
                futures[ex.submit(train_predict_for_stat, player_id, season, stat, features, fast_mode)] = stat
            for fut in as_completed(futures): results.append(fut.result())

    order = {k: i for i, k in enumerate(PROP_MAP.keys())}
    results.sort(key=lambda r: order[r["Stat"]])

    cols = st.columns(3)
    for idx, r in enumerate(results):
        with cols[idx % 3].container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.metric(r["Stat"], value=(round(r["Prediction"],2) if np.isfinite(r["Prediction"]) else "—"))
            st.caption(f"Model: {r['Best Model']} · MAE: {r['MAE']:.2f} · MSE: {r['MSE']:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("Recent Games")
    st.dataframe(
        logs[["GAME_DATE","MATCHUP","PTS","REB","AST","FG3M","STL","BLK","TOV","MIN"]],
        use_container_width=True,
    )

    if st.button("💾 Save this run"):
        rows = _read_jsonl(PRED_FILE)
        rec = {
            "id": str(uuid.uuid4()),
            "ts": dt.datetime.utcnow().isoformat(),
            "player_id": player_id,
            "player_name": name,
            "season": season,
            "fast_mode": fast_mode,
            "results": results,
        }
        rows.append(rec); _write_jsonl(PRED_FILE, rows)
        st.success("Saved.")


def page_backtest(players: pd.DataFrame):
    st.header("Backtesting")
    name = st.selectbox("Player", players["full_name"], key="bt_player")
    player_id = int(players[players["full_name"] == name].iloc[0]["id"])
    season = "2025-26"
    do_this = st.checkbox("This season", value=True)
    do_last = st.checkbox("Last season", value=True)
    run = st.button("Run Backtests")

    if not run: return

    seasons = []
    if do_this: seasons.append(season)
    if do_last: seasons.append(prev_season(season))
    if not seasons:
        st.info("Pick at least one season.")
        return

    for s in seasons:
        logs = load_logs(player_id, s)
        if logs.empty:
            st.warning(f"No logs for {s}."); continue
        features = build_all_features(logs)

        st.subheader(f"{name} — {s}")
        s_rows = []
        for stat in PROP_MAP.keys():
            df_bt, summary = walk_forward_backtest(player_id, s, stat, features)
            st.markdown(f"**{stat}** — MAE: `{summary['MAE']:.2f}` · RMSE: `{summary['RMSE']:.2f}` · N: `{summary['N']}`")
            with st.expander(f"Details — {stat}"):
                st.dataframe(df_bt, use_container_width=True)
            s_rows.append({"Stat": stat, **summary})
        st.dataframe(pd.DataFrame(s_rows).set_index("Stat"), use_container_width=True)


def page_favorites(players: pd.DataFrame):
    st.header("Favorites")
    favs = _load_favorites()

    # Add favorite
    col1, col2 = st.columns([3,1])
    with col1:
        name = st.selectbox("Add a player", players["full_name"], key="fav_add_sel")
    with col2:
        if st.button("➕ Add"):
            row = players[players["full_name"] == name].iloc[0]
            pid = int(row["id"])
            if not any(f["id"] == pid for f in favs):
                favs.append({"id": pid, "full_name": name, "team_id": int(row.get("team_id", 0) or 0)})
                _save_favorites(favs)
                st.success("Added to favorites.")
            else:
                st.info("Already in favorites.")

    # List favorites
    if favs:
        st.subheader("Saved favorites")
        df_f = pd.DataFrame(favs)
        st.dataframe(df_f[["id","full_name","team_id"]], use_container_width=True, hide_index=True)
        remove_id = st.text_input("Remove by player id", value="")
        if st.button("🗑️ Remove"):
            try:
                rid = int(remove_id)
                favs = [f for f in favs if f["id"] != rid]
                _save_favorites(favs); st.success("Removed.")
            except Exception:
                st.error("Enter a valid numeric player id.")
    else:
        st.info("No favorites yet.")

    st.divider()
    st.subheader("Bulk Predict")
    fast_mode = st.toggle("Fast mode (no training, 10-game mean)", value=False, key="fav_fast")
    if st.button("Run predictions for all favorites"):
        if not favs:
            st.info("Add favorites first."); return

        season = "2025-26"
        all_results = []
        progress = st.progress(0, text="Predicting for favorites…")

        for i, fav in enumerate(favs):
            pid, pname = fav["id"], fav["full_name"]
            logs = load_logs(pid, season)
            if logs.empty:
                year = dt.date.today().year
                fallback = f"{year-1}-{str(year)[-2:]}"
                logs = load_logs(pid, fallback)
                season = fallback
            if logs.empty:
                all_results.append({"player_id": pid, "player_name": pname, "season": season, "error": "no logs"})
                progress.progress((i+1)/len(favs)); continue

            feats = build_all_features(logs)
            # per-stat in parallel
            futures = {}
            res = []
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                for stat in PROP_MAP.keys():
                    futures[ex.submit(train_predict_for_stat, pid, season, stat, feats, fast_mode)] = stat
                for fut in as_completed(futures): res.append(fut.result())
            res.sort(key=lambda r: list(PROP_MAP.keys()).index(r["Stat"]))
            all_results.append({"player_id": pid, "player_name": pname, "season": season, "results": res})
            progress.progress((i+1)/len(favs))

        progress.empty()
        st.success("Done.")
        for rec in all_results:
            st.markdown(f"### {rec['player_name']} — {rec['season']}")
            if "error" in rec:
                st.warning(rec["error"]); continue
            cols = st.columns(3)
            for idx, r in enumerate(rec["results"]):
                with cols[idx % 3]:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.metric(r["Stat"], value=(round(r["Prediction"],2) if np.isfinite(r["Prediction"]) else "—"))
                    st.caption(f"Model: {r['Best Model']} · MAE: {r['MAE']:.2f} · MSE: {r['MSE']:.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)

        if st.button("💾 Save batch"):
            rows = _read_jsonl(PRED_FILE)
            for rec in all_results:
                if "error" in rec: continue
                rows.append({
                    "id": str(uuid.uuid4()),
                    "ts": dt.datetime.utcnow().isoformat(),
                    "player_id": rec["player_id"],
                    "player_name": rec["player_name"],
                    "season": rec["season"],
                    "fast_mode": fast_mode,
                    "results": rec["results"],
                })
            _write_jsonl(PRED_FILE, rows); st.success("Saved all.")


def page_saved():
    st.header("Saved Predictions")
    rows = _read_jsonl(PRED_FILE)
    if not rows:
        st.info("No saved runs yet."); return
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    st.dataframe(df[["id","ts","player_id","player_name","season","fast_mode"]].sort_values("ts", ascending=False), use_container_width=True)
    del_id = st.text_input("Delete by run id", value="")
    if st.button("🗑️ Delete run"):
        rid = del_id.strip()
        new_rows = [r for r in rows if r["id"] != rid]
        if len(new_rows) == len(rows):
            st.warning("Run id not found.")
        else:
            _write_jsonl(PRED_FILE, new_rows); st.success("Deleted.")


# =============================================================================
# APP
# =============================================================================

def main():
    st.set_page_config(page_title="NBA Prop Predictor — Elite", page_icon="🏀", layout="wide")
    inject_css()

    players = load_player_list("2025-26")

    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio("Go to", ["Predict", "Backtest", "Favorites", "Saved"], label_visibility="collapsed")

        st.markdown("---")
        st.markdown('<span class="tag">Pro Tier</span> <span class="tag">Walk-Forward CV</span> <span class="tag">Cache</span>', unsafe_allow_html=True)

    if page == "Predict":
        page_predict(players)
    elif page == "Backtest":
        page_backtest(players)
    elif page == "Favorites":
        page_favorites(players)
    else:
        page_saved()


if __name__ == "__main__":
    main()
