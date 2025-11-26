# app.py
"""
NBA Prop Predictor — Pro Tier (Stability Patch + TARGET Fix + Metric Card UI)
Active-only players in dropdown.
"""

from __future__ import annotations
import datetime
import numpy as np
import pandas as pd
import streamlit as st

import data_fetching as dfetch
from models import ModelManager


# ======================================================================
# PROP DEFINITIONS
# ======================================================================

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

STAT_COLUMNS = ["PTS","REB","AST","STL","BLK","TOV","FG3M","MIN"]


# ======================================================================
# FEATURE ENGINEERING HELPERS
# ======================================================================

def compute_opponent_strength(df):
    opp = df.groupby("OPP_TEAM")[["PTS","REB","AST"]].mean().rename(columns={
        "PTS": "OPP_ALLOW_PTS",
        "REB": "OPP_ALLOW_REB",
        "AST": "OPP_ALLOW_AST",
    })
    return df.join(opp, on="OPP_TEAM")


def lag_features(df, col):
    df[f"{col}_L1"] = df[col].shift(1)
    df[f"{col}_L3"] = df[col].shift(3)
    df[f"{col}_L5"] = df[col].shift(5)
    return df


def rolling_features(df, col):
    df[f"{col}_AVG5"] = df[col].rolling(5).mean()
    df[f"{col}_AVG10"] = df[col].rolling(10).mean()
    return df


def trend_feature(df, col):
    # why: simple to keep, acceptable cost for small inputs
    df[f"{col}_TREND"] = df[col].rolling(5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan,
        raw=True,
    )
    return df


def context_features(df):
    df["IS_HOME"] = df["MATCHUP"].apply(lambda x: 1 if "vs" in x else 0)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["REST_DAYS"] = df["GAME_DATE"].diff().dt.days.fillna(2)
    df["BACK_TO_BACK"] = (df["REST_DAYS"] == 1).astype(int)
    return df


# ======================================================================
# PER-PROP FEATURE BUILDER
# ======================================================================

def build_features_for_stat(df, stat):
    df = df.copy()

    cols = PROP_MAP[stat] if isinstance(PROP_MAP[stat], list) else [PROP_MAP[stat]]

    for col in cols:
        df = lag_features(df, col)
        df = rolling_features(df, col)
        df = trend_feature(df, col)

    base_cols = [
        "IS_HOME",
        "REST_DAYS",
        "BACK_TO_BACK",
        "OPP_ALLOW_PTS",
        "OPP_ALLOW_REB",
        "OPP_ALLOW_AST",
    ]

    ignore = [
        "GAME_DATE",
        "MATCHUP",
        "SEASON_ID",
        "TEAM_ABBREVIATION",
        "WL",
        "VIDEO_AVAILABLE",
        "OPP_TEAM",
    ]

    X = df.drop(columns=ignore, errors="ignore")
    X = X.select_dtypes(include=["float", "int"])

    for bc in base_cols:
        if bc in df.columns:
            X[bc] = df[bc].fillna(df[bc].median())

    X = X.dropna()
    return X


# ======================================================================
# TRAINING DATA PREPARATION
# ======================================================================

def build_training_dataset(logs):
    if logs.empty:
        return pd.DataFrame()

    df = logs.copy()
    df["OPP_TEAM"] = df["MATCHUP"].str.extract(r"(?:vs\.|@)\s(.+)$")

    df = compute_opponent_strength(df)
    df = context_features(df)

    df = df.dropna(subset=["PTS", "REB", "AST"])
    return df.reset_index(drop=True)


# ======================================================================
# MODEL CACHE: One model per player, per stat
# ======================================================================

@st.cache_resource
def get_cached_model(player_id, stat, X, y):
    manager = ModelManager(random_state=42)
    manager.train(X, y)
    return manager


# ======================================================================
# ACTIVE-ONLY PLAYER LIST
# ======================================================================

def _filter_active_players(df: pd.DataFrame, season_str: str) -> pd.DataFrame:
    """
    Robust active filter across multiple schemas.
    """
    if df.empty:
        return df

    end_year = None
    try:
        # season like "2025-26" -> 2026
        parts = season_str.split("-")
        if len(parts) == 2:
            end_year = int("20" + parts[1]) if len(parts[1]) == 2 else int(parts[1])
    except Exception:
        end_year = None

    df = df.copy()
    # normalize common fields if present
    for col in ["first_name","last_name"]:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna("")
    if "full_name" not in df.columns:
        if {"first_name","last_name"}.issubset(df.columns):
            df["full_name"] = (df["first_name"].astype(str).str.strip() + " " + df["last_name"].astype(str).str.strip()).str.strip()
        elif "PLAYER" in df.columns:
            df["full_name"] = df["PLAYER"].astype(str)
        else:
            df["full_name"] = df.get("full_name", "Unknown")

    # start with all False, then OR in signals
    active = pd.Series(False, index=df.index)

    for flag in ["is_active","IS_ACTIVE","active"]:
        if flag in df.columns:
            active = active | df[flag].astype(bool)

    for flag in ["STATUS","status","ROSTERSTATUS","rosterstatus"]:
        if flag in df.columns:
            active = active | df[flag].astype(str).str.contains("active", case=False, na=False)

    # consider to_year
    for ty in ["TO_YEAR","to_year","to", "TO"]:
        if ty in df.columns:
            try:
                to_vals = pd.to_numeric(df[ty], errors="coerce")
                if end_year is not None:
                    active = active | (to_vals >= end_year - 1)
                else:
                    active = active | (to_vals >= datetime.date.today().year - 1)
            except Exception:
                pass

    # team presence indicates active roster spot in most schemas
    team_cols = [c for c in ["team_id","TEAM_ID","teamId"] if c in df.columns]
    if team_cols:
        any_team = df[team_cols[0]].notna() & (df[team_cols[0]] != 0)
        active = active | any_team

    # if no positives, fallback to team presence only
    if not active.any() and team_cols:
        active = df[team_cols[0]].notna() & (df[team_cols[0]] != 0)

    out = df[active].copy()
    return out


@st.cache_data(show_spinner=False)
def load_player_list(season: str = "2025-26"):
    """
    Returns only active NBA players with columns: id, full_name, team_id.
    """
    try:
        p = dfetch.get_active_players_balldontlie()  # typically already active
        # normalize
        if "id" not in p.columns and "player_id" in p.columns:
            p = p.rename(columns={"player_id":"id"})
        if "full_name" not in p.columns and {"first_name","last_name"}.issubset(p.columns):
            p["full_name"] = p["first_name"] + " " + p["last_name"]
        if "team_id" not in p.columns:
            if "team" in p.columns and isinstance(p["team"].iloc[0], dict):
                p["team_id"] = p["team"].apply(lambda t: t.get("id") if isinstance(t, dict) else None)
            else:
                p["team_id"] = p.get("TEAM_ID", None)
        p = p[["id","full_name","team_id"]].dropna(subset=["id"]).drop_duplicates(subset=["id"])
        p = p.sort_values("full_name")
        return p
    except Exception:
        fb = dfetch.get_player_list_nba()
        fb = _filter_active_players(fb, season)
        # normalize id
        if "id" not in fb.columns:
            for c in ["PLAYER_ID","player_id","PersonId","PERSON_ID"]:
                if c in fb.columns:
                    fb = fb.rename(columns={c:"id"})
                    break
        # normalize team_id
        if "team_id" not in fb.columns:
            for c in ["TEAM_ID","teamId"]:
                if c in fb.columns:
                    fb = fb.rename(columns={c:"team_id"})
                    break
        # ensure columns
        fb["team_id"] = fb.get("team_id", None)
        out = fb[["id","full_name","team_id"]].dropna(subset=["id"]).drop_duplicates(subset=["id"])
        return out.sort_values("full_name")


# ======================================================================
# MAIN
# ======================================================================

def main():
    st.set_page_config(
        page_title="NBA Prop Predictor Pro",
        page_icon="🏀",
        layout="wide",
    )

    st.title("NBA Prop Predictor — Full Auto Mode")

    season_pref = "2025-26"
    players = load_player_list(season_pref)

    with st.sidebar:
        name = st.selectbox("Select Player", players["full_name"])
        row = players[players["full_name"] == name].iloc[0]
        player_id = int(row["id"])
        run = st.button("Get Predictions Now")

    if not run:
        st.info("Choose a player and click 'Get Predictions Now'")
        return

    # Load season (preferring 2025-26)
    logs = dfetch.get_player_game_logs_nba(player_id, season_pref)
    if logs.empty:
        year = datetime.date.today().year
        fallback = f"{year-1}-{str(year)[-2:]}"
        logs = dfetch.get_player_game_logs_nba(player_id, fallback)

    if logs.empty:
        st.error("No game logs found.")
        return

    df = build_training_dataset(logs)

    results = []

    for stat in PROP_MAP.keys():
        df_local = df.copy()

        # TARGET (always 1D)
        if isinstance(PROP_MAP[stat], list):
            df_local["TARGET"] = df_local[PROP_MAP[stat]].sum(axis=1)
        else:
            df_local["TARGET"] = df_local[PROP_MAP[stat]]

        df_local["TARGET"] = pd.to_numeric(df_local["TARGET"], errors="coerce")
        df_local["TARGET"] = df_local["TARGET"].astype(float)
        df_local["TARGET"] = df_local["TARGET"].squeeze()

        y = df_local["TARGET"]

        X = build_features_for_stat(df_local, stat)

        df_final = pd.concat([y, X], axis=1).dropna()

        # REMOVE DUPLICATE COLUMNS
        df_final = df_final.loc[:, ~df_final.columns.duplicated()]

        # ENSURE y_final IS 1D
        y_final = df_final["TARGET"].astype(float).values.ravel()

        X_final = df_final.drop(columns=["TARGET"])

        # Train or load cached model
        manager = get_cached_model(player_id, stat, X_final, y_final)

        X_next = X_final.tail(1)
        predictions = manager.predict(X_next)

        best = manager.best_model()

        results.append(
            {
                "Stat": stat,
                "Prediction": best.prediction,
                "Best Model": best.name,
                "MAE": best.mae,
                "MSE": best.mse,
            }
        )

    # ======================================================================
    # Metric Card Display (Mobile Friendly)
    # ======================================================================

    st.subheader("Predicted Props (AI Model Ensemble)")

    for r in results:
        with st.container():
            st.metric(
                label=f"{r['Stat']}",
                value=round(r["Prediction"], 2),
                delta=None,
            )
            st.caption(
                f"Model: {r['Best Model']} | MAE: {r['MAE']:.2f} | MSE: {r['MSE']:.2f}"
            )
            st.markdown("---")

    # ======================================================================
    # Recent Games Table
    # ======================================================================

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

