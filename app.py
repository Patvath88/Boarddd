"""
NBA Prop Predictor — Pro Tier (Stability Patch)
Fixes:
- 1D TARGET array
- Proper SUM for combo props
- Correct season handling for 2025–2026
- Ensures X_final and y_final line up
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
# FEATURE ENGINEERING UTILITIES
# ======================================================================

def compute_opponent_strength(df):
    opp = df.groupby("OPP_TEAM")[["PTS","REB","AST"]].mean().rename(columns={
        "PTS":"OPP_ALLOW_PTS",
        "REB":"OPP_ALLOW_REB",
        "AST":"OPP_ALLOW_AST"
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
    df[f"{col}_TREND"] = df[col].rolling(5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x)==5 else np.nan,
        raw=True
    )
    return df


def context_features(df):
    df["IS_HOME"] = df["MATCHUP"].apply(lambda x: 1 if "vs" in x else 0)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["REST_DAYS"] = df["GAME_DATE"].diff().dt.days.fillna(2)
    df["BACK_TO_BACK"] = (df["REST_DAYS"] == 1).astype(int)
    return df


# ======================================================================
# CUSTOM FEATURE SET PER PROP STAT
# ======================================================================

def build_features_for_stat(df, stat):
    df = df.copy()

    # list of raw cols that belong to this stat
    if isinstance(PROP_MAP[stat], list):
        cols = PROP_MAP[stat]
    else:
        cols = [PROP_MAP[stat]]

    # build engineered features for these columns
    for col in cols:
        df = lag_features(df, col)
        df = rolling_features(df, col)
        df = trend_feature(df, col)

    # universal features
    base_cols = ["IS_HOME","REST_DAYS","BACK_TO_BACK",
                 "OPP_ALLOW_PTS","OPP_ALLOW_REB","OPP_ALLOW_AST"]

    ignore = [
        "GAME_DATE","MATCHUP","SEASON_ID","TEAM_ABBREVIATION",
        "WL","VIDEO_AVAILABLE","OPP_TEAM"
    ]

    X = df.drop(columns=ignore, errors="ignore")
    X = X.select_dtypes(include=["float","int"])

    # ensure base columns are present
    for bc in base_cols:
        if bc in df.columns:
            X[bc] = df[bc].fillna(df[bc].median())

    return X.dropna()


# ======================================================================
# DATASET BUILDER
# ======================================================================

def build_training_dataset(logs):
    if logs.empty:
        return pd.DataFrame()

    df = logs.copy()
    df["OPP_TEAM"] = df["MATCHUP"].str.extract(r"(?:vs\.|@)\s(.+)$")

    df = compute_opponent_strength(df)
    df = context_features(df)

    df = df.dropna(subset=["PTS","REB","AST"])
    return df.reset_index(drop=True)


# ======================================================================
# MODEL CACHE — one model per stat per player
# ======================================================================

@st.cache_resource
def get_cached_model(player_id, stat, X, y):
    manager = ModelManager(random_state=42)
    manager.train(X, y)
    return manager


# ======================================================================
# PLAYER LIST
# ======================================================================

@st.cache_data(show_spinner=False)
def load_player_list():
    try:
        p = dfetch.get_active_players_balldontlie()
        p["full_name"] = p["first_name"] + " " + p["last_name"]
        return p.sort_values("full_name")[["id","full_name","team_id"]]
    except:
        fallback = dfetch.get_player_list_nba()
        fallback["full_name"] = fallback["full_name"]
        fallback["team_id"] = None
        return fallback[["id","full_name","team_id"]]


# ======================================================================
# MAIN APP
# ======================================================================

def main():
    st.set_page_config(page_title="NBA Prop Predictor Pro", page_icon="🏀", layout="wide")
    st.title("NBA Prop Predictor — Full Auto-Prop Mode")

    players = load_player_list()

    with st.sidebar:
        name = st.selectbox("Select Player", players["full_name"])
        row = players[players["full_name"] == name].iloc[0]
        player_id = int(row["id"])
        run = st.button("Get Predictions Now")

    if not run:
        st.info("Choose a player and click 'Get Predictions Now'")
        return

    # ================================
    # USE 2025–2026 SEASON FIRST
    # ================================
    season_str = "2025-26"

    logs = dfetch.get_player_game_logs_nba(player_id, season_str)

    if logs.empty():
        st.warning("2025–26 season not found, falling back to current season")
        year = datetime.date.today().year
        fallback_season = f"{year-1}-{str(year)[-2:]}"
        logs = dfetch.get_player_game_logs_nba(player_id, fallback_season)

    if logs.empty:
        st.error("No game logs found.")
        return

    df = build_training_dataset(logs)

    results = []

    for stat in PROP_MAP.keys():

        df_local = df.copy()

        # build TARGET safely (1D)
        if isinstance(PROP_MAP[stat], list):
            df_local["TARGET"] = df_local[PROP_MAP[stat]].sum(axis=1)
        else:
            df_local["TARGET"] = df_local[PROP_MAP[stat]]

        # enforce numeric 1D
        df_local["TARGET"] = pd.to_numeric(df_local["TARGET"], errors="coerce")
        df_local["TARGET"] = df_local["TARGET"].astype(float)
        df_local["TARGET"] = df_local["TARGET"].squeeze()

        y = df_local["TARGET"]

        # build custom features
        X = build_features_for_stat(df_local, stat)

        # align
        df_final = pd.concat([y, X], axis=1).dropna()
        y_final = df_final["TARGET"]
        X_final = df_final.drop(columns=["TARGET"])

        # train or load cached model
        manager = get_cached_model(player_id, stat, X_final, y_final)

        X_next = X_final.tail(1)
        predictions = manager.predict(X_next)
        best = manager.best_model()

        results.append({
            "Stat": stat,
            "Prediction": best.prediction,
            "Best Model": best.name,
            "MAE": best.mae,
            "MSE": best.mse
        })

    st.subheader("Predicted Props")
    st.dataframe(pd.DataFrame(results), use_container_width=True)

    st.subheader("Recent Games")
    st.dataframe(
        logs[["GAME_DATE","MATCHUP","PTS","REB","AST",
              "FG3M","STL","BLK","TOV","MIN"]],
        use_container_width=True
    )


if __name__ == "__main__":
    main()
