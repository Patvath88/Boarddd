"""
NBA Prop Predictor — Pro Tier
Auto-predicts ALL NBA props using custom feature selection per-stat,
individually cached models, and multi-model ML ensemble.

Stats predicted:
- Points, Rebounds, Assists
- PRA, PR, PA, RA
- 3PM
- Steals, Blocks, Turnovers
- Minutes
"""

from __future__ import annotations
import datetime
import numpy as np
import pandas as pd
import streamlit as st

import data_fetching as dfetch
from models import ModelManager


###############################################################################
# PROP DEFINITIONS
###############################################################################

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


###############################################################################
# FEATURE ENGINEERING FUNCTIONS
###############################################################################

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


###############################################################################
# CUSTOM FEATURE SELECTION PER-STAT
###############################################################################

def build_features_for_stat(df, stat):
    """Return a custom feature matrix X per stat target."""

    BASE_COLS = ["IS_HOME", "REST_DAYS", "BACK_TO_BACK",
                 "OPP_ALLOW_PTS", "OPP_ALLOW_REB", "OPP_ALLOW_AST"]

    df = df.copy()

    # Extract target columns
    if isinstance(PROP_MAP[stat], list):
        cols = PROP_MAP[stat]
    else:
        cols = [PROP_MAP[stat]]

    # Create custom engineered features for each stat
    for col in cols:
        df = lag_features(df, col)
        df = rolling_features(df, col)
        df = trend_feature(df, col)

    # Build feature matrix
    IGNORE = [
        "GAME_DATE", "MATCHUP", "SEASON_ID", "TEAM_ABBREVIATION",
        "WL", "VIDEO_AVAILABLE", "OPP_TEAM",
    ]

    X = df.drop(columns=IGNORE, errors="ignore")

    # Keep only numeric columns
    X = X.select_dtypes(include=["float","int"])

    # Add base columns (fill missing)
    for bc in BASE_COLS:
        if bc in df.columns:
            X[bc] = df[bc].fillna(df[bc].median())

    # Drop NaN introduced by rolling / trend
    X = X.dropna(axis=0)

    return X


###############################################################################
# DATASET CONSTRUCTION
###############################################################################

def build_training_dataset(game_logs):
    if game_logs.empty:
        return pd.DataFrame()

    df = game_logs.copy()
    df["OPP_TEAM"] = df["MATCHUP"].str.extract(r"(?:vs\.|@)\s(.+)$")
    df = compute_opponent_strength(df)
    df = context_features(df)

    # Remove any invalid rows early
    df = df.dropna(subset=["PTS","REB","AST"])
    return df.reset_index(drop=True)


###############################################################################
# CACHING — Train Once Per Player Per Stat
###############################################################################

@st.cache_resource
def get_cached_model(player_id, stat, X, y):
    """Train and cache a model specifically for this player & stat."""
    manager = ModelManager(random_state=42)
    manager.train(X, y)
    return manager


###############################################################################
# PLAYER LIST CACHE
###############################################################################

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


###############################################################################
# MAIN APP
###############################################################################

def main():
    st.set_page_config(page_title="NBA Prop Predictor Pro", page_icon="🏀", layout="wide")
    st.title("NBA Prop Predictor — Pro Tier")
    st.caption("Automatically predicts ALL NBA props for a selected player.")

    players = load_player_list()

    with st.sidebar:
        name = st.selectbox("Select Player", players["full_name"])
        selected = players[players["full_name"] == name].iloc[0]
        player_id = int(selected["id"])

        run = st.button("Get Prediction Now")

    if not run:
        st.info("Choose a player and click 'Get Prediction Now'")
        return

    # Fetch data
    year = datetime.date.today().year
    season = f"{year-1}-{str(year)[-2:]}"
    logs = dfetch.get_player_game_logs_nba(player_id, season)

    if logs.empty:
        st.error("No game logs found.")
        return

    df = build_training_dataset(logs)

    results = []

    # ================================
    # PREDICT ALL PROPS
    # ================================
    for stat in PROP_MAP.keys():

        y = None
        df_local = df.copy()

        # Build stat-target
        if isinstance(PROP_MAP[stat], list):
            df_local["TARGET"] = df_local[PROP_MAP[stat]].sum(axis=1)
        else:
            df_local["TARGET"] = df_local[PROP_MAP[stat]]

        y = df_local["TARGET"]

        # Build features for this stat
        X = build_features_for_stat(df_local, stat)

        # Align target and features
        df_final = pd.concat([y, X], axis=1).dropna()
        y_final = df_final["TARGET"]
        X_final = df_final.drop(columns=["TARGET"])

        # Get cached model
        manager = get_cached_model(player_id, stat, X_final, y_final)

        # Predict next row (last sample)
        X_next = X_final.tail(1)
        preds = manager.predict(X_next)
        best = manager.best_model()

        results.append({
            "Stat": stat,
            "Prediction": best.prediction,
            "Best Model": best.name,
            "MAE": best.mae,
            "MSE": best.mse
        })

    # Display results table
    st.subheader("Predicted Props")
    st.dataframe(pd.DataFrame(results), use_container_width=True)

    st.subheader("Recent Games")
    st.dataframe(
        logs[["GAME_DATE","MATCHUP","PTS","REB","AST","FG3M",
              "STL","BLK","TOV","MIN"]],
        use_container_width=True
    )


if __name__ == "__main__":
    main()
