"""
data_fetching.py — Final Version
Balldontlie + NBA API wrappers
Handles game logs, player lists, and ESPN scoreboard fallbacks.
"""

from __future__ import annotations
import datetime
import pandas as pd
import requests

BALLDONTLIE_API_KEY = "7f4db7a9-c34e-478d-a799-fef77b9d1f78"


# =============================
# ACTIVE PLAYERS
# =============================

def get_active_players_balldontlie():
    url = "https://api.balldontlie.io/v1/active_players"
    headers = {"Authorization": BALLDONTLIE_API_KEY}
    players = []
    cursor = None

    while True:
        params = {"cursor": cursor} if cursor else {}
        r = requests.get(url, headers=headers, params=params, timeout=10)
        data = r.json()
        players.extend(data.get("data", []))
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break

    return pd.DataFrame(players)


# =============================
# GAMES / LOGS
# =============================

def get_player_game_logs_nba(player_id, season, num_games=None):
    from nba_api.stats.endpoints import playergamelog
    try:
        logs = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        df = logs.get_data_frames()[0]
        if num_games:
            df = df.head(num_games)
        return df
    except:
        return pd.DataFrame()


def get_player_list_nba():
    from nba_api.stats.static import players
    try:
        return pd.DataFrame(players.get_players())
    except:
        return pd.DataFrame()
