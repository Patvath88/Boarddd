# app.py
"""
NBA Prop Predictor — Elite (IndexError-safe)
- Guards empty players/selectbox resolution (no .iloc[0] crashes)
- Safe player picking via pick_player_row()
- Auto opponent only if team_id is valid
- REST_DAYS robust even if GAME_DATE missing
- All prior features preserved (DEF/PACE, DEF×PACE, photos/logos, glow favorites with ❌,
  backtests, CSV, share images, bar charts)
"""

from __future__ import annotations

import os
import io
import uuid
import json
import hashlib
import zipfile
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import altair as alt
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# Your modules (must exist)
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

BASE_COLS = [
    "IS_HOME", "REST_DAYS", "BACK_TO_BACK",
    "OPP_ALLOW_PTS", "OPP_ALLOW_REB", "OPP_ALLOW_AST",
    "OPP_DEF_PPG", "OPP_DEF_Z",
    "OPP_PACE", "OPP_PACE_Z",
    "OPP_DEF_X_PACE",
]

N_TRAIN = 60
MIN_ROWS_FOR_MODEL = 12
MAX_WORKERS = max(2, min(8, os.cpu_count() or 4))

DATA_DIR = Path("./data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
PRED_FILE = DATA_DIR / "predictions.jsonl"
FAV_FILE = DATA_DIR / "favorites.json"


# =============================================================================
# TEAM META (colors + NBA logo ids)
# =============================================================================

TEAM_META: Dict[str, Dict[str, str | int]] = {
    "ATL": {"color": "#E03A3E", "nba_id": 1610612737, "name": "Hawks"},
    "BOS": {"color": "#007A33", "nba_id": 1610612738, "name": "Celtics"},
    "BKN": {"color": "#000000", "nba_id": 1610612751, "name": "Nets"},
    "CHA": {"color": "#1D1160", "nba_id": 1610612766, "name": "Hornets"},
    "CHI": {"color": "#CE1141", "nba_id": 1610612741, "name": "Bulls"},
    "CLE": {"color": "#860038", "nba_id": 1610612739, "name": "Cavaliers"},
    "DAL": {"color": "#00538C", "nba_id": 1610612742, "name": "Mavericks"},
    "DEN": {"color": "#0E2240", "nba_id": 1610612743, "name": "Nuggets"},
    "DET": {"color": "#C8102E", "nba_id": 1610612765, "name": "Pistons"},
    "GSW": {"color": "#1D428A", "nba_id": 1610612744, "name": "Warriors"},
    "HOU": {"color": "#CE1141", "nba_id": 1610612745, "name": "Rockets"},
    "IND": {"color": "#002D62", "nba_id": 1610612754, "name": "Pacers"},
    "LAC": {"color": "#C8102E", "nba_id": 1610612746, "name": "Clippers"},
    "LAL": {"color": "#552583", "nba_id": 1610612747, "name": "Lakers"},
    "MEM": {"color": "#5D76A9", "nba_id": 1610612763, "name": "Grizzlies"},
    "MIA": {"color": "#98002E", "nba_id": 1610612748, "name": "Heat"},
    "MIL": {"color": "#00471B", "nba_id": 1610612749, "name": "Bucks"},
    "MIN": {"color": "#0C2340", "nba_id": 1610612750, "name": "Timberwolves"},
    "NOP": {"color": "#0C2340", "nba_id": 1610612740, "name": "Pelicans"},
    "NYK": {"color": "#006BB6", "nba_id": 1610612752, "name": "Knicks"},
    "OKC": {"color": "#007AC1", "nba_id": 1610612760, "name": "Thunder"},
    "ORL": {"color": "#0077C0", "nba_id": 1610612753, "name": "Magic"},
    "PHI": {"color": "#006BB6", "nba_id": 1610612755, "name": "76ers"},
    "PHX": {"color": "#1D1160", "nba_id": 1610612756, "name": "Suns"},
    "POR": {"color": "#E03A3E", "nba_id": 1610612757, "name": "Trail Blazers"},
    "SAC": {"color": "#5A2D81", "nba_id": 1610612758, "name": "Kings"},
    "SAS": {"color": "#C4CED4", "nba_id": 1610612759, "name": "Spurs"},
    "TOR": {"color": "#CE1141", "nba_id": 1610612761, "name": "Raptors"},
    "UTA": {"color": "#002B5C", "nba_id": 1610612762, "name": "Jazz"},
    "WAS": {"color": "#002B5C", "nba_id": 1610612764, "name": "Wizards"},
}

def nba_logo_url(team_abbr: str) -> Optional[str]:
    meta = TEAM_META.get(team_abbr or "")
    if not meta:
        return None
    tid = meta["nba_id"]
    return f"https://cdn.nba.com/logos/nba/{tid}/global/L/logo.png"


# =============================================================================
# STYLING
# =============================================================================

def inject_css() -> None:
    st.markdown(
        """
<style>
html, body { font-family: Inter, ui-sans-serif, system-ui; }
.block-container { padding-top: 1.2rem; max-width: 1240px; }
h1, h2, h3, h4 {
  background: linear-gradient(90deg,#e2e8f0 0%, #60a5fa 40%, #34d399 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.card {
  background: rgba(17, 24, 39, 0.6);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
  backdrop-filter: blur(6px);
}
.stButton>button {
  border-radius: 12px; padding: 10px 14px; font-weight: 600;
  background: linear-gradient(90deg,#0ea5e9,#22c55e); border: none;
}
.stButton>button:hover { filter: brightness(1.05); }
.tag {
  display:inline-block; padding:.2rem .55rem; border-radius:999px; font-size:.75rem;
  background:rgba(99,102,241,.15); border:1px solid rgba(99,102,241,.35);
}
.badge { display:inline-block; padding:.25rem .6rem; border-radius:10px; font-size:.8rem;
  background:rgba(34,197,94,.15); border:1px solid rgba(34,197,94,.4); color:#d1fae5; }
[data-testid="stDataFrame"] { border-radius: 12px; border: 1px solid rgba(255,255,255,0.08); }

/* favorites grid */
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 16px; }
.glow-card { padding: 14px; border-radius: 16px; border: 1px solid rgba(255,255,255,.08); background: #0b1220; }
.glow-hdr { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
.glow-name { font-weight: 700; color: #e5e7eb; }
.glow-meta { font-size: .9rem; color: #cbd5e1; }
</style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# UTILITIES
# =============================================================================

def _rerun():
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

def _rolling_slope(values: np.ndarray, window: int) -> np.ndarray:
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

def season_start_year(season: str) -> int:
    return int(season.split("-")[0])

def prev_season(season: str) -> str:
    s0 = season_start_year(season)
    return f"{s0-1}-{str(s0)[-2:]}"


# =============================================================================
# PERSISTENCE
# =============================================================================

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
# PHOTOS
# =============================================================================

@st.cache_data(show_spinner=False)
def get_player_photo_bytes(player_id: int, nba_person_id: Optional[int] = None) -> Optional[bytes]:
    ids = list({v for v in [nba_person_id, player_id] if v is not None})
    if not ids: return None
    url_templates = [
        "https://cdn.nba.com/headshots/nba/latest/1040x760/{pid}.png",
        "https://cdn.nba.com/headshots/nba/latest/260x190/{pid}.png",
        "https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{pid}.png",
        "https://cdn.balldontlie.io/images/headshots/{pid}.png",
    ]
    s = requests.Session(); s.headers.update({"User-Agent": "Mozilla/5.0 (PropPredictor)"})
    for pid in ids:
        for tmpl in url_templates:
            try:
                r = s.get(tmpl.format(pid=int(pid)), timeout=5)
                if r.status_code == 200 and r.content and len(r.content) > 1500:
                    return r.content
            except Exception:
                continue
    return None

def _safe_image_from_bytes(photo_bytes: Optional[bytes], size=(220, 220)) -> Image.Image:
    if not photo_bytes:
        img = Image.new("RGB", size, (15, 23, 42))
        d = ImageDraw.Draw(img)
        d.ellipse([size[0]//4, size[1]//4, 3*size[0]//4, 3*size[1]//4], outline=(80, 90, 110), width=3)
        return img
    try:
        im = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
        return im.resize(size)
    except Exception:
        return Image.new("RGB", size, (15, 23, 42))


# =============================================================================
# TEAMS / PLAYERS (robust)
# =============================================================================

@st.cache_data(show_spinner=False)
def load_teams_bdl() -> pd.DataFrame:
    try:
        r = requests.get("https://www.balldontlie.io/api/v1/teams", timeout=8)
        r.raise_for_status()
        data = r.json().get("data", [])
        return pd.DataFrame(data)[["id","abbreviation","full_name"]]
    except Exception:
        return pd.DataFrame(columns=["id","abbreviation","full_name"])

def _safe_get_team_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "team" in out.columns and out["team"].notna().any():
        out["team_id"] = out["team"].apply(lambda t: t.get("id") if isinstance(t, dict) else None).astype("Int64")
        out["team_abbr"] = out["team"].apply(lambda t: t.get("abbreviation") if isinstance(t, dict) else None)
    for c in ["TEAM_ID", "teamId", "TeamID"]:
        if "team_id" not in out.columns and c in out.columns:
            out = out.rename(columns={c: "team_id"})
    for c in ["TEAM_ABBREVIATION", "team_abbreviation", "TeamAbbreviation"]:
        if "team_abbr" not in out.columns and c in out.columns:
            out = out.rename(columns={c: "team_abbr"})
    return out

def normalize_players_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["id","full_name","team_id","team_abbr","nba_person_id"])
    p = df.copy()

    # id normalization
    id_done = False
    for c in ["id","player_id","PLAYER_ID","PersonId","PERSON_ID"]:
        if c in p.columns:
            if c != "id": p = p.rename(columns={c: "id"})
            id_done = True; break
    if not id_done:
        p["id"] = pd.NA

    # full_name normalization
    if "full_name" not in p.columns:
        if {"first_name","last_name"}.issubset(p.columns):
            p["full_name"] = (p["first_name"].astype(str).str.strip() + " " + p["last_name"].astype(str).str.strip()).str.strip()
        elif "PLAYER" in p.columns:
            p["full_name"] = p["PLAYER"].astype(str)
        elif "DISPLAY_FIRST_LAST" in p.columns:
            p["full_name"] = p["DISPLAY_FIRST_LAST"].astype(str)
        else:
            p["full_name"] = p.get("full_name","Unknown")

    # team cols
    p = _safe_get_team_cols(p)
    if "team_abbr" not in p.columns or p["team_abbr"].isna().all():
        teams = load_teams_bdl()
        if "team_id" in p.columns and not p["team_id"].isna().all() and not teams.empty:
            p = p.merge(
                teams.rename(columns={"id":"team_id","abbreviation":"team_abbr"}),
                on="team_id", how="left"
            )

    # nba_person_id normalization
    nba_done = False
    for c in ["nba_person_id","PERSON_ID","personId","nba_id"]:
        if c in p.columns:
            if c != "nba_person_id": p = p.rename(columns={c: "nba_person_id"})
            nba_done = True; break
    if not nba_done:
        p["nba_person_id"] = pd.NA

    for col in ["team_id","team_abbr"]:
        if col not in p.columns: p[col] = pd.NA

    cols = ["id","full_name","team_id","team_abbr","nba_person_id"]
    out = p[cols].dropna(subset=["id"]).drop_duplicates(subset=["id"]).sort_values("full_name").reset_index(drop=True)
    return out

@st.cache_data(show_spinner=False)
def load_player_list(season: str = "2025-26") -> pd.DataFrame:
    try:
        raw = dfetch.get_active_players_balldontlie()
        p = normalize_players_df(raw)
        return p[["id","full_name","team_id","team_abbr","nba_person_id"]]
    except Exception:
        fb = dfetch.get_player_list_nba()
        p = normalize_players_df(_filter_active_players(fb, season))
        return p[["id","full_name","team_id","team_abbr","nba_person_id"]]


def _filter_active_players(df: pd.DataFrame, season_str: str) -> pd.DataFrame:
    if df.empty: return df
    try:
        parts = season_str.split("-"); end_year = int("20"+parts[1]) if len(parts)==2 and len(parts[1])==2 else int(parts[1])
    except Exception:
        end_year = dt.date.today().year
    df = df.copy()
    if "full_name" not in df.columns:
        if {"first_name","last_name"}.issubset(df.columns):
            df["full_name"] = (df["first_name"].astype(str).str.strip()+" "+df["last_name"].astype(str).str.strip()).str.strip()
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
    if team_cols: active = active | (df[team_cols[0]].notna() & (df[team_cols[0]] != 0))
    if not active.any() and team_cols: active = df[team_cols[0]].notna() & (df[team_cols[0]] != 0)
    return df[active].copy()


@st.cache_data(show_spinner=False)
def load_logs(player_id: int, season: str) -> pd.DataFrame:
    return dfetch.get_player_game_logs_nba(player_id, season).copy()


# =============================================================================
# AUTO NEXT OPPONENT + LEAGUE DEFENSE & PACE
# =============================================================================

def _bdl_paginate(url: str, params: Dict) -> List[Dict]:
    out: List[Dict] = []
    s = requests.Session(); s.headers.update({"User-Agent": "Mozilla/5.0 (PropPredictor)"})
    page = 1
    while True:
        q = params.copy(); q["page"] = page; q.setdefault("per_page", 100)
        try:
            r = s.get(url, params=q, timeout=8); r.raise_for_status()
            j = r.json(); out.extend(j.get("data", []))
            next_page = j.get("meta", {}).get("next_page")
            if not next_page: break
            page = next_page
        except Exception:
            break
    return out

@st.cache_data(show_spinner=False)
def load_team_defense_pace(season: str) -> pd.DataFrame:
    year = season_start_year(season)
    games = _bdl_paginate("https://www.balldontlie.io/api/v1/games", {"seasons[]": year})
    if not games:
        return pd.DataFrame(columns=["team_id","abbreviation","OPP_DEF_PPG","OPP_DEF_Z","PACE","PACE_Z"])

    rows = []
    for g in games:
        h = g.get("home_team", {}); v = g.get("visitor_team", {})
        hs = g.get("home_team_score", 0); vs = g.get("visitor_team_score", 0)
        total = (hs + vs)
        rows.append({"team_id": h.get("id"), "abbreviation": h.get("abbreviation"), "allowed": vs, "total": total})
        rows.append({"team_id": v.get("id"), "abbreviation": v.get("abbreviation"), "allowed": hs, "total": total})

    df = pd.DataFrame(rows).dropna(subset=["team_id"])
    agg = df.groupby(["team_id","abbreviation"], as_index=False).agg(OPP_DEF_PPG=("allowed","mean"), PACE=("total","mean"))
    mu_d, sd_d = agg["OPP_DEF_PPG"].mean(), agg["OPP_DEF_PPG"].std(ddof=0) or 1.0
    mu_p, sd_p = agg["PACE"].mean(), agg["PACE"].std(ddof=0) or 1.0
    agg["OPP_DEF_Z"] = (agg["OPP_DEF_PPG"] - mu_d) / sd_d
    agg["PACE_Z"] = (agg["PACE"] - mu_p) / sd_p
    return agg

@st.cache_data(show_spinner=False)
def auto_next_opponent(team_id: int, season: str) -> Optional[Dict]:
    if not team_id or int(team_id) <= 0:
        return None
    year = season_start_year(season)
    today = dt.date.today().isoformat()
    s = requests.Session(); s.headers.update({"User-Agent": "Mozilla/5.0 (PropPredictor)"})
    try:
        r = s.get(
            "https://www.balldontlie.io/api/v1/games",
            params={"seasons[]": year, "team_ids[]": team_id, "start_date": today, "per_page": 100},
            timeout=8,
        )
        j = r.json(); data = j.get("data", [])
        if not data: return None
        def _d(x): return dt.datetime.fromisoformat(x["date"].replace("Z","+00:00"))
        nxt = sorted(data, key=_d)[0]
        h, v = nxt["home_team"], nxt["visitor_team"]
        is_home = (h["id"] == team_id)
        opp = v if is_home else h
        when = _d(nxt["date"]).date()
        return {"opp_id": opp["id"], "opp_abbr": opp["abbreviation"], "date": when, "is_home": is_home}
    except Exception:
        return None


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def compute_opponent_strength(df: pd.DataFrame) -> pd.DataFrame:
    opp = (
        df.groupby("OPP_TEAM")[["PTS", "REB", "AST"]]
        .mean()
        .rename(columns={"PTS":"OPP_ALLOW_PTS","REB":"OPP_ALLOW_REB","AST":"OPP_ALLOW_AST"})
    )
    return df.join(opp, on="OPP_TEAM")

def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["IS_HOME"] = df["MATCHUP"].apply(lambda x: 1 if isinstance(x, str) and ("vs" in x) else 0)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df = df.sort_values("GAME_DATE")
    df["REST_DAYS"] = df["GAME_DATE"].diff().dt.days.fillna(2)
    df["BACK_TO_BACK"] = (df["REST_DAYS"] == 1).astype(int)
    return df

def attach_defense_pace(df: pd.DataFrame, defense_pace: pd.DataFrame) -> pd.DataFrame:
    if defense_pace is None or defense_pace.empty:
        for c in ["OPP_DEF_PPG","OPP_DEF_Z","OPP_PACE","OPP_PACE_Z","OPP_DEF_X_PACE"]:
            df[c] = np.nan
        return df
    m = defense_pace.rename(columns={"abbreviation":"OPP_TEAM"})
    df = df.merge(m[["OPP_TEAM","OPP_DEF_PPG","OPP_DEF_Z","PACE","PACE_Z"]], on="OPP_TEAM", how="left")
    df = df.rename(columns={"PACE":"OPP_PACE","PACE_Z":"OPP_PACE_Z"})
    df["OPP_DEF_X_PACE"] = df["OPP_DEF_Z"] * df["OPP_PACE_Z"]
    return df

def _ensure_training_base(df: pd.DataFrame, season: str) -> pd.DataFrame:
    df = df.copy()
    df["OPP_TEAM"] = df["MATCHUP"].astype(str).str.extract(r"(?:vs\.|@)\s(.+)$")
    df = compute_opponent_strength(df)
    df = add_context_features(df)
    df = df.dropna(subset=["PTS","REB","AST"])
    dp = load_team_defense_pace(season)
    df = attach_defense_pace(df, dp)
    return df.reset_index(drop=True)

@st.cache_data(show_spinner=False)
def build_all_features(df_in: pd.DataFrame, season: str) -> pd.DataFrame:
    df = _ensure_training_base(df_in, season)
    out = df.copy()
    for c in STAT_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    for c in STAT_COLUMNS:
        s = out[c].to_numpy(dtype=float, copy=False)
        out[f"{c}_L1"]  = np.roll(s, 1); 
        if len(out.index) > 0:
            out.loc[out.index[0], f"{c}_L1"] = np.nan
        out[f"{c}_L3"]  = out[c].shift(3)
        out[f"{c}_L5"]  = out[c].shift(5)
        out[f"{c}_AVG5"]  = pd.Series(s).rolling(5, min_periods=1).mean().to_numpy()
        out[f"{c}_AVG10"] = pd.Series(s).rolling(10, min_periods=1).mean().to_numpy()
        out[f"{c}_TREND"] = _rolling_slope(s, 5)
    for bc in BASE_COLS:
        if bc in out.columns:
            med = float(out[bc].median()) if out[bc].notna().any() else 0.0
            out[bc] = out[bc].fillna(med)
    return out

def select_X_for_stat(features: pd.DataFrame, stat: str) -> pd.DataFrame:
    cols = PROP_MAP[stat] if isinstance(PROP_MAP[stat], list) else [PROP_MAP[stat]]
    bits: List[str] = []
    for c in cols:
        bits.extend([f"{c}_L1", f"{c}_L3", f"{c}_L5", f"{c}_AVG5", f"{c}_AVG10", f"{c}_TREND"])
    keep = [c for c in set(BASE_COLS) | set(bits) if c in features.columns]
    return features[keep].copy()

def build_target(df: pd.DataFrame, stat: str) -> pd.Series:
    tgt = df[PROP_MAP[stat]].sum(axis=1) if isinstance(PROP_MAP[stat], list) else df[PROP_MAP[stat]]
    return pd.to_numeric(tgt, errors="coerce").astype(float)

def _impute_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X = X.ffill()
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
    manager = ModelManager(random_state=42)
    manager.train(X, y)
    ss[key] = manager
    return manager

def train_predict_for_stat(
    player_id: int,
    season: str,
    stat: str,
    features: pd.DataFrame,
    fast_mode: bool,
    upcoming_ctx: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
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

    X_next = X_final.tail(1).copy()
    if upcoming_ctx:
        for k, v in upcoming_ctx.items():
            if k in X_next.columns:
                X_next.loc[:, k] = v

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
# TABLE + CHARTS + SHARE
# =============================================================================

def results_to_table(results: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    order = {k: i for i, k in enumerate(PROP_MAP.keys())}
    df["__order"] = df["Stat"].map(order)
    df = df.sort_values("__order").drop(columns="__order")
    df = df.rename(columns={"Stat":"Stat","Prediction":"Pred","Best Model":"Model","MAE":"MAE","MSE":"MSE"})
    df["Pred"] = pd.to_numeric(df["Pred"], errors="coerce").round(2)
    df["MAE"]  = pd.to_numeric(df["MAE"],  errors="coerce").round(2)
    df["MSE"]  = pd.to_numeric(df["MSE"],  errors="coerce").round(2)
    return df[["Stat","Pred","Model","MAE","MSE"]]

def table_downloaders(df: pd.DataFrame, filename_prefix: str) -> None:
    csv = df.to_csv(index=False)
    st.download_button("⬇️ Download CSV", data=csv.encode("utf-8"), file_name=f"{filename_prefix}.csv", mime="text/csv")
    with st.expander("Copy CSV text"):
        st.text_area("CSV", value=csv, height=160)

def make_share_image(player_name: str, season: str, photo_bytes: Optional[bytes], table_df: pd.DataFrame, next_info: str) -> bytes:
    W, H = 1200, 675
    bg = Image.new("RGB", (W, H), color=(12, 17, 28))
    draw = ImageDraw.Draw(bg)
    for x in range(W):
        r = int(14 + 40 * x / W); g = int(100 + 120 * x / W); b = int(80 + 80 * x / W)
        draw.line([(x, 0), (x, 8)], fill=(r, g, b))
    head = _safe_image_from_bytes(photo_bytes, size=(240, 240))
    bg.paste(head, (50, 80))
    font_big = ImageFont.load_default(); font_med = ImageFont.load_default()
    draw.text((310, 90), f"NBA Prop Predictor — Elite", fill=(224, 231, 255), font=font_big)
    draw.text((310, 120), f"{player_name}  •  {season}", fill=(148, 163, 184), font=font_med)
    draw.text((310, 150), next_info, fill=(120, 130, 145), font=font_med)
    table = table_df.copy().head(8)
    col_x = [310, 540, 720, 875, 1010]
    headers = ["Stat","Pred","Model","MAE","MSE"]
    for cx, htxt in zip(col_x, headers): draw.text((cx, 200), htxt, fill=(203, 213, 225), font=font_med)
    y = 230
    for _, r in table.iterrows():
        vals = [str(r["Stat"]), f"{r['Pred']}", str(r["Model"]), f"{r['MAE']}", f"{r['MSE']}"]
        for cx, v in zip(col_x, vals): draw.text((cx, y), v, fill=(226, 232, 240), font=font_med)
        y += 28
    draw.text((50, H-40), "Share generated by NBA Prop Predictor — Elite", fill=(120, 130, 145), font=font_med)
    buf = io.BytesIO(); bg.save(buf, format="PNG", optimize=True); return buf.getvalue()

def bar_chart_from_table(df: pd.DataFrame, title: str, color: str | None = None):
    c = alt.Chart(df).mark_bar(color=color).encode(
        x=alt.X("Stat:N", sort=df["Stat"].tolist()),
        y=alt.Y("Pred:Q"),
        tooltip=["Stat","Pred","Model","MAE","MSE"],
    ).properties(height=280, title=title)
    st.altair_chart(c, use_container_width=True)


# =============================================================================
# SAFE PLAYER PICKER
# =============================================================================

def pick_player_row(players: pd.DataFrame, selected_name: str) -> Optional[pd.Series]:
    """Resolve selected player safely (no out-of-bounds)."""
    if players is None or players.empty:
        return None
    names = players["full_name"].astype(str).values
    # Resolve by exact match index; fallback to first row if not found
    idx = np.where(names == str(selected_name))[0]
    if idx.size:
        return players.iloc[int(idx[0])]
    # Fallback if somehow label not found
    return players.iloc[0]


# =============================================================================
# PAGES
# =============================================================================

def page_predict(players: pd.DataFrame):
    st.header("NBA Prop Predictor — Elite")
    st.caption("Auto opponent • Defense + Pace • Copyable table • Share image")

    if players is None or players.empty:
        st.error("No active players available. Please reload the app or try again later.")
        return

    names_list = players["full_name"].astype(str).tolist()
    col_left, col_right = st.columns([1, 3])
    with col_left:
        name = st.selectbox("Select Player", names_list, key="predict_player")
        row = pick_player_row(players, name)
        if row is None:
            st.error("Could not resolve the selected player."); 
            return
        player_id = int(row["id"])
        team_id = int(row.get("team_id", 0) or 0)
        team_abbr = str(row.get("team_abbr") or "")
        team_color = TEAM_META.get(team_abbr, {}).get("color", "#60a5fa")
        nba_pid = int(row["nba_person_id"]) if "nba_person_id" in row and not pd.isna(row["nba_person_id"]) else None
        fast_mode = st.toggle("Fast mode (no training)", value=False, key="fast_toggle")
        run = st.button("Get Predictions Now")
    with col_right:
        photo = get_player_photo_bytes(player_id, nba_pid)
        st.image(_safe_image_from_bytes(photo, (240, 240)), caption=name)

    if not run:
        st.info("Choose a player and click **Get Predictions Now**")
        return

    season = "2025-26"
    logs = load_logs(player_id, season)
    if logs.empty:
        year = dt.date.today().year
        fallback = f"{year-1}-{str(year)[-2:]}"
        logs = load_logs(player_id, fallback)
        season = fallback
    if logs.empty:
        st.error("No game logs found."); return

    next_game = auto_next_opponent(team_id, season) if team_id > 0 else None
    with st.spinner("Building features…"):
        features = build_all_features(logs, season)

    # SAFER REST DAYS + badge
    upcoming_ctx = {}
    next_info = "Next: N/A"
    if next_game:
        dp = load_team_defense_pace(season)
        opp_row = dp[dp["abbreviation"] == next_game["opp_abbr"]] if not dp.empty else pd.DataFrame()
        if not opp_row.empty:
            upcoming_ctx["OPP_DEF_PPG"] = float(opp_row["OPP_DEF_PPG"].iloc[0])
            upcoming_ctx["OPP_DEF_Z"]   = float(opp_row["OPP_DEF_Z"].iloc[0])
            upcoming_ctx["OPP_PACE"]    = float(opp_row["PACE"].iloc[0])
            upcoming_ctx["OPP_PACE_Z"]  = float(opp_row["PACE_Z"].iloc[0])
            upcoming_ctx["OPP_DEF_X_PACE"] = upcoming_ctx["OPP_DEF_Z"] * upcoming_ctx["OPP_PACE_Z"]
        if "GAME_DATE" in features.columns and features["GAME_DATE"].notna().any():
            last_date = pd.to_datetime(features["GAME_DATE"]).max().date()
            rest_days = max(0, (next_game["date"] - last_date).days)
        else:
            rest_days = 2
        upcoming_ctx["IS_HOME"] = 1 if next_game["is_home"] else 0
        upcoming_ctx["REST_DAYS"] = rest_days
        upcoming_ctx["BACK_TO_BACK"] = 1 if rest_days == 1 else 0
        next_info = f"Next: {('Home' if next_game['is_home'] else 'Away')} vs {next_game['opp_abbr']} on {next_game['date']} · Rest {rest_days}d"
    st.markdown(f'<span class="badge">{next_info}</span>', unsafe_allow_html=True)

    with st.spinner("Training models & predicting…"):
        futures, results = {}, []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for stat in PROP_MAP.keys():
                futures[ex.submit(train_predict_for_stat, player_id, season, stat, features, fast_mode, upcoming_ctx)] = stat
            for fut in as_completed(futures): results.append(fut.result())

    df_table = results_to_table(results)
    st.subheader("Predicted Props — Table")
    st.dataframe(df_table, use_container_width=True, hide_index=True)
    table_downloaders(df_table, filename_prefix=f"{name.replace(' ','_')}_{season}_predictions")

    bar_chart_from_table(df_table, title="Predictions (bars)", color=team_color)

    img_bytes = make_share_image(name, season, photo, df_table, next_info)
    st.download_button("📸 Share image (PNG)", data=img_bytes, file_name=f"{name.replace(' ','_')}_{season}_predictions.png", mime="image/png")

    st.divider()
    st.subheader("Recent Games")
    cols_show = [c for c in ["GAME_DATE","MATCHUP","PTS","REB","AST","FG3M","STL","BLK","TOV","MIN"] if c in logs.columns]
    st.dataframe(logs[cols_show], use_container_width=True)

    if st.button("💾 Save this run"):
        rows = _read_jsonl(PRED_FILE)
        rec = {
            "id": str(uuid.uuid4()),
            "ts": dt.datetime.utcnow().isoformat(),
            "player_id": player_id,
            "player_name": name,
            "season": season,
            "fast_mode": fast_mode,
            "next_info": next_info,
            "results": df_table.to_dict(orient="records"),
        }
        rows.append(rec); _write_jsonl(PRED_FILE, rows)
        st.success("Saved.")


def _walk_forward_backtest_internal(player_id: int, season: str, stat: str, features: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    y_all = build_target(features, stat)
    X_all = select_X_for_stat(features, stat)
    use_cols = ["GAME_DATE"] if "GAME_DATE" in features.columns else []
    df = pd.concat([features[use_cols], y_all.rename("TARGET"), X_all], axis=1)
    df = df.dropna(subset=["TARGET"]).reset_index(drop=True)

    preds, truth, dates = [], [], []
    start_idx = max(MIN_ROWS_FOR_MODEL + 5, 12)
    start_idx = min(start_idx, max(len(df) - 1, 1))
    for t in range(start_idx, len(df)):
        train = df.iloc[max(0, t - N_TRAIN): t]
        test = df.iloc[[t]]
        y_tr = train["TARGET"].to_numpy()
        X_tr = _impute_features(train.drop(columns=[c for c in ["TARGET", "GAME_DATE"] if c in train.columns]))
        X_te = _impute_features(test.drop(columns=[c for c in ["TARGET", "GAME_DATE"] if c in test.columns]))
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
        if "GAME_DATE" in test.columns:
            dates.append(pd.to_datetime(test["GAME_DATE"]).iloc[0])
        else:
            dates.append(pd.NaT)

    out = pd.DataFrame({"GAME_DATE": dates, "y_true": truth, "y_pred": preds})
    out["abs_err"] = (out["y_true"] - out["y_pred"]).abs()
    out["sq_err"] = (out["y_true"] - out["y_pred"]) ** 2
    mae = float(out["abs_err"].mean()) if len(out) else float("nan")
    rmse = float(np.sqrt(out["sq_err"].mean())) if len(out) else float("nan")
    return out, {"MAE": mae, "RMSE": rmse, "N": int(len(out))}

def page_backtest(players: pd.DataFrame):
    st.header("Backtesting")

    if players is None or players.empty:
        st.error("No active players available for backtesting.")
        return

    names_list = players["full_name"].astype(str).tolist()
    name = st.selectbox("Player", names_list, key="bt_player")
    prow = pick_player_row(players, name)
    if prow is None:
        st.error("Could not resolve the selected player.")
        return

    player_id = int(prow["id"])
    team_abbr = str(prow.get("team_abbr") or "")
    team_color = TEAM_META.get(team_abbr, {}).get("color", "#60a5fa")

    season = "2025-26"
    do_this = st.checkbox("This season", value=True)
    do_last = st.checkbox("Last season", value=True)
    run = st.button("Run Backtests")
    if not run: return

    seasons = []
    if do_this: seasons.append(season)
    if do_last: seasons.append(prev_season(season))

    for s in seasons:
        logs = load_logs(player_id, s)
        if logs.empty:
            st.warning(f"No logs for {s}."); 
            continue
        features = build_all_features(logs, s)

        st.subheader(f"{name} — {s}")
        s_rows = []
        for stat in PROP_MAP.keys():
            df_bt, summary = _walk_forward_backtest_internal(player_id, s, stat, features)
            st.markdown(f"**{stat}** — MAE: `{summary['MAE']:.2f}` · RMSE: `{summary['RMSE']:.2f}` · N: `{summary['N']}`")
            s_rows.append({"Stat": stat, **summary})
        df_sum = pd.DataFrame(s_rows).set_index("Stat")
        st.dataframe(df_sum, use_container_width=True)
        chart_df = df_sum.reset_index()[["Stat","MAE"]]
        c = alt.Chart(chart_df).mark_bar(color=team_color).encode(x=alt.X("Stat:N", sort=chart_df["Stat"].tolist()), y="MAE:Q").properties(height=280, title="Backtest MAE by Stat")
        st.altair_chart(c, use_container_width=True)


def page_favorites(players: pd.DataFrame):
    st.header("Favorites (Auto Opp • Defense & Pace • Glow Cards)")

    if players is None or players.empty:
        st.error("No active players available to add to favorites.")
        return

    favs = _load_favorites()

    # Add favorite
    col1, col2 = st.columns([3,1])
    with col1:
        names_list = players["full_name"].astype(str).tolist()
        name = st.selectbox("Add a player", names_list, key="fav_add_sel")
    with col2:
        if st.button("➕ Add"):
            row = pick_player_row(players, name)
            if row is None:
                st.error("Could not resolve the selected player.")
            else:
                pid = int(row["id"])
                nba_pid = int(row["nba_person_id"]) if "nba_person_id" in row and not pd.isna(row["nba_person_id"]) else None
                team_id = int(row.get("team_id", 0) or 0)
                team_abbr = str(row.get("team_abbr") or "")
                if not any(f["id"] == pid for f in favs):
                    favs.append({"id": pid, "full_name": name, "team_id": team_id, "team_abbr": team_abbr, "nba_person_id": nba_pid})
                    _save_favorites(favs); st.success("Added to favorites.")
                else:
                    st.info("Already in favorites.")

    # List favorites as glow cards with ❌
    if favs:
        st.subheader("Saved favorites")
        cols = st.columns(3)
        for i, f in enumerate(list(favs)):
            abbr = f.get("team_abbr") or ""
            color = TEAM_META.get(abbr, {}).get("color", "#60a5fa")
            logo = nba_logo_url(abbr)
            with cols[i % 3]:
                st.markdown(f"""<div class="glow-card" style="box-shadow: 0 0 24px {color}55;">
  <div class="glow-hdr">
    <img src="{logo or ''}" style="width:42px;height:42px;border-radius:8px;border:1px solid #222;background:#111" />
    <div>
      <div class="glow-name">{f['full_name']}</div>
      <div class="glow-meta">{abbr}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)
                rcol = st.columns([8,1])[1]
                if rcol.button("❌", key=f"del_{f['id']}", help="Remove from favorites"):
                    favs = [x for x in favs if x["id"] != f["id"]]
                    _save_favorites(favs)
                    _rerun()
    else:
        st.info("No favorites yet.")

    st.divider()
    st.subheader("Bulk Predict (auto opponent)")
    fast_mode = st.toggle("Fast mode (no training)", value=False, key="fav_fast")
    run_all = st.button("Run predictions for all favorites")

    if not run_all or not favs: return

    season = "2025-26"
    all_rows: List[pd.DataFrame] = []
    share_images: List[Tuple[str, bytes]] = []
    dp = load_team_defense_pace(season)
    progress = st.progress(0, text="Predicting for favorites…")

    for i, fav in enumerate(favs):
        pid, pname = fav["id"], fav["full_name"]
        team_id = int(fav.get("team_id", 0) or 0)
        abbr = fav.get("team_abbr") or ""
        nba_pid = fav.get("nba_person_id")

        logs = load_logs(pid, season)
        if logs.empty:
            year = dt.date.today().year
            fallback = f"{year-1}-{str(year)[-2:]}"
            logs = load_logs(pid, fallback)
            season = fallback
        if logs.empty:
            progress.progress((i+1)/len(favs)); continue

        feats = build_all_features(logs, season)
        next_game = auto_next_opponent(team_id, season) if team_id > 0 else None

        upcoming_ctx = {}
        next_info = "Next: N/A"
        if next_game:
            opp_row = dp[dp["abbreviation"] == next_game["opp_abbr"]] if not dp.empty else pd.DataFrame()
            if not opp_row.empty:
                upcoming_ctx["OPP_DEF_PPG"] = float(opp_row["OPP_DEF_PPG"].iloc[0])
                upcoming_ctx["OPP_DEF_Z"]   = float(opp_row["OPP_DEF_Z"].iloc[0])
                upcoming_ctx["OPP_PACE"]    = float(opp_row["PACE"].iloc[0])
                upcoming_ctx["OPP_PACE_Z"]  = float(opp_row["PACE_Z"].iloc[0])
                upcoming_ctx["OPP_DEF_X_PACE"] = upcoming_ctx["OPP_DEF_Z"] * upcoming_ctx["OPP_PACE_Z"]
            if "GAME_DATE" in feats.columns and feats["GAME_DATE"].notna().any():
                last_date = pd.to_datetime(feats["GAME_DATE"]).max().date()
                rest_days = max(0, (next_game["date"] - last_date).days)
            else:
                rest_days = 2
            upcoming_ctx["IS_HOME"] = 1 if next_game["is_home"] else 0
            upcoming_ctx["REST_DAYS"] = rest_days
            upcoming_ctx["BACK_TO_BACK"] = 1 if rest_days == 1 else 0
            next_info = f"Next: {('Home' if next_game['is_home'] else 'Away')} vs {next_game['opp_abbr']} on {next_game['date']} · Rest {rest_days}d"

        futures, res = {}, []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for stat in PROP_MAP.keys():
                futures[ex.submit(train_predict_for_stat, pid, season, stat, feats, fast_mode, upcoming_ctx)] = stat
            for fut in as_completed(futures): res.append(fut.result())
        df_table = results_to_table(res)
        df_table.insert(0, "Player", pname); df_table.insert(1, "Season", season)
        all_rows.append(df_table)

        # per-player bar chart
        team_color = TEAM_META.get(abbr, {}).get("color", "#60a5fa")
        c = alt.Chart(df_table.rename(columns={"Pred":"Prediction"})).mark_bar(color=team_color).encode(
            x=alt.X("Stat:N", sort=df_table["Stat"].tolist()),
            y=alt.Y("Prediction:Q"),
            tooltip=["Stat","Prediction","Model","MAE","MSE"],
        ).properties(width="container", height=220, title=f"{pname} — Predictions")
        st.altair_chart(c, use_container_width=True)

        # share-card per player
        photo = get_player_photo_bytes(pid, nba_pid)
        img_bytes = make_share_image(pname, season, photo, df_table[["Stat","Pred","Model","MAE","MSE"]], next_info)
        share_images.append((f"{pname.replace(' ','_')}_{season}.png", img_bytes))

        progress.progress((i+1)/len(favs))

    progress.empty()
    if all_rows:
        df_all = pd.concat(all_rows, ignore_index=True)
        st.subheader("Bulk predictions — Table")
        st.dataframe(df_all, use_container_width=True, hide_index=True)
        table_downloaders(df_all, filename_prefix=f"favorites_{season}_predictions")

    if share_images:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for fname, content in share_images: z.writestr(fname, content)
        st.download_button("📦 Download share images (ZIP)", data=buf.getvalue(), file_name=f"favorites_{season}_share_images.zip", mime="application/zip")


def page_saved():
    st.header("Saved Predictions")
    rows = _read_jsonl(PRED_FILE)
    if not rows: st.info("No saved runs yet."); return
    df = pd.DataFrame(rows); df["ts"] = pd.to_datetime(df["ts"])
    st.dataframe(df[["id","ts","player_id","player_name","season","fast_mode","next_info"]].sort_values("ts", ascending=False), use_container_width=True)
    del_id = st.text_input("Delete by run id", value="")
    if st.button("🗑️ Delete run"):
        rid = del_id.strip()
        new_rows = [r for r in rows if r["id"] != rid]
        if len(new_rows) == len(rows): st.warning("Run id not found.")
        else: _write_jsonl(PRED_FILE, new_rows); st.success("Deleted.")


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
        st.markdown('<span class="tag">Auto Opp</span> <span class="tag">Defense</span> <span class="tag">Pace</span> <span class="tag">CSV</span> <span class="tag">Share</span>', unsafe_allow_html=True)

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
