# app.py
"""
NBA Prop Predictor — Elite (Predict & Favorites only)
- Animated basketball loaders; polished theme & card animations
- Team-colored metric cards with prop label, predicted value, model, and confidence %
- Robust player loading; Auto opponent; DEF_Z/ PACE_Z/ DEF×PACE
- Favorites with glow cards and ❌
- Model budget control (Full / Lite 3 models / Single)
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

# Local modules expected in your repo
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
# STYLING + ANIMATIONS
# =============================================================================

def inject_css() -> None:
    st.markdown(
        """
<style>
html, body { font-family: Inter, ui-sans-serif, system-ui; }
.block-container { padding-top: 1.1rem; max-width: 1240px; }
h1, h2, h3, h4 {
  background: linear-gradient(90deg,#e2e8f0 0%, #60a5fa 40%, #34d399 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  animation: fadeIn 700ms ease both;
}

/* Global animations */
@keyframes fadeIn { from {opacity:0; transform: translateY(6px)} to {opacity:1; transform:none} }
@keyframes rise { from {opacity:.0; transform: translateY(12px)} to {opacity:1; transform:none} }
@keyframes pulseGlow { 0%,100%{box-shadow:0 0 0 rgba(255,255,255,0)} 50%{box-shadow:0 0 28px rgba(255,255,255,.08)} }

/* Cards */
.card {
  background: rgba(17,24,39,.6);
  border: 1px solid rgba(255,255,255,.08);
  border-radius: 16px; padding: 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,.25);
  backdrop-filter: blur(6px);
  animation: rise 540ms ease both;
}

/* Metric grid */
.metric-grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 14px; }
.metric-card {
  border-radius: 16px; padding: 16px 14px; color: #fff; border:1px solid rgba(255,255,255,.10);
  transform-origin: center; animation: rise 480ms ease both;
}
.metric-title { font-size:.9rem; opacity:.9; margin-bottom:.35rem; letter-spacing:.25px; }
.metric-value { font-size:1.9rem; font-weight:800; line-height:1.1; }
.metric-sub { font-size:.82rem; opacity:.9; margin-top:.2rem }
.metric-chip { display:inline-block; font-size:.72rem; padding:.2rem .5rem; border-radius: 999px; background: rgba(255,255,255,.12); margin-left:.35rem; }

/* Favorites grid */
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 16px; }
.glow-card { padding: 14px; border-radius: 16px; border: 1px solid rgba(255,255,255,.08); background: #0b1220; animation: rise 420ms ease both; }
.glow-hdr { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
.glow-name { font-weight: 700; color: #e5e7eb; }
.glow-meta { font-size: .9rem; color: #cbd5e1; }

/* Buttons */
.stButton>button { border-radius: 12px; padding: 10px 14px; font-weight: 600;
  background: linear-gradient(90deg,#0ea5e9,#22c55e); border: none; }
.stButton>button:hover { filter: brightness(1.05); }

/* Tag/badge */
.tag { display:inline-block; padding:.2rem .55rem; border-radius:999px; font-size:.75rem;
  background:rgba(99,102,241,.15); border:1px solid rgba(99,102,241,.35); }
.badge { display:inline-block; padding:.25rem .6rem; border-radius:10px; font-size:.8rem;
  background:rgba(34,197,94,.15); border:1px solid rgba(34,197,94,.4); color:#d1fae5; }

/* DataFrame wrap */
[data-testid="stDataFrame"] { border-radius: 12px; border: 1px solid rgba(255,255,255,0.08); }

/* Basketball Loader */
.loader-wrap { padding: 10px 0 18px 0; }
.court {
  width: 100%; height: 14px; border-radius: 999px; position: relative;
  background: linear-gradient(90deg, rgba(255,255,255,.08), rgba(255,255,255,.16));
  overflow: hidden; border: 1px solid rgba(255,255,255,.12);
}
.ball {
  width: 22px; height: 22px; border-radius: 50%;
  background: radial-gradient(circle at 30% 30%, #ffb86b, #d97706 55%, #92400e 100%);
  position: absolute; top: -4px; left: -22px;
  box-shadow: inset 0 0 0 2px rgba(0,0,0,.12), 0 6px 14px rgba(0,0,0,.35);
  animation: dribble 1.6s linear infinite;
}
@keyframes dribble {
  0%   { transform: translateX(0) translateY(0); }
  10%  { transform: translateX(10%) translateY(2px); }
  20%  { transform: translateX(20%) translateY(0); }
  30%  { transform: translateX(30%) translateY(2px); }
  40%  { transform: translateX(40%) translateY(0); }
  50%  { transform: translateX(50%) translateY(2px); }
  60%  { transform: translateX(60%) translateY(0); }
  70%  { transform: translateX(70%) translateY(2px); }
  80%  { transform: translateX(80%) translateY(0); }
  90%  { transform: translateX(90%) translateY(2px); }
  100% { transform: translateX(110%) translateY(0); }
}
.loader-text { color:#cbd5e1; font-size:.9rem; margin-bottom:.35rem; }
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

def safe_int(v, default=0) -> int:
    # Avoids NAType.__bool__
    try:
        if v is None: return default
        if isinstance(v, str) and v.strip() == "": return default
        if pd.isna(v): return default
        return int(v)
    except Exception:
        return default

def safe_str(v, default="") -> str:
    try:
        if v is None: return default
        if pd.isna(v): return default
        s = str(v)
        return s if s.strip() != "" else default
    except Exception:
        return default

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
# PERSISTENCE (favorites)
# =============================================================================

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

    id_done = False
    for c in ["id","player_id","PLAYER_ID","PersonId","PERSON_ID"]:
        if c in p.columns:
            if c != "id": p = p.rename(columns={c: "id"})
            id_done = True; break
    if not id_done:
        p["id"] = pd.NA

    if "full_name" not in p.columns:
        if {"first_name","last_name"}.issubset(p.columns):
            p["full_name"] = (p["first_name"].astype(str).str.strip() + " " + p["last_name"].astype(str).str.strip()).str.strip()
        elif "PLAYER" in p.columns:
            p["full_name"] = p["PLAYER"].astype(str)
        elif "DISPLAY_FIRST_LAST" in p.columns:
            p["full_name"] = p["DISPLAY_FIRST_LAST"].astype(str)
        else:
            p["full_name"] = p.get("full_name","Unknown")

    p = _safe_get_team_cols(p)
    if "team_abbr" not in p.columns or p["team_abbr"].isna().all():
        teams = load_teams_bdl()
        if "team_id" in p.columns and not p["team_id"].isna().all() and not teams.empty:
            p = p.merge(
                teams.rename(columns={"id":"team_id","abbreviation":"team_abbr"}),
                on="team_id", how="left"
            )

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
    out = out[out["full_name"].astype(str).str.strip().ne("")]
    return out

@st.cache_data(show_spinner=False)
def bdl_fetch_active_players_direct() -> pd.DataFrame:
    out = []
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (PropPredictor)"})
    url = "https://www.balldontlie.io/api/v1/players"
    page = 1
    while True:
        try:
            r = s.get(url, params={"active": "true", "per_page": 100, "page": page}, timeout=8)
            r.raise_for_status()
            j = r.json()
            data = j.get("data", [])
            if not data: break
            out.extend(data)
            meta = j.get("meta", {})
            if not meta or not meta.get("next_page"): break
            page = meta["next_page"]
        except Exception:
            break
    return pd.DataFrame(out)

@st.cache_data(show_spinner=False)
def load_player_list(season: str = "2025-26") -> pd.DataFrame:
    try:
        raw = dfetch.get_active_players_balldontlie()
        p = normalize_players_df(raw)
        if not p.empty:
            return p[["id","full_name","team_id","team_abbr","nba_person_id"]]
    except Exception:
        pass
    try:
        raw2 = bdl_fetch_active_players_direct()
        p2 = normalize_players_df(raw2)
        if not p2.empty:
            return p2[["id","full_name","team_id","team_abbr","nba_person_id"]]
    except Exception:
        pass
    try:
        fb = dfetch.get_player_list_nba()
        p3 = normalize_players_df(_filter_active_players(fb, season))
        if not p3.empty:
            return p3[["id","full_name","team_id","team_abbr","nba_person_id"]]
    except Exception:
        pass
    favs = _load_favorites()
    if favs:
        p4 = pd.DataFrame(favs)
        for col in ["team_id","team_abbr","nba_person_id"]:
            if col not in p4.columns: p4[col] = None
        p4["team_id"] = p4["team_id"].apply(lambda x: safe_int(x, 0))
        p4["team_abbr"] = p4["team_abbr"].apply(lambda x: safe_str(x, ""))
        p4["nba_person_id"] = p4["nba_person_id"].apply(lambda x: None if x in (None, "", 0) else int(x))
        return p4[["id","full_name","team_id","team_abbr","nba_person_id"]].drop_duplicates(subset=["id"]).sort_values("full_name").reset_index(drop=True)
    return pd.DataFrame(columns=["id","full_name","team_id","team_abbr","nba_person_id"])

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
# OPPONENT DEFENSE & PACE (Balldontlie)
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
    if not team_id or int(team_id) <= 0: return None
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
        out[f"{c}_L1"]  = np.roll(s, 1)
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
# CONFIDENCE & MODELING
# =============================================================================

def _confidence_from_error(mae: float, mse: float, y_scale: float) -> float:
    # Why: convert errors to an intuitive 0–100 score, higher = better trust
    if not np.isfinite(y_scale) or y_scale <= 0: y_scale = 1.0
    rmse = np.sqrt(mse) if np.isfinite(mse) else (mae if np.isfinite(mae) else np.nan)
    if not np.isfinite(mae) and not np.isfinite(rmse):
        return 60.0
    mae = 0.0 if not np.isfinite(mae) else mae
    rmse = 0.0 if not np.isfinite(rmse) else rmse
    norm = 0.5 * (mae / y_scale) + 0.5 * (rmse / y_scale)
    conf = float(np.clip(np.exp(-norm), 0.35, 0.95) * 100.0)
    return round(conf, 1)

def _apply_model_budget(manager: ModelManager, budget: str) -> None:
    if budget == "Full ensemble": 
        return
    if budget == "Lite (3 models)":
        wanted = {"lasso","ridge","random_forest","rf"}
    else:  # "Single (Lasso)"
        wanted = {"lasso"}
    try:
        if hasattr(manager, "set_model_whitelist"):
            manager.set_model_whitelist(list(wanted)); return
        if hasattr(manager, "available_models") and isinstance(manager.available_models, list):
            manager.available_models = [m for m in manager.available_models
                                        if str(getattr(m, "name", m)).lower() in wanted or
                                           str(getattr(m, "key", m)).lower() in wanted]
        elif hasattr(manager, "models") and isinstance(manager.models, list):
            manager.models = [m for m in manager.models
                              if str(getattr(m, "name", m)).lower() in wanted or
                                 str(getattr(m, "key", m)).lower() in wanted]
    except Exception:
        pass

def get_or_train_model_cached(player_id: int, season: str, stat: str, X: pd.DataFrame, y: np.ndarray, budget: str) -> ModelManager:
    key = _hash_frame_small(X, y, player_id, season, stat) + f"|{budget}"
    ss: Dict[str, ModelManager] = st.session_state.setdefault("model_cache", {})
    if key in ss: return ss[key]
    manager = ModelManager(random_state=42)
    _apply_model_budget(manager, budget)
    manager.train(X, y)
    ss[key] = manager
    return manager

def train_predict_for_stat(
    player_id: int,
    season: str,
    stat: str,
    features: pd.DataFrame,
    fast_mode: bool,
    model_budget: str,
    upcoming_ctx: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    y_all = build_target(features, stat).to_numpy()
    X_all = select_X_for_stat(features, stat)
    df_join = pd.concat([pd.Series(y_all, name="TARGET", index=X_all.index), X_all], axis=1)
    df_join = df_join.loc[~df_join["TARGET"].isna()].copy()
    if df_join.empty:
        return {"Stat": stat, "Prediction": float("nan"), "Best Model": "NoData", "MAE": float("nan"), "MSE": float("nan"), "Confidence": 50.0, "Scale": 1.0}

    y_final = df_join["TARGET"].to_numpy(dtype=float)
    X_final = _impute_features(df_join.drop(columns=["TARGET"]))
    if len(X_final) > N_TRAIN:
        X_final = X_final.iloc[-N_TRAIN:].copy()
        y_final = y_final[-N_TRAIN:].copy()

    y_scale = float(np.nanstd(y_final) or 1.0)

    X_next = X_final.tail(1).copy()
    if upcoming_ctx:
        for k, v in upcoming_ctx.items():
            if k in X_next.columns:
                X_next.loc[:, k] = v

    if fast_mode or len(X_final) < MIN_ROWS_FOR_MODEL:
        pred = float(np.nanmean(y_final[-10:])) if np.isfinite(y_final[-10:]).any() else float("nan")
        # baseline error vs local mean
        w = y_final[-10:] if len(y_final) >= 5 else y_final
        if w.size:
            m = float(np.nanmean(w))
            mae_b = float(np.nanmean(np.abs(w - m)))
            mse_b = float(np.nanmean((w - m) ** 2))
        else:
            mae_b = mse_b = np.nan
        conf = _confidence_from_error(mae_b, mse_b, y_scale)
        return {"Stat": stat, "Prediction": pred, "Best Model": "Baseline", "MAE": mae_b, "MSE": mse_b, "Confidence": conf, "Scale": y_scale}

    try:
        manager = get_or_train_model_cached(player_id, season, stat, X_final, y_final, model_budget)
        _ = manager.predict(X_next)
        best = manager.best_model()
        conf = _confidence_from_error(float(best.mae), float(best.mse), y_scale)
        return {"Stat": stat, "Prediction": float(best.prediction), "Best Model": best.name, "MAE": float(best.mae), "MSE": float(best.mse), "Confidence": conf, "Scale": y_scale}
    except Exception:
        pred = float(np.nanmean(y_final[-10:])) if np.isfinite(y_final[-10:]).any() else float("nan")
        conf = _confidence_from_error(np.nan, np.nan, y_scale)
        return {"Stat": stat, "Prediction": pred, "Best Model": "Baseline(Fallback)", "MAE": float("nan"), "MSE": float("nan"), "Confidence": conf, "Scale": y_scale}


# =============================================================================
# TABLE/CSV + CHARTS + SHARE
# =============================================================================

def results_to_table(results: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    order = {k: i for i, k in enumerate(PROP_MAP.keys())}
    df["__order"] = df["Stat"].map(order)
    df = df.sort_values("__order").drop(columns="__order")
    df["Pred"] = pd.to_numeric(df["Prediction"], errors="coerce").round(2)
    df["MAE"]  = pd.to_numeric(df["MAE"], errors="coerce").round(2)
    df["MSE"]  = pd.to_numeric(df["MSE"], errors="coerce").round(2)
    df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce").round(1)
    return df[["Stat","Pred","Best Model","MAE","MSE","Confidence"]].rename(columns={"Best Model":"Model"})

def table_downloaders(df: pd.DataFrame, filename_prefix: str) -> None:
    csv = df.to_csv(index=False)
    st.download_button("⬇️ Download CSV", data=csv.encode("utf-8"), file_name=f"{filename_prefix}.csv", mime="text/csv")
    with st.expander("Copy CSV text"):
        st.text_area("CSV", value=csv, height=160)

def lighten_hex(hex_color: str, amount: float = 0.35) -> str:
    hex_color = hex_color.lstrip("#")
    try:
        r = int(hex_color[0:2], 16); g = int(hex_color[2:4], 16); b = int(hex_color[4:6], 16)
    except Exception:
        r, g, b = (96, 165, 250)
    r = int(r + (255 - r) * amount); g = int(g + (255 - g) * amount); b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"

def render_metric_cards(results: List[Dict], team_color: str) -> None:
    base = team_color or "#60a5fa"
    soft = lighten_hex(base, 0.55)
    html = ['<div class="metric-grid">']
    for i, r in enumerate(results):
        stat = str(r["Stat"])
        pred = f'{float(r["Prediction"]):.2f}' if np.isfinite(r["Prediction"]) else "—"
        model = safe_str(r.get("Best Model"), "—")
        conf = r.get("Confidence", 60.0)
        # Why inline styles: dynamic team color & staggered animation delay
        html.append(f"""
<div class="metric-card" style="background: linear-gradient(135deg, {base} 0%, {soft} 95%); animation-delay:{i*0.03:.2f}s">
  <div class="metric-title">{stat} <span class="metric-chip">{model}</span></div>
  <div class="metric-value">{pred}</div>
  <div class="metric-sub">Confidence: {conf:.1f}%</div>
</div>""")
    html.append("</div>")
    st.markdown("\n".join(html), unsafe_allow_html=True)

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
        tooltip=["Stat","Pred","Model","MAE","MSE","Confidence"],
    ).properties(height=280, title=title)
    st.altair_chart(c, use_container_width=True)

def show_basketball_loader(placeholder: st.delta_generator.DeltaGenerator, text: str) -> None:
    placeholder.markdown(f"""
<div class="loader-wrap">
  <div class="loader-text">{text}</div>
  <div class="court"><div class="ball"></div></div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# SAFE PLAYER PICKER
# =============================================================================

def pick_player_row(players: pd.DataFrame, selected_name: str) -> Optional[pd.Series]:
    if players is None or players.empty: return None
    names = players["full_name"].astype(str).values
    idx = np.where(names == str(selected_name))[0]
    if idx.size:
        return players.iloc[int(idx[0])]
    return players.iloc[0]


# =============================================================================
# PAGES
# =============================================================================

def page_predict(players: pd.DataFrame):
    st.header("NBA Prop Predictor — Elite")
    st.caption("Auto opponent • Defense + Pace • Animated loaders • Team-colored cards • Share image")

    if players is None or players.empty:
        st.error("No active players available. Click **Refresh players** in the sidebar and try again.")
        return

    names_list = players["full_name"].astype(str).tolist()
    col_left, col_right = st.columns([1, 3])
    with col_left:
        name = st.selectbox("Select Player", names_list, key="predict_player")
        row = pick_player_row(players, name)
        if row is None:
            st.error("Could not resolve the selected player."); 
            return
        player_id = safe_int(row.get("id"), 0)
        team_id = safe_int(row.get("team_id"), 0)
        team_abbr = safe_str(row.get("team_abbr"), "")
        team_color = TEAM_META.get(team_abbr, {}).get("color", "#60a5fa")
        nba_pid = None
        if "nba_person_id" in row and not pd.isna(row["nba_person_id"]):
            nba_pid = safe_int(row["nba_person_id"], None)
        fast_mode = st.toggle("Fast mode (no training)", value=False, key="fast_toggle")
        model_budget = st.radio("Model budget", ["Full ensemble", "Lite (3 models)", "Single (Lasso)"], index=0, help="Fewer models = faster", key="model_budget")
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

    load_ph = st.empty()
    show_basketball_loader(load_ph, "Building features…")
    features = build_all_features(logs, season)
    load_ph.empty()

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

    load2 = st.empty()
    show_basketball_loader(load2, "Training models & predicting…")
    futures, results = {}, []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for stat in PROP_MAP.keys():
            futures[ex.submit(train_predict_for_stat, player_id, season, stat, features, fast_mode, model_budget, upcoming_ctx)] = stat
        for fut in as_completed(futures): results.append(fut.result())
    load2.empty()

    # --- Metric cards (replace table) ---
    st.subheader("Predicted Props")
    render_metric_cards(results, TEAM_META.get(team_abbr, {}).get("color", "#60a5fa"))

    # CSV + bar chart still available
    df_table = results_to_table(results)
    table_downloaders(df_table, filename_prefix=f"{name.replace(' ','_')}_{season}_predictions")
    bar_chart_from_table(df_table, title="Predictions (bars)", color=TEAM_META.get(team_abbr, {}).get("color", "#60a5fa"))

    img_bytes = make_share_image(name, season, photo, df_table[["Stat","Pred","Model","MAE","MSE"]], next_info)
    st.download_button("📸 Share image (PNG)", data=img_bytes, file_name=f"{name.replace(' ','_')}_{season}_predictions.png", mime="image/png")

    st.divider()
    st.subheader("Recent Games")
    cols_show = [c for c in ["GAME_DATE","MATCHUP","PTS","REB","AST","FG3M","STL","BLK","TOV","MIN"] if c in logs.columns]
    st.dataframe(logs[cols_show], use_container_width=True)


def page_favorites(players: pd.DataFrame):
    st.header("Favorites (Auto Opp • Defense & Pace • Glow Cards)")

    if players is None or players.empty:
        st.error("No active players available to add to favorites.")
        return

    favs = _load_favorites()

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
                pid = safe_int(row.get("id"), 0)
                nba_pid = None
                if "nba_person_id" in row and not pd.isna(row["nba_person_id"]):
                    nba_pid = safe_int(row["nba_person_id"], None)
                team_id = safe_int(row.get("team_id"), 0)
                team_abbr = safe_str(row.get("team_abbr"), "")
                if not any(f["id"] == pid for f in favs):
                    favs.append({"id": pid, "full_name": name, "team_id": team_id, "team_abbr": team_abbr, "nba_person_id": nba_pid})
                    _save_favorites(favs); st.success("Added to favorites.")
                else:
                    st.info("Already in favorites.")

    if favs:
        st.subheader("Saved favorites")
        cols = st.columns(3)
        for i, f in enumerate(list(favs)):
            abbr = safe_str(f.get("team_abbr"), "")
            color = TEAM_META.get(abbr, {}).get("color", "#60a5fa")
            logo = nba_logo_url(abbr)
            with cols[i % 3]:
                st.markdown(f"""<div class="glow-card" style="box-shadow: 0 0 24px {color}55;">
  <div class="glow-hdr">
    <img src="{logo or ''}" style="width:42px;height:42px;border-radius:8px;border:1px solid #222;background:#111" />
    <div>
      <div class="glow-name">{safe_str(f.get('full_name'),'')}</div>
      <div class="glow-meta">{abbr}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)
                rcol = st.columns([8,1])[1]
                if rcol.button("❌", key=f"del_{safe_int(f.get('id'),0)}", help="Remove from favorites"):
                    favs = [x for x in favs if safe_int(x.get("id"),0) != safe_int(f.get("id"),0)]
                    _save_favorites(favs)
                    _rerun()
    else:
        st.info("No favorites yet.")

    st.divider()
    st.subheader("Bulk Predict (auto opponent)")
    fast_mode = st.toggle("Fast mode (no training)", value=False, key="fav_fast")
    model_budget = st.radio("Model budget (bulk)", ["Full ensemble", "Lite (3 models)", "Single (Lasso)"], index=1, horizontal=True, key="fav_budget")
    run_all = st.button("Run predictions for all favorites")

    if not run_all or not favs: return

    season = "2025-26"
    all_rows: List[pd.DataFrame] = []
    share_images: List[Tuple[str, bytes]] = []
    dp = load_team_defense_pace(season)
    progress = st.progress(0, text="Predicting for favorites…")

    for i, fav in enumerate(favs):
        pid, pname = safe_int(fav.get("id"), 0), safe_str(fav.get("full_name"), "")
        team_id = safe_int(fav.get("team_id"), 0)
        abbr = safe_str(fav.get("team_abbr"), "")
        nba_pid = fav.get("nba_person_id")
        nba_pid = None if nba_pid in (None, "", 0) else safe_int(nba_pid, None)

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
                futures[ex.submit(train_predict_for_stat, pid, season, stat, feats, fast_mode, model_budget, upcoming_ctx)] = stat
            for fut in as_completed(futures): res.append(fut.result())
        df_table = results_to_table(res)
        df_table.insert(0, "Player", pname); df_table.insert(1, "Season", season)
        all_rows.append(df_table)

        team_color = TEAM_META.get(abbr, {}).get("color", "#60a5fa")
        c = alt.Chart(df_table.rename(columns={"Pred":"Prediction"})).mark_bar(color=team_color).encode(
            x=alt.X("Stat:N", sort=df_table["Stat"].tolist()),
            y=alt.Y("Prediction:Q"),
            tooltip=["Stat","Prediction","Model","MAE","MSE","Confidence"],
        ).properties(width="container", height=220, title=f"{pname} — Predictions")
        st.altair_chart(c, use_container_width=True)

        photo = get_player_photo_bytes(pid, nba_pid)
        share_bytes = make_share_image(pname, season, photo, df_table[["Stat","Pred","Model","MAE","MSE"]], next_info)
        share_images.append((f"{pname.replace(' ','_')}_{season}.png", share_bytes))

        progress.progress((i+1)/len(favs))

    progress.empty()
    if all_rows:
        df_all = pd.concat(all_rows, ignore_index=True)
        st.subheader("Bulk predictions — Table")
        st.dataframe(df_all, use_container_width=True, hide_index=True)
        csv = df_all.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download all (CSV)", data=csv, file_name=f"favorites_{season}_predictions.csv", mime="text/csv")

    if share_images:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for fname, content in share_images: z.writestr(fname, content)
        st.download_button("📦 Download share images (ZIP)", data=buf.getvalue(), file_name=f"favorites_{season}_share_images.zip", mime="application/zip")


# =============================================================================
# APP
# =============================================================================

def main():
    st.set_page_config(page_title="NBA Prop Predictor — Elite", page_icon="🏀", layout="wide")
    inject_css()

    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio("Go to", ["Predict", "Favorites"], label_visibility="collapsed")
        st.markdown("---")
        if st.button("🔄 Refresh players"):
            load_player_list.clear()
            _rerun()
        st.markdown('<span class="tag">Auto Opp</span> <span class="tag">Defense</span> <span class="tag">Pace</span> <span class="tag">Cards</span> <span class="tag">Share</span>', unsafe_allow_html=True)

    players = load_player_list("2025-26")

    if page == "Predict":
        page_predict(players)
    else:
        page_favorites(players)


if __name__ == "__main__":
    main()
