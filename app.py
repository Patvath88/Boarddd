# app.py
"""
Hot Shot Props AI — Streamlit
Layout identical to reference: Favorites | Prop Projection Lab | Projection Tracker
- Player card with photo + team logo
- Orange stat tiles (fixed order); Add to Favorites; Track Projection
- Auto opponent + DEF_Z, PACE_Z, DEF×PACE features
- Team-colored bar chart; share image; CSV; robust loading
"""

from __future__ import annotations

import os
import io
import json
import zipfile
import hashlib
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

# local modules
import data_fetching as dfetch
from models import ModelManager


# =========================
# CONSTANTS / CONFIG
# =========================

PROP_MAP = {
    "Points": "PTS",
    "Assists": "AST",
    "Rebounds": "REB",
    "Minutes": "MIN",
    "Steals": "STL",
    "Blocks": "BLK",
    "TOV": "TOV",
    "3PM": "FG3M",
    "PRA": ["PTS", "REB", "AST"],
    "PR": ["PTS", "REB"],
    "PA": ["PTS", "AST"],
    "RA": ["REB", "AST"],
}
STAT_COLUMNS = ["PTS","REB","AST","STL","BLK","TOV","FG3M","MIN"]

# Fixed card order exactly like screenshot
CARD_ORDER_TOP = ["Points","Assists","Rebounds","Minutes","Steals","Blocks"]
CARD_ORDER_BOTTOM = ["PA","PR","RA","PRA","TOV"]
BOX_SCORE_ORDER = CARD_ORDER_TOP + ["3PM"] + CARD_ORDER_BOTTOM  # used for charts/tables too

BASE_COLS = [
    "IS_HOME","REST_DAYS","BACK_TO_BACK",
    "OPP_ALLOW_PTS","OPP_ALLOW_REB","OPP_ALLOW_AST",
    "OPP_DEF_PPG","OPP_DEF_Z","OPP_PACE","OPP_PACE_Z","OPP_DEF_X_PACE",
]

N_TRAIN = 60
MIN_ROWS_FOR_MODEL = 12
MAX_WORKERS = max(2, min(8, os.cpu_count() or 4))

DATA_DIR = Path("./data"); DATA_DIR.mkdir(parents=True, exist_ok=True)
FAV_FILE = DATA_DIR / "favorites.json"


TEAM_META: Dict[str, Dict[str, str | int]] = {
    "ATL":{"color":"#E03A3E","nba_id":1610612737, "name":"Hawks"},
    "BOS":{"color":"#007A33","nba_id":1610612738, "name":"Celtics"},
    "BKN":{"color":"#000000","nba_id":1610612751, "name":"Nets"},
    "CHA":{"color":"#1D1160","nba_id":1610612766, "name":"Hornets"},
    "CHI":{"color":"#CE1141","nba_id":1610612741, "name":"Bulls"},
    "CLE":{"color":"#860038","nba_id":1610612739, "name":"Cavaliers"},
    "DAL":{"color":"#00538C","nba_id":1610612742, "name":"Mavericks"},
    "DEN":{"color":"#0E2240","nba_id":1610612743, "name":"Nuggets"},
    "DET":{"color":"#C8102E","nba_id":1610612765, "name":"Pistons"},
    "GSW":{"color":"#1D428A","nba_id":1610612744, "name":"Warriors"},
    "HOU":{"color":"#CE1141","nba_id":1610612745, "name":"Rockets"},
    "IND":{"color":"#002D62","nba_id":1610612754, "name":"Pacers"},
    "LAC":{"color":"#C8102E","nba_id":1610612746, "name":"Clippers"},
    "LAL":{"color":"#552583","nba_id":1610612747, "name":"Lakers"},
    "MEM":{"color":"#5D76A9","nba_id":1610612763, "name":"Grizzlies"},
    "MIA":{"color":"#98002E","nba_id":1610612748, "name":"Heat"},
    "MIL":{"color":"#00471B","nba_id":1610612749, "name":"Bucks"},
    "MIN":{"color":"#0C2340","nba_id":1610612750, "name":"Timberwolves"},
    "NOP":{"color":"#0C2340","nba_id":1610612740, "name":"Pelicans"},
    "NYK":{"color":"#F58426","nba_id":1610612752, "name":"Knicks"},
    "OKC":{"color":"#007AC1","nba_id":1610612760, "name":"Thunder"},
    "ORL":{"color":"#0077C0","nba_id":1610612753, "name":"Magic"},
    "PHI":{"color":"#006BB6","nba_id":1610612755, "name":"76ers"},
    "PHX":{"color":"#1D1160","nba_id":1610612756, "name":"Suns"},
    "POR":{"color":"#E03A3E","nba_id":1610612757, "name":"Trail Blazers"},
    "SAC":{"color":"#5A2D81","nba_id":1610612758, "name":"Kings"},
    "SAS":{"color":"#C4CED4","nba_id":1610612759, "name":"Spurs"},
    "TOR":{"color":"#CE1141","nba_id":1610612761, "name":"Raptors"},
    "UTA":{"color":"#002B5C","nba_id":1610612762, "name":"Jazz"},
    "WAS":{"color":"#002B5C","nba_id":1610612764, "name":"Wizards"},
}
def nba_logo_url(team_abbr: str) -> Optional[str]:
    meta = TEAM_META.get(team_abbr or "")
    if not meta: return None
    return f"https://cdn.nba.com/logos/nba/{meta['nba_id']}/global/L/logo.png"


# =========================
# THEME / CSS
# =========================

def inject_css_hotshot():
    st.markdown("""
<style>
:root{
  --bg:#0b0f14; --panel:#121821; --tile:#2a3038; --tileText:#e5e7eb;
  --accent:#e85d12; --accentDark:#c04a0e; --soft:#9aa3ad;
}
html,body{background:var(--bg)}
.block-container{max-width:1220px;padding-top:0.6rem}
header{visibility:hidden}
h1.app-title{
  display:flex; align-items:center; gap:.6rem; color:#fff; font-weight:800; letter-spacing:.2px;
}
.app-title .logo{
  width:28px;height:28px;border-radius:50%;background:var(--accent);
  display:inline-flex;align-items:center;justify-content:center;color:#000;
}
.topbar{display:flex;align-items:center;justify-content:space-between;margin:4px 0 12px}
.nav{
  display:flex;gap:12px;border-bottom:1px solid #222;padding-bottom:8px;margin-bottom:12px
}
.nav .tab{padding:8px 14px;border-radius:10px;background:var(--panel);color:#eee;border:1px solid #1f2630;cursor:pointer;font-weight:700}
.nav .tab.active{background:var(--accent);color:#fff;border-color:var(--accentDark)}
.player-card{display:grid;grid-template-columns: 280px 1fr; gap:24px; align-items:center; background:var(--panel); border:1px solid #1e252f; border-radius:14px; padding:16px; margin-top:6px}
.player-photo{border-radius:12px;border:1px solid #1e252f;width:240px;height:240px;object-fit:cover;background:#0e141d}
.team-logo{width:180px;height:180px;object-fit:contain;filter:drop-shadow(0 4px 18px rgba(0,0,0,.5))}
.player-name{font-weight:800;color:#fff;font-size:1.6rem;margin-bottom:2px}
.player-team{color:#cbd5e1}
.tiles{display:grid;grid-template-columns: repeat(6, 1fr);gap:12px;margin-top:10px}
.tile{background:var(--tile);border-radius:12px;border:1px solid #1f2630;overflow:hidden}
.tile-head{background:var(--accent);color:#fff;font-weight:800;padding:8px 10px}
.tile-body{color:var(--tileText);font-weight:800;font-size:1.6rem; padding:10px 12px}
.tiles.bottom{grid-template-columns: repeat(5, 1fr)}
.actions{display:flex;gap:12px;margin-top:12px}
.btn{
  background:var(--panel); color:#eee; border:1px solid #2a323f; padding:10px 14px; border-radius:12px; font-weight:700; cursor:pointer;
}
.btn.primary{background:var(--accent); border-color:var(--accentDark); color:#fff}
.btn:hover{filter:brightness(1.05)}
.glow-card{padding:14px;border-radius:14px;border:1px solid #1f2630;background:var(--panel);box-shadow:0 0 22px rgba(255,255,255,.05)}
</style>
""", unsafe_allow_html=True)


# =========================
# UTILS
# =========================

def _rerun():
    try: st.rerun()
    except Exception: st.experimental_rerun()

def safe_int(v, default=0) -> int:
    try:
        if v is None or (isinstance(v,str) and v.strip()=="") or pd.isna(v): return default
        return int(v)
    except Exception: return default

def safe_str(v, default="") -> str:
    try:
        if v is None or pd.isna(v): return default
        s = str(v).strip()
        return s if s else default
    except Exception: return default

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

def order_results(results: List[Dict]) -> List[Dict]:
    rank = {name: i for i, name in enumerate(BOX_SCORE_ORDER)}
    return sorted(results, key=lambda r: (rank.get(str(r.get("Stat")), 10_000), str(r.get("Stat"))))


# =========================
# FAVORITES PERSISTENCE
# =========================

def _load_favorites() -> List[dict]:
    if not FAV_FILE.exists(): return []
    return json.loads(FAV_FILE.read_text("utf-8"))

def _save_favorites(rows: List[dict]) -> None:
    FAV_FILE.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


# =========================
# IMAGES
# =========================

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
    s = requests.Session(); s.headers.update({"User-Agent":"Mozilla/5.0 (HotShotPropsAI)"})
    for pid in ids:
        for tmpl in url_templates:
            try:
                r = s.get(tmpl.format(pid=int(pid)), timeout=6)
                if r.status_code == 200 and r.content and len(r.content) > 1500:
                    return r.content
            except Exception:
                continue
    return None

def _safe_image_from_bytes(photo_bytes: Optional[bytes], size=(240, 240)) -> Image.Image:
    if not photo_bytes:
        img = Image.new("RGB", size, (15, 23, 42))
        return img
    try:
        im = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
        return im.resize(size)
    except Exception:
        return Image.new("RGB", size, (15, 23, 42))


# =========================
# PLAYERS / TEAMS
# =========================

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
    for c in ["TEAM_ID","teamId","TeamID"]:
        if "team_id" not in out.columns and c in out.columns: out = out.rename(columns={c:"team_id"})
    for c in ["TEAM_ABBREVIATION","team_abbreviation","TeamAbbreviation"]:
        if "team_abbr" not in out.columns and c in out.columns: out = out.rename(columns={c:"team_abbr"})
    return out

def normalize_players_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["id","full_name","team_id","team_abbr","nba_person_id"])
    p = df.copy()
    id_done = False
    for c in ["id","player_id","PLAYER_ID","PersonId","PERSON_ID"]:
        if c in p.columns:
            if c != "id": p = p.rename(columns={c:"id"}); id_done=True; break
    if not id_done: p["id"] = pd.NA

    if "full_name" not in p.columns:
        if {"first_name","last_name"}.issubset(p.columns):
            p["full_name"] = (p["first_name"].astype(str).str.strip()+" "+p["last_name"].astype(str).str.strip())
        elif "PLAYER" in p.columns:
            p["full_name"] = p["PLAYER"].astype(str)
        elif "DISPLAY_FIRST_LAST" in p.columns:
            p["full_name"] = p["DISPLAY_FIRST_LAST"].astype(str)
        else:
            p["full_name"] = "Unknown"

    p = _safe_get_team_cols(p)
    if "team_abbr" not in p.columns or p["team_abbr"].isna().all():
        teams = load_teams_bdl()
        if "team_id" in p.columns and not p["team_id"].isna().all() and not teams.empty:
            p = p.merge(teams.rename(columns={"id":"team_id","abbreviation":"team_abbr"}), on="team_id", how="left")

    nba_done = False
    for c in ["nba_person_id","PERSON_ID","personId","nba_id"]:
        if c in p.columns:
            if c != "nba_person_id": p = p.rename(columns={c:"nba_person_id"}); nba_done=True; break
    if not nba_done: p["nba_person_id"] = pd.NA

    for col in ["team_id","team_abbr"]:
        if col not in p.columns: p[col] = pd.NA

    cols = ["id","full_name","team_id","team_abbr","nba_person_id"]
    out = p[cols].dropna(subset=["id"]).drop_duplicates(subset=["id"]).sort_values("full_name").reset_index(drop=True)
    out = out[out["full_name"].astype(str).str.strip().ne("")]
    return out

@st.cache_data(show_spinner=False)
def bdl_fetch_active_players_direct() -> pd.DataFrame:
    out = []; s = requests.Session(); s.headers.update({"User-Agent":"Mozilla/5.0 (HotShotPropsAI)"})
    url = "https://www.balldontlie.io/api/v1/players"; page = 1
    while True:
        try:
            r = s.get(url, params={"active":"true","per_page":100,"page":page}, timeout=8)
            j = r.json(); data = j.get("data", []); 
            if not data: break
            out.extend(data); page = j.get("meta",{}).get("next_page"); 
            if not page: break
        except Exception: break
    return pd.DataFrame(out)

@st.cache_data(show_spinner=False)
def load_player_list(season: str="2025-26") -> pd.DataFrame:
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
        if not p2.empty: return p2[["id","full_name","team_id","team_abbr","nba_person_id"]]
    except Exception:
        pass
    favs = _load_favorites()
    if favs:
        p4 = pd.DataFrame(favs)
        for col in ["team_id","team_abbr","nba_person_id"]:
            if col not in p4.columns: p4[col] = None
        p4["team_id"] = p4["team_id"].apply(lambda x: safe_int(x, 0))
        p4["team_abbr"] = p4["team_abbr"].apply(lambda x: safe_str(x, ""))
        p4["nba_person_id"] = p4["nba_person_id"].apply(lambda x: None if x in (None,"",0) else int(x))
        return p4[["id","full_name","team_id","team_abbr","nba_person_id"]].drop_duplicates(subset=["id"]).sort_values("full_name").reset_index(drop=True)
    return pd.DataFrame(columns=["id","full_name","team_id","team_abbr","nba_person_id"])

@st.cache_data(show_spinner=False)
def load_logs(player_id: int, season: str) -> pd.DataFrame:
    return dfetch.get_player_game_logs_nba(player_id, season).copy()

def pick_player_row(players: pd.DataFrame, selected_name: str) -> Optional[pd.Series]:
    if players is None or players.empty: return None
    names = players["full_name"].astype(str).values
    idx = np.where(names == str(selected_name))[0]
    if idx.size: return players.iloc[int(idx[0])]
    return players.iloc[0]


# =========================
# OPP DEFENSE & PACE (BDL)
# =========================

def _bdl_paginate(url: str, params: Dict) -> List[Dict]:
    out: List[Dict] = []
    s = requests.Session(); s.headers.update({"User-Agent":"Mozilla/5.0 (HotShotPropsAI)"})
    page = 1
    while True:
        q = params.copy(); q["page"] = page; q.setdefault("per_page",100)
        try:
            r = s.get(url, params=q, timeout=8); r.raise_for_status()
            j = r.json(); out.extend(j.get("data", []))
            npg = j.get("meta",{}).get("next_page"); 
            if not npg: break
            page = npg
        except Exception: break
    return out

def season_start_year(season: str) -> int: return int(season.split("-")[0])

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
        total = hs + vs
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
    s = requests.Session(); s.headers.update({"User-Agent":"Mozilla/5.0 (HotShotPropsAI)"})
    try:
        r = s.get("https://www.balldontlie.io/api/v1/games",
                 params={"seasons[]":year,"team_ids[]":team_id,"start_date":today,"per_page":100},
                 timeout=8)
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


# =========================
# FEATURES
# =========================

def compute_opponent_strength(df: pd.DataFrame) -> pd.DataFrame:
    opp = (df.groupby("OPP_TEAM")[["PTS","REB","AST"]]
           .mean().rename(columns={"PTS":"OPP_ALLOW_PTS","REB":"OPP_ALLOW_REB","AST":"OPP_ALLOW_AST"}))
    return df.join(opp, on="OPP_TEAM")

def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["IS_HOME"] = df["MATCHUP"].apply(lambda x: 1 if isinstance(x,str) and ("vs" in x) else 0)
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
    df = compute_opponent_strength(df); df = add_context_features(df)
    df = df.dropna(subset=["PTS","REB","AST"])
    dp = load_team_defense_pace(season); df = attach_defense_pace(df, dp)
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
        if len(out.index)>0: out.loc[out.index[0], f"{c}_L1"] = np.nan
        out[f"{c}_L3"]  = out[c].shift(3); out[f"{c}_L5"]  = out[c].shift(5)
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
    X = X.copy(); X = X.ffill(); med = X.median(numeric_only=True); X = X.fillna(med); X = X.fillna(0.0); return X


# =========================
# MODELING WRAPPERS
# =========================

def _confidence_from_error(mae: float, mse: float, y_scale: float) -> float:
    if not np.isfinite(y_scale) or y_scale <= 0: y_scale = 1.0
    rmse = np.sqrt(mse) if np.isfinite(mse) else (mae if np.isfinite(mae) else np.nan)
    if not np.isfinite(mae) and not np.isfinite(rmse): return 60.0
    mae = 0.0 if not np.isfinite(mae) else mae; rmse = 0.0 if not np.isfinite(rmse) else rmse
    norm = 0.5*(mae/y_scale) + 0.5*(rmse/y_scale)
    return round(float(np.clip(np.exp(-norm), 0.35, 0.95) * 100.0), 1)

def _apply_model_budget(manager: ModelManager, budget: str) -> None:
    if budget == "Full ensemble": return
    if budget == "Lite (3 models)": wanted = {"elasticnet","hgb","stack"}
    else: wanted = {"elasticnet"}
    if hasattr(manager,"set_model_whitelist"): manager.set_model_whitelist(list(wanted))

def get_or_train_model_cached(player_id: int, season: str, stat: str, X: pd.DataFrame, y: np.ndarray, budget: str) -> ModelManager:
    key = _hash_frame_small(X, y, player_id, season, stat) + f"|{budget}"
    ss: Dict[str, ModelManager] = st.session_state.setdefault("model_cache", {})
    if key in ss: return ss[key]
    manager = ModelManager(random_state=42); _apply_model_budget(manager, budget); manager.train(X, y); ss[key]=manager; return manager

def train_predict_for_stat(player_id: int, season: str, stat: str, features: pd.DataFrame,
                           fast_mode: bool, model_budget: str, upcoming_ctx: Optional[Dict[str, float]]=None) -> Dict[str, float]:
    y_all = build_target(features, stat).to_numpy()
    X_all = select_X_for_stat(features, stat)
    df_join = pd.concat([pd.Series(y_all, name="TARGET", index=X_all.index), X_all], axis=1).loc[lambda d: ~d["TARGET"].isna()]
    if df_join.empty:
        return {"Stat": stat, "Prediction": float("nan"), "Best Model":"NoData", "MAE": float("nan"), "MSE": float("nan"), "Confidence": 50.0, "Scale": 1.0}
    y_final = df_join["TARGET"].to_numpy(dtype=float)
    X_final = _impute_features(df_join.drop(columns=["TARGET"]))
    if len(X_final) > N_TRAIN: X_final = X_final.iloc[-N_TRAIN:]; y_final = y_final[-N_TRAIN:]
    y_scale = float(np.nanstd(y_final) or 1.0)
    X_next = X_final.tail(1).copy()
    if upcoming_ctx:
        for k,v in upcoming_ctx.items():
            if k in X_next.columns: X_next.loc[:,k] = v

    if fast_mode or len(X_final) < MIN_ROWS_FOR_MODEL:
        pred = float(np.nanmean(y_final[-10:])) if np.isfinite(y_final[-10:]).any() else float("nan")
        w = y_final[-10:] if len(y_final) >= 5 else y_final
        mae_b = float(np.nanmean(np.abs(w - np.nanmean(w)))) if w.size else np.nan
        mse_b = float(np.nanmean((w - np.nanmean(w))**2)) if w.size else np.nan
        conf = _confidence_from_error(mae_b, mse_b, y_scale)
        return {"Stat": stat, "Prediction": pred, "Best Model":"Baseline", "MAE": mae_b, "MSE": mse_b, "Confidence": conf, "Scale": y_scale}

    try:
        manager = get_or_train_model_cached(player_id, season, stat, X_final, y_final, model_budget)
        _ = manager.predict(X_next); best = manager.best_model()
        conf = _confidence_from_error(float(best.mae), float(best.mse), y_scale)
        return {"Stat": stat, "Prediction": float(best.prediction), "Best Model": best.name, "MAE": float(best.mae), "MSE": float(best.mse), "Confidence": conf, "Scale": y_scale}
    except Exception:
        pred = float(np.nanmean(y_final[-10:])) if np.isfinite(y_final[-10:]).any() else float("nan")
        conf = _confidence_from_error(np.nan, np.nan, y_scale)
        return {"Stat": stat, "Prediction": pred, "Best Model":"Baseline(Fallback)", "MAE": float("nan"), "MSE": float("nan"), "Confidence": conf, "Scale": y_scale}


# =========================
# RENDERING
# =========================

def render_hotshot_tiles(results: List[Dict]):
    res_map = {r["Stat"]: r for r in results}
    def val(k):
        v = res_map.get(k, {}).get("Prediction", np.nan)
        return "—" if not np.isfinite(v) else f"{float(v):.1f}"
    st.markdown('<div class="tiles">', unsafe_allow_html=True)
    for stat in CARD_ORDER_TOP:
        st.markdown(f"""
<div class="tile">
  <div class="tile-head">{stat}</div>
  <div class="tile-body">{val(stat)}</div>
</div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="tiles bottom">', unsafe_allow_html=True)
    for stat in CARD_ORDER_BOTTOM:
        st.markdown(f"""
<div class="tile">
  <div class="tile-head">{stat}</div>
  <div class="tile-body">{val(stat)}</div>
</div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def results_to_table(results: List[Dict]) -> pd.DataFrame:
    ordered = order_results(results)
    df = pd.DataFrame(ordered)
    df["Pred"] = pd.to_numeric(df["Prediction"], errors="coerce").round(2)
    df["MAE"]  = pd.to_numeric(df["MAE"], errors="coerce").round(2)
    df["MSE"]  = pd.to_numeric(df["MSE"], errors="coerce").round(2)
    df["Confidence"] = pd.to_numeric(df.get("Confidence", 60.0), errors="coerce").round(1)
    return df[["Stat","Pred","Best Model","MAE","MSE","Confidence"]].rename(columns={"Best Model":"Model"})

def bar_chart_from_table(df: pd.DataFrame, title: str, color: str):
    order = [s for s in BOX_SCORE_ORDER if s in df["Stat"].unique().tolist()]
    c = alt.Chart(df).mark_bar(color=color).encode(
        x=alt.X("Stat:N", sort=order), y=alt.Y("Pred:Q"),
        tooltip=["Stat","Pred","Model","MAE","MSE","Confidence"]
    ).properties(height=220, title=title)
    st.altair_chart(c, use_container_width=True)

def make_share_image(player_name: str, season: str, photo_bytes: Optional[bytes], table_df: pd.DataFrame, subtitle: str) -> bytes:
    W, H = 1200, 675
    bg = Image.new("RGB", (W, H), color=(11, 15, 20))
    draw = ImageDraw.Draw(bg)
    head = _safe_image_from_bytes(photo_bytes, size=(240, 240))
    bg.paste(head, (50, 80))
    font_big = ImageFont.load_default(); font_med = ImageFont.load_default()
    draw.text((310, 90), "Hot Shot Props AI", fill=(255, 255, 255), font=font_big)
    draw.text((310, 120), f"{player_name}  •  {season}", fill=(230, 230, 230), font=font_med)
    draw.text((310, 150), subtitle, fill=(170, 180, 190), font=font_med)
    table = table_df.copy().head(10)
    col_x = [310, 540, 720, 875, 1010]; headers = ["Stat","Pred","Model","MAE","MSE"]
    for cx, htxt in zip(col_x, headers): draw.text((cx, 200), htxt, fill=(203, 213, 225), font=font_med)
    y = 230
    for _, r in table.iterrows():
        vals = [str(r["Stat"]), f"{r['Pred']}", str(r["Model"]), f"{r['MAE']}", f"{r['MSE']}"]
        for cx, v in zip(col_x, vals): draw.text((cx, y), v, fill=(226, 232, 240), font=font_med)
        y += 28
    buf = io.BytesIO(); bg.save(buf, format="PNG", optimize=True); return buf.getvalue()


# =========================
# PAGES
# =========================

def page_lab(players: pd.DataFrame):
    # Header bar
    st.markdown('<div class="topbar"><h1 class="app-title"><span class="logo">🏀</span> Hot Shot Props AI</h1><div></div></div>', unsafe_allow_html=True)

    if players is None or players.empty:
        st.error("No active players found. Use Refresh players in the sidebar.")
        return

    names_list = players["full_name"].astype(str).tolist()
    c1, c2 = st.columns([1,3])
    with c1:
        name = st.selectbox("Select Player", names_list, key="lab_player")
        row = pick_player_row(players, name)
        if row is None: st.error("Player not found"); return
        pid = safe_int(row.get("id"), 0)
        tid = safe_int(row.get("team_id"), 0)
        abbr = safe_str(row.get("team_abbr"), "")
        color = TEAM_META.get(abbr,{}).get("color","#e85d12")
        nba_pid = None if pd.isna(row.get("nba_person_id")) else safe_int(row.get("nba_person_id"), None)
        fast_mode = st.toggle("Fast mode", value=False, help="No model training; quick baseline")
        budget = st.radio("Model budget", ["Full ensemble","Lite (3 models)","Single (Lasso)"], index=1, horizontal=False)
        run = st.button("Get Predictions", type="primary")
    with c2:
        photo = get_player_photo_bytes(pid, nba_pid)
        logo = nba_logo_url(abbr)
        st.markdown(f"""
<div class="player-card">
  <div>
    <img class="player-photo" src="data:image/png;base64,{(photo or b'').hex()}" style="display:none"/>
</div>
  <div style="display:flex;gap:18px;align-items:center">
    <img src="data:image/png;base64,{(photo or b'').hex()}" style="display:none"/>
  </div>
</div>""", unsafe_allow_html=True)
        # Streamlit can't render hex directly; show image + logo separately:
        st.image(_safe_image_from_bytes(photo, (240,240)), caption=name)
        if logo: st.image(logo, width=180)

    if not run:
        st.info("Pick a player and click **Get Predictions**.")
        return

    season = "2025-26"
    logs = load_logs(pid, season)
    if logs.empty:
        year = dt.date.today().year
        season = f"{year-1}-{str(year)[-2:]}"
        logs = load_logs(pid, season)
    if logs.empty: st.error("No logs found"); return

    next_game = auto_next_opponent(tid, season) if tid>0 else None
    features = build_all_features(logs, season)

    upcoming_ctx = {}
    subtitle = "Next: N/A"
    if next_game:
        dp = load_team_defense_pace(season)
        opp_row = dp[dp["abbreviation"]==next_game["opp_abbr"]] if not dp.empty else pd.DataFrame()
        if not opp_row.empty:
            upcoming_ctx["OPP_DEF_PPG"] = float(opp_row["OPP_DEF_PPG"].iloc[0])
            upcoming_ctx["OPP_DEF_Z"] = float(opp_row["OPP_DEF_Z"].iloc[0])
            upcoming_ctx["OPP_PACE"] = float(opp_row["PACE"].iloc[0])
            upcoming_ctx["OPP_PACE_Z"] = float(opp_row["PACE_Z"].iloc[0])
            upcoming_ctx["OPP_DEF_X_PACE"] = upcoming_ctx["OPP_DEF_Z"] * upcoming_ctx["OPP_PACE_Z"]
        if "GAME_DATE" in features.columns and features["GAME_DATE"].notna().any():
            last_date = pd.to_datetime(features["GAME_DATE"]).max().date()
            rest_days = max(0, (next_game["date"] - last_date).days)
        else:
            rest_days = 2
        upcoming_ctx["IS_HOME"] = 1 if next_game["is_home"] else 0
        upcoming_ctx["REST_DAYS"] = rest_days
        upcoming_ctx["BACK_TO_BACK"] = 1 if rest_days == 1 else 0
        subtitle = f"Next: {('Home' if next_game['is_home'] else 'Away')} vs {next_game['opp_abbr']} on {next_game['date']} · Rest {rest_days}d"

    # Predict (parallel)
    futures, results = {}, []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for stat in PROP_MAP.keys():
            futures[ex.submit(train_predict_for_stat, pid, season, stat, features, fast_mode, budget, upcoming_ctx)] = stat
        for fut in as_completed(futures): results.append(fut.result())

    results = order_results(results)
    # Player header block (name + team)
    team_name = TEAM_META.get(abbr,{}).get("name","")
    st.markdown(f"""
<div class="glow-card" style="margin-top:6px">
  <div class="player-name">{safe_str(name)}</div>
  <div class="player-team">{team_name or abbr}</div>
</div>
""", unsafe_allow_html=True)

    render_hotshot_tiles(results)

    # Actions
    colA, colB = st.columns([1,1])
    with colA:
        if st.button("⭐ Add to Favorites", key="fav_add_btn"):
            favs = _load_favorites()
            if not any(f.get("id")==pid for f in favs):
                favs.append({"id":pid,"full_name":name,"team_id":tid,"team_abbr":abbr,"nba_person_id":None})
                _save_favorites(favs); st.success("Added to favorites.")
            else:
                st.info("Already in favorites.")
    with colB:
        if st.button("⏺ Track Projection", key="track_btn"):
            tbl = results_to_table(results)
            rec = {"player":name,"team":abbr,"season":season,"ts":dt.datetime.now().isoformat(), "rows":tbl.to_dict(orient="records")}
            st.session_state.setdefault("tracker", []).append(rec)
            st.success("Projection saved to tracker.")

    # Chart + CSV + Share
    df_table = results_to_table(results)
    color = TEAM_META.get(abbr,{}).get("color","#e85d12")
    bar_chart_from_table(df_table, title="Predictions", color=color)
    st.download_button("⬇️ Download CSV", data=df_table.to_csv(index=False).encode("utf-8"),
                       file_name=f"{name.replace(' ','_')}_{season}_predictions.csv", mime="text/csv")
    photo = get_player_photo_bytes(pid, None)
    img_bytes = make_share_image(name, season, photo, df_table[["Stat","Pred","Model","MAE","MSE"]], subtitle)
    st.download_button("📸 Share image (PNG)", data=img_bytes,
                       file_name=f"{name.replace(' ','_')}_{season}_predictions.png", mime="image/png")

    # Recent games
    st.subheader("Recent Games")
    cols_show = [c for c in ["GAME_DATE","MATCHUP","PTS","REB","AST","FG3M","STL","BLK","TOV","MIN"] if c in logs.columns]
    st.dataframe(logs[cols_show], use_container_width=True)


def page_favorites(players: pd.DataFrame):
    st.markdown('<div class="topbar"><h1 class="app-title"><span class="logo">🏀</span> Hot Shot Props AI</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="nav"><div class="tab active">Favorites</div><div class="tab">Prop Projection Lab</div><div class="tab">Projection Tracker</div></div>', unsafe_allow_html=True)

    favs = _load_favorites()
    if not favs:
        st.info("No favorites yet. Use **Prop Projection Lab → Add to Favorites**.")
        return

    cols = st.columns(3)
    for i, f in enumerate(favs):
        with cols[i % 3]:
            abbr = safe_str(f.get("team_abbr"), "")
            color = TEAM_META.get(abbr,{}).get("color","#e85d12")
            logo = nba_logo_url(abbr)
            st.markdown(f"""
<div class="glow-card" style="box-shadow:0 0 22px {color}44">
  <div style="display:flex;gap:10px;align-items:center">
    <img src="{logo or ''}" style="width:42px;height:42px;border-radius:8px;border:1px solid #222;background:#111"/>
    <div>
      <div style="font-weight:800;color:#fff">{safe_str(f.get('full_name'),'')}</div>
      <div style="color:#cbd5e1">{abbr}</div>
    </div>
    <div style="margin-left:auto">
      <button class="btn" onclick="">{' '}</button>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
            if st.button("❌ Remove", key=f"del_{safe_int(f.get('id'),0)}"):
                _save_favorites([x for x in favs if safe_int(x.get("id"),0) != safe_int(f.get("id"),0)])
                _rerun()


def page_tracker():
    st.markdown('<div class="topbar"><h1 class="app-title"><span class="logo">🏀</span> Hot Shot Props AI</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="nav"><div class="tab">Favorites</div><div class="tab">Prop Projection Lab</div><div class="tab active">Projection Tracker</div></div>', unsafe_allow_html=True)

    hist = st.session_state.get("tracker", [])
    if not hist:
        st.info("No tracked projections yet. Click **Track Projection** in the Lab tab.")
        return

    # Flatten to CSV-like table
    rows = []
    for rec in hist:
        for r in rec["rows"]:
            rows.append({"Player": rec["player"], "Team": rec["team"], "Season": rec["season"], "Timestamp": rec["ts"], **r})
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.download_button("⬇️ Download Tracker CSV", data=df.to_csv(index=False).encode("utf-8"),
                       file_name="projection_tracker.csv", mime="text/csv")


# =========================
# APP
# =========================

def main():
    st.set_page_config(page_title="Hot Shot Props AI", page_icon="🏀", layout="wide")
    inject_css_hotshot()

    with st.sidebar:
        st.markdown("### Navigation")
        page = st.radio("Go to", ["Favorites","Prop Projection Lab","Projection Tracker"], index=1)
        st.markdown("---")
        if st.button("🔄 Refresh players"):
            load_player_list.clear()
            _rerun()
        st.caption("Auto Opponent • DEF+PACE • Fixed tile order • Share • Tracker")

    players = load_player_list("2025-26")

    if page == "Prop Projection Lab":
        page_lab(players)
    elif page == "Favorites":
        page_favorites(players)
    else:
        page_tracker()


if __name__ == "__main__":
    main()
