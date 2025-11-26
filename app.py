# app.py
"""
NBA Prop Predictor — Elite
- Predictions-only UI (ML hidden)
- Predict, Favorites, Research
- Auto opponent; Defense & Pace + positional difficulty
- Team logo; next opponent with EST time + DefRtg tier
- Animated loader; team-colored metric cards
- Blank selects; stable box-score order
- Favorites: horizontal rows with mini-cards, per-player CSV & Share
- Trading-card share image (mobile 1080x1920) with large fonts
"""

from __future__ import annotations

import os
import io
import json
import hashlib
import zipfile
import datetime as dt
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import altair as alt
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import streamlit as st

# Local modules (unchanged)
import data_fetching as dfetch
from models import ModelManager


# =============================================================================
# CONFIG
# =============================================================================

PROP_MAP = {
    "Points": "PTS",
    "Rebounds": "REB",
    "Assists": "AST",
    "3PM": "FG3M",
    "Steals": "STL",
    "Blocks": "BLK",
    "Turnovers": "TOV",
    "Minutes": "MIN",
    "PRA": ["PTS", "REB", "AST"],
    "PR": ["PTS", "REB"],
    "PA": ["PTS", "AST"],
    "RA": ["REB", "AST"],
}
STAT_COLUMNS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "MIN"]

# box-score order lock (requested)
BOX_SCORE_ORDER = [
    "Points", "Rebounds", "Assists", "3PM", "Steals", "Blocks", "Turnovers", "Minutes",
    "PRA", "PR", "PA", "RA",
]

BASE_COLS = [
    "IS_HOME", "REST_DAYS", "BACK_TO_BACK",
    "OPP_ALLOW_PTS", "OPP_ALLOW_REB", "OPP_ALLOW_AST",
    "OPP_DEF_PPG", "OPP_DEF_Z",
    "OPP_PACE", "OPP_PACE_Z",
    "OPP_DEF_X_PACE",
    "DEF_RTG", "DEF_RTG_PCT",
]

# lightweight positional multipliers
POS_WEIGHTS: Dict[str, Dict[str, float]] = {
    "G": {"PTS": 1.00, "REB": 0.85, "AST": 1.20, "FG3M": 1.20, "STL": 1.05, "BLK": 0.80, "TOV": 1.10, "MIN": 1.00},
    "F": {"PTS": 1.00, "REB": 1.10, "AST": 1.00, "FG3M": 1.00, "STL": 1.00, "BLK": 1.00, "TOV": 1.00, "MIN": 1.00},
    "C": {"PTS": 0.98, "REB": 1.25, "AST": 0.95, "FG3M": 0.70, "STL": 0.95, "BLK": 1.20, "TOV": 1.00, "MIN": 1.00},
}

N_TRAIN = 60
MIN_ROWS_FOR_MODEL = 12
MAX_WORKERS = max(2, min(8, os.cpu_count() or 4))

DATA_DIR = Path("./data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
FAV_FILE = DATA_DIR / "favorites.json"


# =============================================================================
# TEAM META
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
.block-container { padding-top: 1.1rem; max-width: 1240px; }
h1, h2, h3, h4 {
  background: linear-gradient(90deg,#e2e8f0 0%, #60a5fa 40%, #34d399 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  animation: fadeIn 700ms ease both;
}
@keyframes fadeIn { from {opacity:0; transform: translateY(6px)} to {opacity:1; transform:none} }
@keyframes rise { from {opacity:.0; transform: translateY(12px)} to {opacity:1; transform:none} }

.card {
  background: rgba(17,24,39,.6);
  border: 1px solid rgba(255,255,255,.08);
  border-radius: 16px; padding: 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,.25);
  backdrop-filter: blur(6px);
  animation: rise 540ms ease both;
}

/* Metric cards */
.metric-grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 14px; }
.metric-card {
  border-radius: 16px; padding: 16px 14px; color: #fff; border:1px solid rgba(255,255,255,.10);
  transform-origin: center; animation: rise 480ms ease both;
}
.metric-title { font-size: .95rem; opacity:.95; margin-bottom:.35rem; letter-spacing:.25px; }
.metric-value { font-size: 2.1rem; font-weight:800; line-height:1.1; }

/* Mini metrics (favorites rows) */
.mm-wrap { display:flex; flex-wrap:wrap; gap:8px; }
.mm { min-width:86px; padding:10px 12px; border-radius:12px; border:1px solid rgba(255,255,255,.12);
      background: rgba(255,255,255,.04); }
.mm .t { font-size:.78rem; opacity:.9; }
.mm .v { font-size:1.2rem; font-weight:800; }

/* Favorites */
.fav-box {
  background: linear-gradient(180deg, #0b1220 0%, #0a1324 100%);
  border: 1px solid rgba(255,255,255,.12);
  border-radius: 16px;
  padding: 12px 14px;
  box-shadow: 0 12px 34px rgba(0,0,0,.28);
  animation: rise 420ms ease both;
}
.fav-name { font-weight: 800; color: #f3f4f6; line-height: 1.2; font-size: 1.05rem; }
.fav-meta { font-size: .92rem; color: #d1d5db; }

/* Close btn */
.fav-x > button {
  background: #111827 !important; color: #ef4444 !important; border: 1px solid #ef4444 !important;
  border-radius: 10px !important; padding: 4px 10px !important; font-weight: 800 !important;
}
.fav-x > button:hover { filter: brightness(1.15); }

/* Buttons */
.stButton>button { border-radius: 12px; padding: 10px 14px; font-weight: 600;
  background: linear-gradient(90deg,#0ea5e9,#22c55e); border: none; }
.stButton>button:hover { filter: brightness(1.05); }

/* Tags */
.tag { display:inline-block; padding:.2rem .55rem; border-radius:999px; font-size:.75rem;
  background:rgba(99,102,241,.15); border:1px solid rgba(99,102,241,.35); }
.badge { display:inline-block; padding:.25rem .6rem; border-radius:10px; font-size:.8rem;
  background:rgba(34,197,94,.15); border:1px solid rgba(34,197,94,.4); color:#d1fae5; }

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
  50%  { transform: translateX(50%) translateY(2px); }
  100% { transform: translateX(110%) translateY(0); }
}
.loader-text { color:#cbd5e1; font-size:.9rem; margin-bottom:.35rem; }
</style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# UTILS
# =============================================================================

def _rerun():
    try: st.rerun()
    except Exception: st.experimental_rerun()

def safe_int(v, default=0) -> int:
    try:
        if v is None or (isinstance(v, str) and v.strip()=="") or pd.isna(v): return default
        return int(v)
    except Exception:
        return default

def safe_str(v, default="") -> str:
    try:
        if v is None or pd.isna(v): return default
        s = str(v)
        return s if s.strip() != "" else default
    except Exception:
        return default

def clamp(x, lo, hi): return max(lo, min(hi, x))

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
    s0 = season_start_year(season); return f"{s0-1}-{str(s0)[-2:]}"


def order_results(results: List[Dict]) -> List[Dict]:
    rank = {name: i for i, name in enumerate(BOX_SCORE_ORDER)}
    return sorted(results, key=lambda r: (rank.get(str(r.get("Stat")), 10_000), str(r.get("Stat"))))


# =============================================================================
# PERSISTENCE
# =============================================================================

def _load_favorites() -> List[dict]:
    if not FAV_FILE.exists(): return []
    return json.loads(FAV_FILE.read_text("utf-8"))

def _save_favorites(rows: List[dict]) -> None:
    FAV_FILE.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


# =============================================================================
# PHOTOS & LOGOS
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

def default_placeholder_bytes(size=(72,72)) -> bytes:
    img = Image.new("RGB", size, (17, 26, 45))
    d = ImageDraw.Draw(img)
    d.rectangle([6, 6, size[0]-6, size[1]-6], outline=(90, 100, 120), width=2)
    buf = io.BytesIO(); img.save(buf, format="PNG"); return buf.getvalue()

def get_logo_or_default(team_abbr: str, size=(128,128)) -> bytes:
    url = nba_logo_url(team_abbr)
    if url:
        try:
            r = requests.get(url, timeout=6)
            if r.status_code == 200 and r.content:
                return r.content
        except Exception:
            pass
    return default_placeholder_bytes(size)

def get_photo_or_logo_bytes(fav: dict) -> bytes:
    pid = safe_int(fav.get("id"), 0)
    nba_pid = fav.get("nba_person_id")
    nba_pid = None if nba_pid in (None, "", 0, pd.NA) else safe_int(nba_pid, None)
    head = get_player_photo_bytes(pid, nba_pid)
    if head:
        return head
    abbr = safe_str(fav.get("team_abbr"), "")
    return get_logo_or_default(abbr)


# =============================================================================
# TEAMS / PLAYERS
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

@st.cache_data(show_spinner=False)
def team_abbr_to_id() -> dict:
    t = load_teams_bdl()
    if t.empty:
        return {}
    return {str(r["abbreviation"]): int(r["id"]) for _, r in t.iterrows() if pd.notna(r["id"])}

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
        return pd.DataFrame(columns=["id","full_name","team_id","team_abbr","nba_person_id","position"])
    p = df.copy()

    # id
    for c in ["id","player_id","PLAYER_ID","PersonId","PERSON_ID"]:
        if c in p.columns:
            if c != "id": p = p.rename(columns={c: "id"}); break
    if "id" not in p.columns: p["id"] = pd.NA

    # name
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
            p = p.merge(teams.rename(columns={"id":"team_id","abbreviation":"team_abbr"}), on="team_id", how="left")

    for c in ["nba_person_id","PERSON_ID","personId","nba_id"]:
        if c in p.columns:
            if c != "nba_person_id": p = p.rename(columns={c:"nba_person_id"}); break
    if "nba_person_id" not in p.columns: p["nba_person_id"] = pd.NA

    if "position" not in p.columns:
        for c in ["POSITION","pos","Pos","Position"]:
            if c in p.columns: p = p.rename(columns={c:"position"}); break
    if "position" not in p.columns: p["position"] = pd.NA

    cols = ["id","full_name","team_id","team_abbr","nba_person_id","position"]
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
def bdl_fetch_all_players_direct() -> pd.DataFrame:
    out = []
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (PropPredictor)"})
    url = "https://www.balldontlie.io/api/v1/players"
    page = 1
    while True:
        try:
            r = s.get(url, params={"per_page": 100, "page": page}, timeout=8)
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
            return p[["id","full_name","team_id","team_abbr","nba_person_id","position"]]
    except Exception:
        pass
    try:
        raw2 = bdl_fetch_active_players_direct()
        p2 = normalize_players_df(raw2)
        if not p2.empty:
            return p2[["id","full_name","team_id","team_abbr","nba_person_id","position"]]
    except Exception:
        pass
    try:
        fb = dfetch.get_player_list_nba()
        p3 = normalize_players_df(_filter_active_players(fb, season))
        if not p3.empty:
            return p3[["id","full_name","team_id","team_abbr","nba_person_id","position"]]
    except Exception:
        pass
    favs = _load_favorites()
    if favs:
        p4 = pd.DataFrame(favs)
        for col in ["team_id","team_abbr","nba_person_id","position"]:
            if col not in p4.columns: p4[col] = None
        p4["team_id"] = p4["team_id"].apply(lambda x: safe_int(x, 0))
        p4["team_abbr"] = p4["team_abbr"].apply(lambda x: safe_str(x, ""))
        p4["nba_person_id"] = p4["nba_person_id"].apply(lambda x: None if x in (None, "", 0) else int(x))
        return p4[["id","full_name","team_id","team_abbr","nba_person_id","position"]].drop_duplicates(subset=["id"]).sort_values("full_name").reset_index(drop=True)
    return pd.DataFrame(columns=["id","full_name","team_id","team_abbr","nba_person_id","position"])

@st.cache_data(show_spinner=False)
def load_player_list_all() -> pd.DataFrame:
    try:
        fb = dfetch.get_player_list_nba()
        p = normalize_players_df(fb)
        if not p.empty:
            return p[["id","full_name","team_id","team_abbr","nba_person_id","position"]]
    except Exception:
        pass
    try:
        raw2 = bdl_fetch_all_players_direct()
        p2 = normalize_players_df(raw2)
        if not p2.empty:
            return p2[["id","full_name","team_id","team_abbr","nba_person_id","position"]]
    except Exception:
        pass
    return load_player_list("2025-26")

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
# OPPONENT DEFENSE & PACE
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
        return pd.DataFrame(columns=["team_id","abbreviation","OPP_DEF_PPG","OPP_DEF_Z","PACE","PACE_Z","DEF_RTG","DEF_RTG_PCT","TIER"])

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

    agg["DEF_RTG"] = (agg["OPP_DEF_PPG"] / (agg["PACE"].replace(0, np.nan) / 100.0)).fillna(agg["OPP_DEF_PPG"])
    agg["DEF_RTG_PCT"] = agg["DEF_RTG"].rank(pct=True)

    def tier(p):
        if p <= 0.2: return "Elite (tough)"
        if p <= 0.4: return "Strong"
        if p <= 0.6: return "Average"
        if p <= 0.8: return "Weak"
        return "Soft (easy)"
    agg["TIER"] = agg["DEF_RTG_PCT"].apply(tier)
    return agg

def resolve_team_for_player(row: pd.Series, logs: Optional[pd.DataFrame]) -> tuple[int, str]:
    """Resolve team reliably from row or last logs."""
    tid = safe_int(row.get("team_id"), 0)
    abbr = safe_str(row.get("team_abbr"), "")
    if (tid <= 0) and abbr:
        tid = team_abbr_to_id().get(abbr, 0)
    if (tid <= 0 or not abbr) and logs is not None and not logs.empty:
        if not abbr and "TEAM_ABBREVIATION" in logs.columns and logs["TEAM_ABBREVIATION"].notna().any():
            abbr = safe_str(logs["TEAM_ABBREVIATION"].dropna().astype(str).iloc[-1], "")
        if (tid <= 0) and "TEAM_ID" in logs.columns and logs["TEAM_ID"].notna().any():
            tid = safe_int(logs["TEAM_ID"].dropna().iloc[-1], 0)
    return tid, abbr

@st.cache_data(show_spinner=False)
def auto_next_opponent(team_id: int, season: str, team_abbr: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Find next game with aggressive fallbacks; returns EST datetime + opp abbr."""
    if (not team_id or team_id <= 0) and team_abbr:
        team_id = team_abbr_to_id().get(team_abbr, 0)
    if not team_id:
        return None

    year = season_start_year(season)
    today = dt.date.today()
    s = requests.Session(); s.headers.update({"User-Agent": "Mozilla/5.0 (PropPredictor)"})

    data = []
    for days in (30, 90, 180, 370):
        try:
            r = s.get(
                "https://www.balldontlie.io/api/v1/games",
                params={
                    "seasons[]": year, "team_ids[]": team_id,
                    "start_date": today.isoformat(), "end_date": (today + dt.timedelta(days=days)).isoformat(),
                    "per_page": 100,
                },
                timeout=10,
            )
            j = r.json()
            data = j.get("data", []) or []
            if data: break
        except Exception:
            data = []

    if not data:
        try:
            data = _bdl_paginate("https://www.balldontlie.io/api/v1/games", {"seasons[]": year, "team_ids[]": team_id})
        except Exception:
            data = []

    if not data:
        return None

    def _dt(g): return dt.datetime.fromisoformat(g["date"].replace("Z", "+00:00"))
    now = dt.datetime.now(tz=ZoneInfo("UTC")) - dt.timedelta(hours=6)
    future = [g for g in data if _dt(g) >= now]
    if not future:
        return None

    nxt = sorted(future, key=_dt)[0]
    h, v = nxt.get("home_team", {}), nxt.get("visitor_team", {})
    is_home = (h.get("id") == team_id)
    opp = v if is_home else h
    dt_utc = _dt(nxt)
    dt_est = dt_utc.astimezone(ZoneInfo("America/New_York"))

    return {
        "opp_id": opp.get("id"),
        "opp_abbr": opp.get("abbreviation"),
        "is_home": is_home,
        "dt_utc": dt_utc,
        "dt_est": dt_est,
        "date": dt_est.date(),
    }


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
        for c in ["OPP_DEF_PPG","OPP_DEF_Z","OPP_PACE","OPP_PACE_Z","OPP_DEF_X_PACE","DEF_RTG","DEF_RTG_PCT"]:
            df[c] = np.nan
        return df
    m = defense_pace.rename(columns={"abbreviation":"OPP_TEAM"})
    df = df.merge(m[["OPP_TEAM","OPP_DEF_PPG","OPP_DEF_Z","PACE","PACE_Z","DEF_RTG","DEF_RTG_PCT"]], on="OPP_TEAM", how="left")
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
        if len(out.index) > 0: out.loc[out.index[0], f"{c}_L1"] = np.nan
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
    X = X.copy(); X = X.ffill()
    med = X.median(numeric_only=True)
    X = X.fillna(med).fillna(0.0)
    return X


# =============================================================================
# MODELING
# =============================================================================

def _apply_model_budget(manager: ModelManager, budget: str) -> None:
    if budget == "Full ensemble": return
    if budget == "Lite (3 models)": wanted = {"elasticnet","hgb","stack"}
    else: wanted = {"elasticnet"}
    try:
        if hasattr(manager, "set_model_whitelist"): manager.set_model_whitelist(list(wanted)); return
        if hasattr(manager, "available_models") and isinstance(manager.available_models, list):
            manager.available_models = [m for m in manager.available_models if str(m).lower() in wanted]
        elif hasattr(manager, "models") and isinstance(manager.models, list):
            manager.models = [m for m in manager.models if str(m).lower() in wanted]
    except Exception: pass

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
    player_id: int, season: str, stat: str, features: pd.DataFrame,
    fast_mode: bool, model_budget: str, upcoming_ctx: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    y_all = build_target(features, stat).to_numpy()
    X_all = select_X_for_stat(features, stat)
    df_join = pd.concat([pd.Series(y_all, name="TARGET", index=X_all.index), X_all], axis=1)
    df_join = df_join.loc[~df_join["TARGET"].isna()].copy()
    if df_join.empty: return {"Stat": stat, "Prediction": float("nan")}
    y_final = df_join["TARGET"].to_numpy(dtype=float)
    X_final = _impute_features(df_join.drop(columns=["TARGET"]))
    if len(X_final) > N_TRAIN:
        X_final = X_final.iloc[-N_TRAIN:].copy(); y_final = y_final[-N_TRAIN:].copy()
    X_next = X_final.tail(1).copy()
    if upcoming_ctx:
        for k, v in upcoming_ctx.items():
            if k in X_next.columns: X_next.loc[:, k] = v
    if fast_mode or len(X_final) < MIN_ROWS_FOR_MODEL:
        pred = float(np.nanmean(y_final[-10:])) if np.isfinite(y_final[-10:]).any() else float("nan")
        return {"Stat": stat, "Prediction": pred}
    try:
        manager = get_or_train_model_cached(player_id, season, stat, X_final, y_final, model_budget)
        _ = manager.predict(X_next)
        best = manager.best_model()
        return {"Stat": stat, "Prediction": float(best.prediction)}
    except Exception:
        pred = float(np.nanmean(y_final[-10:])) if np.isfinite(y_final[-10:]).any() else float("nan")
        return {"Stat": stat, "Prediction": pred}


# =============================================================================
# ADJUSTMENTS (defense + pace + positional)
# =============================================================================

def adjust_predictions(results: List[Dict], opp_row: Optional[pd.Series], position: str) -> List[Dict]:
    if opp_row is None or opp_row.empty:
        return results
    alpha_def = 0.07
    beta_pace = 0.03
    pos_key = "G" if "G" in (position or "") else ("C" if "C" in (position or "") else "F")
    w = POS_WEIGHTS.get(pos_key, POS_WEIGHTS["F"])
    z = float(opp_row.get("OPP_DEF_Z", 0.0))
    pace_z = float(opp_row.get("PACE_Z", 0.0))

    base = {r["Stat"]: float(r["Prediction"]) for r in results if np.isfinite(r.get("Prediction", np.nan))}
    for stat in ["Points","Rebounds","Assists","3PM","Steals","Blocks","Turnovers","Minutes"]:
        if stat in base:
            weight = w.get(PROP_MAP[stat] if isinstance(PROP_MAP[stat], str) else stat, 1.0)
            mult = (1 - alpha_def * z * weight) * (1 + beta_pace * pace_z)
            base[stat] = max(0.0, base[stat] * mult)

    def _sum(stats): return float(sum(base.get(s, np.nan) for s in stats if s in base))
    if all(k in base for k in ["Points","Rebounds","Assists"]): base["PRA"] = _sum(["Points","Rebounds","Assists"])
    if all(k in base for k in ["Points","Rebounds"]): base["PR"] = _sum(["Points","Rebounds"])
    if all(k in base for k in ["Points","Assists"]): base["PA"] = _sum(["Points","Assists"])
    if all(k in base for k in ["Rebounds","Assists"]): base["RA"] = _sum(["Rebounds","Assists"])

    out = [{"Stat": s, "Prediction": base.get(s, np.nan)} for s in BOX_SCORE_ORDER if s in base]
    return out


# =============================================================================
# SHARE IMAGES & UI HELPERS
# =============================================================================

def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for p in paths:
        try: return ImageFont.truetype(p, size=size)
        except Exception: continue
    return ImageFont.load_default()

def results_to_table(results: List[Dict]) -> pd.DataFrame:
    ordered = order_results(results)
    df = pd.DataFrame(ordered)
    df["Pred"] = pd.to_numeric(df["Prediction"], errors="coerce").round(2)
    return df[["Stat","Pred"]]

def table_downloaders(df: pd.DataFrame, filename_prefix: str) -> None:
    csv = df.to_csv(index=False)
    st.download_button("⬇️ Download CSV", data=csv.encode("utf-8"), file_name=f"{filename_prefix}.csv", mime="text/csv")

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
    ordered = order_results(results)
    html = ['<div class="metric-grid">']
    for i, r in enumerate(ordered):
        stat = str(r["Stat"])
        pred = f'{float(r["Prediction"]):.2f}' if np.isfinite(r["Prediction"]) else "—"
        html.append(f"""
<div class="metric-card" style="background: linear-gradient(135deg, {base} 0%, {soft} 95%); animation-delay:{i*0.03:.2f}s">
  <div class="metric-title">{stat}</div>
  <div class="metric-value">{pred}</div>
</div>""")
    html.append("</div>")
    st.markdown("\n".join(html), unsafe_allow_html=True)

def make_share_image_trading_card(
    player_name: str, team_abbr: str, team_name: str, team_color: str,
    season: str, next_info: str, photo_bytes: Optional[bytes], logo_bytes: Optional[bytes],
    df_table: pd.DataFrame, title_suffix: str = "Projections",
) -> bytes:
    W, H = 1080, 1920
    base_col = team_color.lstrip("#") if team_color else "60a5fa"
    r = int(base_col[0:2], 16); g = int(base_col[2:4], 16); b = int(base_col[4:6], 16)

    bg = Image.new("RGB", (W, H), color=(8, 12, 22))
    overlay = Image.new("RGB", (W, H), (r, g, b)).filter(ImageFilter.GaussianBlur(radius=210))
    bg = Image.blend(bg, overlay, alpha=0.22)

    draw = ImageDraw.Draw(bg)
    draw.rectangle([0, 0, W, 180], fill=(20, 26, 40))
    draw.line([(0, 180), (W, 180)], fill=(255, 255, 255, 44), width=1)

    if logo_bytes:
        try:
            logo = Image.open(io.BytesIO(logo_bytes)).convert("RGBA").resize((128, 128))
            bg.paste(logo, (48, 26), logo)
        except Exception:
            pass

    head = _safe_image_from_bytes(photo_bytes, size=(540, 540)).convert("RGBA")
    mask = Image.new("L", (540, 540), 0); ImageDraw.Draw(mask).ellipse((0, 0, 540, 540), fill=255)
    head.putalpha(mask); bg.paste(head, (270, 160), head)

    f_title = get_font(64); f_sub = get_font(36); f_small = get_font(30)
    f_metric = get_font(80); f_metric_title = get_font(36)

    card = Image.new("RGBA", (980, 1080), (16, 22, 36, 245))
    cd = ImageDraw.Draw(card)
    cd.rounded_rectangle([0, 0, 980, 1080], radius=34, outline=(255, 255, 255, 50), width=2)
    cd.rectangle([0, 0, 980, 140], fill=(r, g, b, 235))

    cd.text((28, 24), f"{player_name}", fill=(255, 255, 255), font=f_title)
    cd.text((28, 86), f"{team_name} ({team_abbr})  •  {season}  •  {title_suffix}", fill=(245, 248, 255), font=f_sub)
    cd.text((28, 120), next_info, fill=(225, 232, 247), font=f_small)

    stats_show = [s for s in BOX_SCORE_ORDER if s in df_table["Stat"].tolist()][:8]
    grid = []
    for s in stats_show:
        val = df_table[df_table["Stat"] == s]["Pred"].values
        pred = f"{float(val[0]):.1f}" if len(val) else "—"
        grid.append((s, pred))

    x0, y0 = 40, 190; w, h = 430, 170; gap_x, gap_y = 40, 34
    for i, (s, pred) in enumerate(grid):
        cx = x0 + (i % 2) * (w + gap_x); cy = y0 + (i // 2) * (h + gap_y)
        cd.rounded_rectangle([cx, cy, cx + w, cy + h], radius=20, fill=(12, 16, 28, 255), outline=(255, 255, 255, 34), width=2)
        cd.text((cx + 18, cy + 16), s, fill=(208, 218, 242), font=f_metric_title)
        cd.text((cx + 18, cy + 72), pred, fill=(255, 255, 255), font=f_metric)

    bg.paste(card, (50, 720), card)
    buf = io.BytesIO(); bg.save(buf, format="PNG", optimize=True); return buf.getvalue()

def bar_chart_from_table(df: pd.DataFrame, title: str, color: str | None = None):
    order = [s for s in BOX_SCORE_ORDER if s in df["Stat"].unique().tolist()]
    c = alt.Chart(df).mark_bar(color=color).encode(
        x=alt.X("Stat:N", sort=order),
        y=alt.Y("Pred:Q"),
        tooltip=["Stat","Pred"],
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
    if idx.size: return players.iloc[int(idx[0])]
    return None


# =============================================================================
# PAGES — PREDICT
# =============================================================================

def _opp_line(next_game: Optional[Dict], dp: pd.DataFrame) -> Tuple[str, Optional[pd.Series]]:
    if not next_game: return "Next: N/A", None
    row = dp[dp["abbreviation"] == next_game["opp_abbr"]].head(1)
    if row.empty:
        side = "Home" if next_game["is_home"] else "Away"
        when_str = next_game["dt_est"].strftime("%a, %b %d %I:%M %p ET")
        return f"Next: {side} vs {next_game['opp_abbr']} — {when_str}", None
    r = row.iloc[0]
    side = "Home" if next_game["is_home"] else "Away"
    when_str = next_game["dt_est"].strftime("%a, %b %d %I:%M %p ET")
    txt = f"Next: {side} vs {next_game['opp_abbr']} — {when_str} • DefRtg {r['DEF_RTG']:.1f} ({r['TIER']})"
    return txt, r

def page_predict(players: pd.DataFrame):
    st.header("NBA Prop Predictor — Elite")
    st.caption("Opponent-aware projections.")

    if players is None or players.empty:
        st.error("No active players available. Click **Refresh players** in the sidebar and try again.")
        return

    names_list = players["full_name"].astype(str).tolist()
    placeholder_option = "— Select Player —"
    options = [placeholder_option] + names_list

    col_left, col_right = st.columns([1, 3])
    with col_left:
        name = st.selectbox("Select Player", options, index=0, key="predict_player")
        row = None if name == placeholder_option else pick_player_row(players, name)
        if row is None:
            st.info("Pick a player to continue.")
            return
        player_id = safe_int(row.get("id"), 0)
        team_abbr = safe_str(row.get("team_abbr"), "")
        position = safe_str(row.get("position"), "")
        team_meta = TEAM_META.get(team_abbr, {})
        team_name = team_meta.get("name", team_abbr or "")
        team_color = team_meta.get("color", "#60a5fa")
        team_logo = nba_logo_url(team_abbr)
        fast_mode = False; model_budget = "Single (Lasso)"
        run = st.button("Get Projections")
    with col_right:
        nba_pid = None if pd.isna(row.get("nba_person_id")) else safe_int(row.get("nba_person_id"), None)
        photo = get_player_photo_bytes(player_id, nba_pid)
        st.image(_safe_image_from_bytes(photo, (240, 240)), caption=name)
        if team_logo or team_name:
            cols = st.columns([1,4])
            with cols[0]:
                if team_logo: st.image(team_logo, width=64)
            with cols[1]:
                st.markdown(f"**Team:** {team_name} ({team_abbr})")

    if not run:
        st.info("Click **Get Projections** to compute.")
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

    # resolve team & upcoming
    team_id_resolved, abbr_resolved = resolve_team_for_player(row, logs)
    dp = load_team_defense_pace(season)
    next_game = auto_next_opponent(team_id_resolved, season, abbr_resolved)

    load_ph = st.empty(); show_basketball_loader(load_ph, "Building features…")
    features = build_all_features(logs, season)
    load_ph.empty()

    upcoming_ctx = {}
    rest_days = None
    if next_game:
        txt, opp_row = _opp_line(next_game, dp)
        if opp_row is not None:
            upcoming_ctx.update({
                "OPP_DEF_PPG": float(opp_row["OPP_DEF_PPG"]),
                "OPP_DEF_Z": float(opp_row["OPP_DEF_Z"]),
                "OPP_PACE": float(opp_row["PACE"]),
                "OPP_PACE_Z": float(opp_row["PACE_Z"]),
                "OPP_DEF_X_PACE": float(opp_row["OPP_DEF_Z"] * opp_row["PACE_Z"]),
                "DEF_RTG": float(opp_row["DEF_RTG"]),
                "DEF_RTG_PCT": float(opp_row["DEF_RTG_PCT"]),
            })
        if "GAME_DATE" in features.columns and features["GAME_DATE"].notna().any():
            last_date = pd.to_datetime(features["GAME_DATE"]).max().date()
            next_date = next_game["dt_est"].date() if isinstance(next_game.get("dt_est"), dt.datetime) else next_game.get("date")
            rest_days = max(0, (next_date - last_date).days) if isinstance(next_date, dt.date) else None
        upcoming_ctx["IS_HOME"] = 1 if next_game and next_game["is_home"] else 0
        if rest_days is not None:
            upcoming_ctx["REST_DAYS"] = rest_days; upcoming_ctx["BACK_TO_BACK"] = 1 if rest_days == 1 else 0
    else:
        txt, opp_row = "Next: N/A", None

    st.markdown(f'<span class="badge">{txt}</span>', unsafe_allow_html=True)

    load2 = st.empty(); show_basketball_loader(load2, "Computing projections…")
    futures, results = {}, []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for stat in PROP_MAP.keys():
            futures[ex.submit(train_predict_for_stat, player_id, season, stat, features, fast_mode, model_budget, upcoming_ctx)] = stat
        for fut in as_completed(futures): results.append(fut.result())
    load2.empty()
    results = order_results(results)

    # Adjust by opponent defense/pace + position
    results = adjust_predictions(results, opp_row, position)

    st.subheader("Projected Props")
    render_metric_cards(results, TEAM_META.get(team_abbr, {}).get("color", "#60a5fa"))

    df_table = results_to_table(results)
    table_downloaders(df_table, filename_prefix=f"{name.replace(' ','_')}_{season}_projections")

    logo_bytes = get_logo_or_default(team_abbr) if team_abbr else None
    img_bytes = make_share_image_trading_card(
        player_name=name, team_abbr=team_abbr, team_name=team_name,
        team_color=TEAM_META.get(team_abbr,{}).get("color","#60a5fa"),
        season=season, next_info=txt, photo_bytes=photo, logo_bytes=logo_bytes,
        df_table=df_table[["Stat","Pred"]], title_suffix="Projections",
    )
    st.download_button("📸 Share image (PNG, mobile)", data=img_bytes, file_name=f"{name.replace(' ','_')}_{season}_card.png", mime="image/png")

    st.divider(); st.subheader("Recent Games")
    cols_show = [c for c in ["GAME_DATE","MATCHUP","PTS","REB","AST","FG3M","STL","BLK","TOV","MIN"] if c in logs.columns]
    st.dataframe(logs[cols_show], use_container_width=True)


# =============================================================================
# PAGES — FAVORITES (horizontal rows)
# =============================================================================

def _mini_metrics_html(df_table: pd.DataFrame) -> str:
    order = [s for s in BOX_SCORE_ORDER if s in df_table["Stat"].tolist()]
    vals = {r["Stat"]: r["Pred"] for _, r in df_table.iterrows()}
    chips = []
    for s in order:
        v = vals.get(s, None)
        if v is None or (isinstance(v, float) and np.isnan(v)): continue
        chips.append(f'<div class="mm"><div class="t">{s}</div><div class="v">{v:.1f}</div></div>')
    return f'<div class="mm-wrap">{"".join(chips)}</div>'

def page_favorites(players: pd.DataFrame):
    st.header("Favorites")

    if players is None or players.empty:
        st.error("No active players available to add to favorites."); return

    favs = _load_favorites()

    col1, col2 = st.columns([3,1])
    with col1:
        names_list = players["full_name"].astype(str).tolist()
        options = ["— Select Player —"] + names_list
        name = st.selectbox("Add a player", options, key="fav_add_sel", index=0)
    with col2:
        if st.button("➕ Add"):
            if name == "— Select Player —": st.warning("Pick a player first.")
            else:
                row = pick_player_row(players, name)
                if row is None: st.error("Could not resolve the selected player.")
                else:
                    pid = safe_int(row.get("id"), 0)
                    nba_pid = None if pd.isna(row.get("nba_person_id")) else safe_int(row.get("nba_person_id"), None)
                    team_id = safe_int(row.get("team_id"), 0)
                    team_abbr = safe_str(row.get("team_abbr"), "")
                    position = safe_str(row.get("position"), "")
                    if not any(f["id"] == pid for f in favs):
                        favs.append({"id": pid, "full_name": name, "team_id": team_id, "team_abbr": team_abbr, "nba_person_id": nba_pid, "position": position})
                        _save_favorites(favs); st.success("Added to favorites.")
                    else: st.info("Already in favorites.")

    if not favs:
        st.info("No favorites yet."); return

    st.subheader("Saved favorites")
    season = "2025-26"
    dp = load_team_defense_pace(season)

    all_rows = []
    for f in list(favs):
        pid, pname = safe_int(f.get("id"), 0), safe_str(f.get("full_name"), "")
        abbr = safe_str(f.get("team_abbr"), "")
        position = safe_str(f.get("position"), "")
        nba_pid = f.get("nba_person_id")
        nba_pid = None if nba_pid in (None, "", 0) else safe_int(nba_pid, None)
        photo = get_player_photo_bytes(pid, nba_pid)
        color = TEAM_META.get(abbr, {}).get("color", "#60a5fa")
        team_name = TEAM_META.get(abbr, {}).get("name", abbr or "")

        logs = load_logs(pid, season)
        if logs.empty:
            year = dt.date.today().year
            fallback = f"{year-1}-{str(year)[-2:]}"
            logs = load_logs(pid, fallback)
        if logs.empty:
            st.warning(f"No logs for {pname}."); continue
        feats = build_all_features(logs, season)

        # robust team + next
        team_id_resolved, abbr_resolved = resolve_team_for_player(pd.Series(f), logs)
        next_game = auto_next_opponent(team_id_resolved, season, abbr_resolved)

        txt, opp_row = _opp_line(next_game, dp) if next_game else ("Next: N/A", None)
        upcoming_ctx = {}
        if opp_row is not None:
            upcoming_ctx.update({
                "OPP_DEF_PPG": float(opp_row["OPP_DEF_PPG"]), "OPP_DEF_Z": float(opp_row["OPP_DEF_Z"]),
                "OPP_PACE": float(opp_row["PACE"]), "OPP_PACE_Z": float(opp_row["PACE_Z"]),
                "OPP_DEF_X_PACE": float(opp_row["OPP_DEF_Z"] * opp_row["PACE_Z"]),
                "DEF_RTG": float(opp_row["DEF_RTG"]), "DEF_RTG_PCT": float(opp_row["DEF_RTG_PCT"]),
            })

        futures, res = {}, []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for stat in PROP_MAP.keys():
                futures[ex.submit(train_predict_for_stat, pid, season, stat, feats, False, "Single (Lasso)", upcoming_ctx)] = stat
            for fut in as_completed(futures): res.append(fut.result())
        res = order_results(res)
        res = adjust_predictions(res, opp_row, position)
        df_table = results_to_table(res)

        df_all = df_table.copy()
        df_all.insert(0, "Player", pname); df_all.insert(1,"Season",season)
        all_rows.append(df_all)

        with st.container():
            st.markdown('<div class="fav-box">', unsafe_allow_html=True)
            c1, c2 = st.columns([1, 5])
            with c1:
                st.image(_safe_image_from_bytes(photo, (84,84)), use_column_width=False)
            with c2:
                st.markdown(f"<div class='fav-name'>{pname} — {team_name}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='fav-meta'>{txt}</div>", unsafe_allow_html=True)
                st.markdown(_mini_metrics_html(df_table), unsafe_allow_html=True)
                cshare, ccsv, cdel = st.columns([1,1,1])
                with cshare:
                    logo_bytes = get_logo_or_default(abbr)
                    share_bytes = make_share_image_trading_card(
                        player_name=pname, team_abbr=abbr, team_name=team_name, team_color=color,
                        season=season, next_info=txt, photo_bytes=photo, logo_bytes=logo_bytes,
                        df_table=df_table[["Stat","Pred"]], title_suffix="Projections",
                    )
                    st.download_button("📸 Share", data=share_bytes, file_name=f"{pname.replace(' ','_')}_{season}_card.png", mime="image/png", key=f"sh_{pid}")
                with ccsv:
                    csv = df_all.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ CSV", data=csv, file_name=f"{pname.replace(' ','_')}_{season}_projections.csv", mime="text/csv", key=f"csv_{pid}")
                with cdel:
                    st.markdown("<div class='fav-x'>", unsafe_allow_html=True)
                    if st.button("❌", key=f"del_{pid}", help="Remove"):
                        favs = [x for x in favs if safe_int(x.get("id"),0) != pid]; _save_favorites(favs); _rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    if all_rows:
        st.subheader("Export all favorites")
        all_df = pd.concat(all_rows, ignore_index=True)
        st.dataframe(all_df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download ALL as CSV", data=all_df.to_csv(index=False).encode("utf-8"),
                           file_name=f"favorites_{season}_projections.csv", mime="text/csv")


# =============================================================================
# PAGES — RESEARCH
# =============================================================================

def _season_strings(start_year: int = 2000, end_year: Optional[int] = None) -> List[str]:
    if end_year is None: end_year = dt.date.today().year
    return [f"{y}-{str(y+1)[-2:]}" for y in range(start_year, end_year + 1)]

def fetch_logs_multi(player_id: int, seasons: List[str]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        try:
            g = load_logs(player_id, s)
            if not g.empty:
                g = g.copy(); g["SEASON"] = s
                frames.append(g)
        except Exception:
            continue
    if not frames: return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        df = df.sort_values(["SEASON","GAME_DATE"])
    return df

def _window_avg(df: pd.DataFrame, n: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty: return pd.DataFrame(), pd.DataFrame()
    d = df.copy()
    d["GAME_DATE"] = pd.to_datetime(d["GAME_DATE"], errors="coerce")
    d = d.sort_values("GAME_DATE")
    w = d.tail(n).copy()
    if w.empty: return w, pd.DataFrame()
    avg = w[["PTS","REB","AST","FG3M","STL","BLK","TOV","MIN"]].mean().to_frame("Avg").T.round(2)
    return w, avg

def _charts_for_window(df_w: pd.DataFrame, title_prefix: str, color: str):
    if df_w is None or df_w.empty or "GAME_DATE" not in df_w.columns: return
    value_cols = [c for c in ["PTS", "REB", "AST"] if c in df_w.columns]
    if not value_cols: return
    df_w = df_w.copy(); df_w["GAME_DATE"] = pd.to_datetime(df_w["GAME_DATE"], errors="coerce")
    df_w = df_w.dropna(subset=["GAME_DATE"])
    if df_w.empty: return
    base = df_w[["GAME_DATE"] + value_cols]
    if base[value_cols].notna().sum().sum() == 0: return
    long = base.melt(id_vars=["GAME_DATE"], var_name="Stat", value_name="Val").dropna(subset=["Val"])
    if long.empty: return
    c_line = alt.Chart(long).mark_line().encode(x="GAME_DATE:T", y="Val:Q", color=alt.Color("Stat:N"),
                                                tooltip=["GAME_DATE:T","Stat:N","Val:Q"]).properties(height=220, title=f"{title_prefix} — Trends")
    st.altair_chart(c_line, use_container_width=True)
    avg_long = long.groupby("Stat", as_index=False)["Val"].mean()
    if not avg_long.empty:
        c_bar = alt.Chart(avg_long).mark_bar().encode(x=alt.X("Stat:N", sort=["PTS","REB","AST"]), y="Val:Q", tooltip=["Stat","Val"]).properties(height=220, title=f"{title_prefix} — Averages")
        st.altair_chart(c_bar, use_container_width=True)

def make_research_share_image_trading_card(player_name: str, team_abbr: str, team_name: str, team_color: str,
                                           title_suffix: str, photo_bytes: Optional[bytes], logo_bytes: Optional[bytes],
                                           df_metrics: pd.DataFrame) -> bytes:
    return make_share_image_trading_card(player_name, team_abbr, team_name, team_color, title_suffix, "Research view",
                                         photo_bytes, logo_bytes, df_metrics, title_suffix="Research")

def page_research():
    st.header("Research")
    st.caption("Recent performance + rolling windows.")

    all_players = load_player_list_all()
    if all_players.empty:
        st.error("Could not load player index."); return

    names_list = all_players["full_name"].astype(str).tolist()
    options = ["— Select Player —"] + names_list
    name = st.selectbox("Choose player", options, index=0, key="research_pick")
    if name == "— Select Player —": st.info("Pick a player to load research."); return

    row = pick_player_row(all_players, name)
    if row is None: st.info("Select a player to continue."); return

    pid = safe_int(row.get("id"), 0)
    nba_pid = None if pd.isna(row.get("nba_person_id")) else safe_int(row.get("nba_person_id"), None)
    abbr = safe_str(row.get("team_abbr"), "")
    team_meta = TEAM_META.get(abbr, {})
    color = team_meta.get("color", "#60a5fa")
    team_name = team_meta.get("name", abbr or "")
    team_logo = nba_logo_url(abbr)

    colA, colB = st.columns([1,3])
    with colA:
        photo = get_player_photo_bytes(pid, nba_pid)
        st.image(_safe_image_from_bytes(photo, (220, 220)), caption=name)
    with colB:
        if team_logo or team_name:
            cols = st.columns([1,5])
            with cols[0]:
                if team_logo: st.image(team_logo, width=64)
            with cols[1]:
                st.markdown(f"**Team:** {team_name} ({abbr})")
        run = st.button("Load player data")

    if not run:
        st.info("Click **Load player data** to view last game & rolling windows."); return

    ph = st.empty(); show_basketball_loader(ph, "Loading game logs across seasons…")
    seasons_all = [f"{y}-{str(y+1)[-2:]}" for y in range(2000, dt.date.today().year+1)]
    logs = fetch_logs_multi(pid, seasons_all); ph.empty()
    if logs.empty: st.warning("No logs found."); return

    st.subheader("Most Recent Performance (Last Game Played)")
    lg = logs.copy().sort_values("GAME_DATE"); last = lg.tail(1)
    cols_show = [c for c in ["SEASON","GAME_DATE","MATCHUP","PTS","REB","AST","FG3M","STL","BLK","TOV","MIN"] if c in last.columns]
    st.dataframe(last[cols_show], use_container_width=True, hide_index=True)

    def _section(label: str, n: int):
        w, avg = _window_avg(lg, n); st.markdown(f"### {label} Averages")
        if not avg.empty:
            st.dataframe(avg.assign(**{"Games": len(w)}), use_container_width=True, hide_index=True)
            with st.expander("Details"):
                show_cols = [c for c in ["SEASON","GAME_DATE","MATCHUP","PTS","REB","AST","FG3M","STL","BLK","TOV","MIN"] if c in w.columns]
                st.dataframe(w[show_cols], use_container_width=True, hide_index=True)
                _charts_for_window(w, label, color)
        else: st.info("Not enough games for this window.")
    _section("Last 5 Games", 5); _section("Last 10 Games", 10); _section("Last 20 Games", 20)

    w10, avg10 = _window_avg(lg, 10)
    if not avg10.empty:
        table = avg10.T.reset_index().rename(columns={"index":"Stat","Avg":"Pred"})
        table["Stat"] = table["Stat"].map({"PTS":"Points","REB":"Rebounds","AST":"Assists","FG3M":"3PM","STL":"Steals","BLK":"Blocks","TOV":"Turnovers","MIN":"Minutes"})
        table = table.dropna().reset_index(drop=True)
        logo_bytes = get_logo_or_default(abbr)
        share_bytes = make_research_share_image_trading_card(name, abbr, team_name, color, "L10 Averages", photo, logo_bytes, table[["Stat","Pred"]])
        st.download_button("📸 Share image (PNG, mobile)", data=share_bytes, file_name=f"{name.replace(' ','_')}_research_card.png", mime="image/png")


# =============================================================================
# APP
# =============================================================================

def main():
    st.set_page_config(page_title="NBA Prop Predictor — Elite", page_icon="🏀", layout="wide")
    inject_css()

    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio("Go to", ["Predict", "Favorites", "Research"], label_visibility="collapsed")
        st.markdown("---")
        if st.button("🔄 Refresh players"):
            load_player_list.clear(); load_player_list_all.clear()
            _rerun()
        st.markdown('<span class="tag">Projections</span> <span class="tag">Research</span>', unsafe_allow_html=True)

    players = load_player_list("2025-26")

    if page == "Predict":
        page_predict(players)
    elif page == "Favorites":
        page_favorites(players)
    else:
        page_research()


if __name__ == "__main__":
    main()
