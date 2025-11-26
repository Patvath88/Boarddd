# app.py
"""
NBA Prop Predictor — Elite
- Predict • Favorites • Research • Slate
- Opponent-aware (Defense + Pace + positional) projections
- Team-colored metric cards with model & confidence
- Trading-card share image (mobile 1080x1920)
- Analyst-style write-up
"""

from __future__ import annotations

import os
import io
import json
import hashlib
import datetime as dt
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

import numpy as np
import pandas as pd
import requests
import altair as alt
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import streamlit as st

# Local modules (keep your existing ones)
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
STAT_COLUMNS = ["PTS","REB","AST","STL","BLK","TOV","FG3M","MIN"]

BOX_SCORE_ORDER = [
    "Points","Rebounds","Assists","3PM","Steals","Blocks","Turnovers","Minutes",
    "PRA","PR","PA","RA"
]

BASE_COLS = [
    "IS_HOME","REST_DAYS","BACK_TO_BACK",
    "OPP_ALLOW_PTS","OPP_ALLOW_REB","OPP_ALLOW_AST",
    "OPP_DEF_PPG","OPP_DEF_Z","OPP_PACE","OPP_PACE_Z","OPP_DEF_X_PACE",
    "DEF_RTG","DEF_RTG_PCT",
]

POS_WEIGHTS = {
    "G":{"PTS":1.00,"REB":0.85,"AST":1.20,"FG3M":1.20,"STL":1.05,"BLK":0.80,"TOV":1.10,"MIN":1.00},
    "F":{"PTS":1.00,"REB":1.10,"AST":1.00,"FG3M":1.00,"STL":1.00,"BLK":1.00,"TOV":1.00,"MIN":1.00},
    "C":{"PTS":0.98,"REB":1.25,"AST":0.95,"FG3M":0.70,"STL":0.95,"BLK":1.20,"TOV":1.00,"MIN":1.00},
}

N_TRAIN = 60
MIN_ROWS_FOR_MODEL = 12
MAX_WORKERS = max(2, min(8, os.cpu_count() or 4))

DATA_DIR = Path("./data"); DATA_DIR.mkdir(parents=True, exist_ok=True)
FAV_FILE = DATA_DIR / "favorites.json"


# =============================================================================
# TEAM META
# =============================================================================

TEAM_META: Dict[str, Dict[str, str | int]] = {
    "ATL":{"color":"#E03A3E","nba_id":1610612737,"name":"Hawks"},
    "BOS":{"color":"#007A33","nba_id":1610612738,"name":"Celtics"},
    "BKN":{"color":"#000000","nba_id":1610612751,"name":"Nets"},
    "CHA":{"color":"#1D1160","nba_id":1610612766,"name":"Hornets"},
    "CHI":{"color":"#CE1141","nba_id":1610612741,"name":"Bulls"},
    "CLE":{"color":"#860038","nba_id":1610612739,"name":"Cavaliers"},
    "DAL":{"color":"#00538C","nba_id":1610612742,"name":"Mavericks"},
    "DEN":{"color":"#0E2240","nba_id":1610612743,"name":"Nuggets"},
    "DET":{"color":"#C8102E","nba_id":1610612765,"name":"Pistons"},
    "GSW":{"color":"#1D428A","nba_id":1610612744,"name":"Warriors"},
    "HOU":{"color":"#CE1141","nba_id":1610612745,"name":"Rockets"},
    "IND":{"color":"#002D62","nba_id":1610612754,"name":"Pacers"},
    "LAC":{"color":"#C8102E","nba_id":1610612746,"name":"Clippers"},
    "LAL":{"color":"#552583","nba_id":1610612747,"name":"Lakers"},
    "MEM":{"color":"#5D76A9","nba_id":1610612763,"name":"Grizzlies"},
    "MIA":{"color":"#98002E","nba_id":1610612748,"name":"Heat"},
    "MIL":{"color":"#00471B","nba_id":1610612749,"name":"Bucks"},
    "MIN":{"color":"#0C2340","nba_id":1610612750,"name":"Timberwolves"},
    "NOP":{"color":"#0C2340","nba_id":1610612740,"name":"Pelicans"},
    "NYK":{"color":"#006BB6","nba_id":1610612752,"name":"Knicks"},
    "OKC":{"color":"#007AC1","nba_id":1610612760,"name":"Thunder"},
    "ORL":{"color":"#0077C0","nba_id":1610612753,"name":"Magic"},
    "PHI":{"color":"#006BB6","nba_id":1610612755,"name":"76ers"},
    "PHX":{"color":"#1D1160","nba_id":1610612756,"name":"Suns"},
    "POR":{"color":"#E03A3E","nba_id":1610612757,"name":"Trail Blazers"},
    "SAC":{"color":"#5A2D81","nba_id":1610612758,"name":"Kings"},
    "SAS":{"color":"#C4CED4","nba_id":1610612759,"name":"Spurs"},
    "TOR":{"color":"#CE1141","nba_id":1610612761,"name":"Raptors"},
    "UTA":{"color":"#002B5C","nba_id":1610612762,"name":"Jazz"},
    "WAS":{"color":"#002B5C","nba_id":1610612764,"name":"Wizards"},
}
def nba_logo_url(team_abbr: str) -> Optional[str]:
    meta = TEAM_META.get(team_abbr or "")
    if not meta: return None
    return f"https://cdn.nba.com/logos/nba/{meta['nba_id']}/global/L/logo.png"


# =============================================================================
# STYLING
# =============================================================================

def inject_css():
    st.markdown("""
<style>
html, body { font-family: Inter, ui-sans-serif, system-ui; }
.block-container { padding-top: 1.0rem; max-width: 1240px; }
h1, h2, h3, h4 {
  background: linear-gradient(90deg,#e2e8f0 0%, #60a5fa 40%, #34d399 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.card { background: rgba(17,24,39,.6); border: 1px solid rgba(255,255,255,.08);
  border-radius: 16px; padding: 18px; box-shadow: 0 10px 30px rgba(0,0,0,.25); backdrop-filter: blur(6px); }

.metric-grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 14px; }
.metric-card { border-radius: 16px; padding: 16px 14px; color:#fff; border:1px solid rgba(255,255,255,.10); }
.metric-title { font-size:.95rem; opacity:.95; margin-bottom:.35rem; letter-spacing:.25px; }
.metric-value { font-size: 2.1rem; font-weight:800; line-height:1.1; }
.metric-sub { font-size:.85rem; opacity:.95; margin-top:.35rem; }

.mm-wrap { display:flex; flex-wrap:wrap; gap:8px; }
.mm { min-width:86px; padding:10px 12px; border-radius:12px; border:1px solid rgba(255,255,255,.12);
      background: rgba(255,255,255,.04); }
.mm .t { font-size:.78rem; opacity:.9; }
.mm .v { font-size:1.2rem; font-weight:800; }

.fav-box { background: linear-gradient(180deg,#0b1220,#0a1324); border:1px solid rgba(255,255,255,.12);
  border-radius:16px; padding:12px 14px; box-shadow:0 12px 34px rgba(0,0,0,.28); }
.fav-name { font-weight:800; color:#f3f4f6; line-height:1.2; font-size:1.05rem; }
.fav-meta { font-size:.92rem; color:#d1d5db; }

.fav-x > button { background:#111827 !important; color:#ef4444 !important; border:1px solid #ef4444 !important;
  border-radius:10px !important; padding:4px 10px !important; font-weight:800 !important; }

.stButton>button { border-radius:12px; padding:10px 14px; font-weight:600;
  background: linear-gradient(90deg,#0ea5e9,#22c55e); border:none; }
.stButton>button:hover { filter:brightness(1.05); }

.tag { display:inline-block; padding:.2rem .55rem; border-radius:999px; font-size:.75rem;
  background:rgba(99,102,241,.15); border:1px solid rgba(99,102,241,.35); }
.badge { display:inline-block; padding:.25rem .6rem; border-radius:10px; font-size:.8rem;
  background:rgba(34,197,94,.15); border:1px solid rgba(34,197,94,.4); color:#d1fae5; }

.loader-wrap { padding:10px 0 18px 0; }
.court { width:100%; height:14px; border-radius:999px; position:relative;
  background:linear-gradient(90deg, rgba(255,255,255,.08), rgba(255,255,255,.16));
  overflow:hidden; border:1px solid rgba(255,255,255,.12); }
.ball { width:22px; height:22px; border-radius:50%;
  background: radial-gradient(circle at 30% 30%, #ffb86b, #d97706 55%, #92400e 100%);
  position:absolute; top:-4px; left:-22px; box-shadow: inset 0 0 0 2px rgba(0,0,0,.12), 0 6px 14px rgba(0,0,0,.35);
  animation: dribble 1.6s linear infinite; }
@keyframes dribble { 0% {transform: translateX(0)} 50% {transform: translateX(50%)} 100% {transform: translateX(110%)} }
.loader-text { color:#cbd5e1; font-size:.9rem; margin-bottom:.35rem; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# UTILS
# =============================================================================

def _rerun():
    try: st.rerun()
    except Exception: st.experimental_rerun()

def safe_int(v, default=0): 
    try:
        if v is None or (isinstance(v,str) and v.strip()=="") or pd.isna(v): return default
        return int(v)
    except Exception:
        return default

def safe_str(v, default=""): 
    try:
        if v is None or pd.isna(v): return default
        s = str(v); return s if s.strip()!="" else default
    except Exception:
        return default

def clamp(x, lo, hi): return max(lo, min(hi, x))

def show_basketball_loader(placeholder, text):
    placeholder.markdown(f"""
<div class="loader-wrap">
  <div class="loader-text">{text}</div>
  <div class="court"><div class="ball"></div></div>
</div>""", unsafe_allow_html=True)

def _hash_frame_small(X: pd.DataFrame, y: np.ndarray, player_id: int, season: str, stat: str) -> str:
    h = hashlib.sha1()
    h.update(f"{player_id}|{season}|{stat}|{X.shape}|{','.join(map(str,X.columns))}".encode())
    if len(X)>0:
        idx = np.linspace(0,len(X)-1,num=min(128,len(X)),dtype=int)
        h.update(np.nan_to_num(X.iloc[idx].to_numpy(), nan=0.0).tobytes())
        ys = y[idx if len(y)==len(X) else np.clip(idx,0,len(y)-1)]
        h.update(np.nan_to_num(ys, nan=0.0).tobytes())
    return h.hexdigest()

def season_start_year(season: str) -> int: return int(season.split("-")[0])
def prev_season(season: str) -> str: s0 = season_start_year(season); return f"{s0-1}-{str(s0)[-2:]}"


# =============================================================================
# FAVORITES PERSISTENCE
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
    s = requests.Session(); s.headers.update({"User-Agent":"Mozilla/5.0 (PropPredictor)"})
    for pid in ids:
        for tmpl in url_templates:
            try:
                r = s.get(tmpl.format(pid=int(pid)), timeout=5)
                if r.status_code==200 and r.content and len(r.content)>1500: return r.content
            except Exception: continue
    return None

def _safe_image_from_bytes(photo_bytes: Optional[bytes], size=(220,220)) -> Image.Image:
    if not photo_bytes:
        img = Image.new("RGB", size, (15,23,42)); d = ImageDraw.Draw(img)
        d.ellipse([size[0]//4, size[1]//4, 3*size[0]//4, 3*size[1]//4], outline=(80,90,110), width=3)
        return img
    try:
        im = Image.open(io.BytesIO(photo_bytes)).convert("RGB"); return im.resize(size)
    except Exception:
        return Image.new("RGB", size, (15,23,42))

def default_placeholder_bytes(size=(72,72)) -> bytes:
    img = Image.new("RGB", size, (17,26,45))
    d = ImageDraw.Draw(img); d.rectangle([6,6,size[0]-6,size[1]-6], outline=(90,100,120), width=2)
    buf = io.BytesIO(); img.save(buf, format="PNG"); return buf.getvalue()

def get_logo_or_default(team_abbr: str, size=(128,128)) -> bytes:
    url = nba_logo_url(team_abbr)
    if url:
        try:
            r = requests.get(url, timeout=6)
            if r.status_code==200 and r.content: return r.content
        except Exception: pass
    return default_placeholder_bytes(size)

def get_photo_or_logo_bytes(fav: dict) -> bytes:
    pid = safe_int(fav.get("id"),0)
    nba_pid = fav.get("nba_person_id")
    nba_pid = None if nba_pid in (None,"",0,pd.NA) else safe_int(nba_pid,None)
    head = get_player_photo_bytes(pid, nba_pid)
    if head: return head
    abbr = safe_str(fav.get("team_abbr"),"")
    return get_logo_or_default(abbr)


# =============================================================================
# TEAMS / PLAYERS / LOGS
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
    return {} if t.empty else {str(r["abbreviation"]): int(r["id"]) for _, r in t.iterrows() if pd.notna(r["id"])}

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
        return pd.DataFrame(columns=["id","full_name","team_id","team_abbr","nba_person_id","position"])
    p = df.copy()
    for c in ["id","player_id","PLAYER_ID","PersonId","PERSON_ID"]:
        if c in p.columns:
            if c!="id": p = p.rename(columns={c:"id"}); break
    if "full_name" not in p.columns:
        if {"first_name","last_name"}.issubset(p.columns):
            p["full_name"] = (p["first_name"].astype(str).str.strip()+" "+p["last_name"].astype(str).str.strip()).str.strip()
        elif "PLAYER" in p.columns: p["full_name"] = p["PLAYER"].astype(str)
        elif "DISPLAY_FIRST_LAST" in p.columns: p["full_name"] = p["DISPLAY_FIRST_LAST"].astype(str)
        else: p["full_name"] = p.get("full_name","Unknown")
    p = _safe_get_team_cols(p)
    if "team_abbr" not in p.columns or p["team_abbr"].isna().all():
        teams = load_teams_bdl()
        if "team_id" in p.columns and not p["team_id"].isna().all() and not teams.empty:
            p = p.merge(teams.rename(columns={"id":"team_id","abbreviation":"team_abbr"}), on="team_id", how="left")
    for c in ["nba_person_id","PERSON_ID","personId","nba_id"]:
        if c in p.columns:
            if c!="nba_person_id": p = p.rename(columns={c:"nba_person_id"}); break
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
    out=[]; s=requests.Session(); s.headers.update({"User-Agent":"Mozilla/5.0 (PropPredictor)"})
    url="https://www.balldontlie.io/api/v1/players"; page=1
    while True:
        try:
            r=s.get(url, params={"active":"true","per_page":100,"page":page}, timeout=8); r.raise_for_status()
            j=r.json(); data=j.get("data",[]); 
            if not data: break
            out.extend(data); page=j.get("meta",{}).get("next_page") or None
            if not page: break
        except Exception: break
    return pd.DataFrame(out)

@st.cache_data(show_spinner=False)
def load_player_list(season: str="2025-26") -> pd.DataFrame:
    try:
        raw = dfetch.get_active_players_balldontlie(); p = normalize_players_df(raw)
        if not p.empty: return p[["id","full_name","team_id","team_abbr","nba_person_id","position"]]
    except Exception: pass
    try:
        raw2 = bdl_fetch_active_players_direct(); p2 = normalize_players_df(raw2)
        if not p2.empty: return p2[["id","full_name","team_id","team_abbr","nba_person_id","position"]]
    except Exception: pass
    favs = _load_favorites()
    if favs:
        p3 = pd.DataFrame(favs)
        for col in ["team_id","team_abbr","nba_person_id","position"]:
            if col not in p3.columns: p3[col]=None
        p3["team_id"]=p3["team_id"].apply(lambda x:safe_int(x,0))
        p3["team_abbr"]=p3["team_abbr"].apply(lambda x:safe_str(x,""))
        p3["nba_person_id"]=p3["nba_person_id"].apply(lambda x: None if x in (None,"",0) else int(x))
        return p3[["id","full_name","team_id","team_abbr","nba_person_id","position"]].drop_duplicates("id").sort_values("full_name").reset_index(drop=True)
    return pd.DataFrame(columns=["id","full_name","team_id","team_abbr","nba_person_id","position"])

@st.cache_data(show_spinner=False)
def load_player_list_all() -> pd.DataFrame:
    try:
        fb = dfetch.get_player_list_nba(); p = normalize_players_df(fb)
        if not p.empty: return p[["id","full_name","team_id","team_abbr","nba_person_id","position"]]
    except Exception: pass
    try:
        raw2 = requests.get("https://www.balldontlie.io/api/v1/players?per_page=100", timeout=8).json().get("data",[])
        p2 = normalize_players_df(pd.DataFrame(raw2))
        if not p2.empty: return p2[["id","full_name","team_id","team_abbr","nba_person_id","position"]]
    except Exception: pass
    return load_player_list("2025-26")

@st.cache_data(show_spinner=False)
def load_logs(player_id: int, season: str) -> pd.DataFrame:
    return dfetch.get_player_game_logs_nba(player_id, season).copy()


# =============================================================================
# DEFENSE & PACE
# =============================================================================

def _bdl_paginate(url: str, params: Dict) -> List[Dict]:
    out=[]; s=requests.Session(); s.headers.update({"User-Agent":"Mozilla/5.0 (PropPredictor)"})
    page=1
    while True:
        q=params.copy(); q["page"]=page; q.setdefault("per_page",100)
        try:
            r=s.get(url, params=q, timeout=8); r.raise_for_status()
            j=r.json(); out.extend(j.get("data",[]))
            nxt=j.get("meta",{}).get("next_page"); 
            if not nxt: break
            page=nxt
        except Exception: break
    return out

@st.cache_data(show_spinner=False)
def load_team_defense_pace(season: str) -> pd.DataFrame:
    year=season_start_year(season)
    games=_bdl_paginate("https://www.balldontlie.io/api/v1/games", {"seasons[]":year})
    if not games: return pd.DataFrame(columns=["team_id","abbreviation","OPP_DEF_PPG","OPP_DEF_Z","PACE","PACE_Z","DEF_RTG","DEF_RTG_PCT","TIER"])
    rows=[]
    for g in games:
        h=g.get("home_team",{}); v=g.get("visitor_team",{})
        hs=g.get("home_team_score",0); vs=g.get("visitor_team_score",0)
        total=hs+vs
        rows.append({"team_id":h.get("id"),"abbreviation":h.get("abbreviation"),"allowed":vs,"total":total})
        rows.append({"team_id":v.get("id"),"abbreviation":v.get("abbreviation"),"allowed":hs,"total":total})
    df=pd.DataFrame(rows).dropna(subset=["team_id"])
    agg=df.groupby(["team_id","abbreviation"],as_index=False).agg(OPP_DEF_PPG=("allowed","mean"), PACE=("total","mean"))
    mu_d,sd_d=agg["OPP_DEF_PPG"].mean(), agg["OPP_DEF_PPG"].std(ddof=0) or 1.0
    mu_p,sd_p=agg["PACE"].mean(), agg["PACE"].std(ddof=0) or 1.0
    agg["OPP_DEF_Z"]=(agg["OPP_DEF_PPG"]-mu_d)/sd_d
    agg["PACE_Z"]=(agg["PACE"]-mu_p)/sd_p
    agg["DEF_RTG"]=(agg["OPP_DEF_PPG"]/(agg["PACE"].replace(0,np.nan)/100.0)).fillna(agg["OPP_DEF_PPG"])
    agg["DEF_RTG_PCT"]=agg["DEF_RTG"].rank(pct=True)
    def tier(p): 
        if p<=0.2: return "Elite (tough)"
        if p<=0.4: return "Strong"
        if p<=0.6: return "Average"
        if p<=0.8: return "Weak"
        return "Soft (easy)"
    agg["TIER"]=agg["DEF_RTG_PCT"].apply(tier)
    return agg

def resolve_team_for_player(row: pd.Series, logs: Optional[pd.DataFrame]) -> tuple[int,str]:
    tid=safe_int(row.get("team_id"),0); abbr=safe_str(row.get("team_abbr"),"")
    if (tid<=0) and abbr: tid = team_abbr_to_id().get(abbr,0)
    if (tid<=0 or not abbr) and logs is not None and not logs.empty:
        if not abbr and "TEAM_ABBREVIATION" in logs.columns and logs["TEAM_ABBREVIATION"].notna().any():
            abbr=safe_str(logs["TEAM_ABBREVIATION"].dropna().astype(str).iloc[-1],"")
        if (tid<=0) and "TEAM_ID" in logs.columns and logs["TEAM_ID"].notna().any():
            tid=safe_int(logs["TEAM_ID"].dropna().iloc[-1],0)
    return tid, abbr

@st.cache_data(show_spinner=False)
def auto_next_opponent(team_id: int, season: str, team_abbr: Optional[str]=None) -> Optional[Dict[str,Any]]:
    if (not team_id or team_id<=0) and team_abbr:
        team_id = team_abbr_to_id().get(team_abbr,0)
    if not team_id: return None
    year=season_start_year(season); today=dt.date.today()
    s=requests.Session(); s.headers.update({"User-Agent":"Mozilla/5.0 (PropPredictor)"})
    data=[]
    for days in (30,90,180,370):
        try:
            r=s.get("https://www.balldontlie.io/api/v1/games",
                    params={"seasons[]":year,"team_ids[]":team_id,"start_date":today.isoformat(),
                            "end_date":(today+dt.timedelta(days=days)).isoformat(),"per_page":100}, timeout=10)
            j=r.json(); data=j.get("data",[]) or []
            if data: break
        except Exception: data=[]
    if not data:
        try: data=_bdl_paginate("https://www.balldontlie.io/api/v1/games", {"seasons[]":year,"team_ids[]":team_id})
        except Exception: data=[]
    if not data: return None
    def _dt(g): return dt.datetime.fromisoformat(g["date"].replace("Z","+00:00"))
    now=dt.datetime.now(tz=ZoneInfo("UTC"))-dt.timedelta(hours=6)
    future=[g for g in data if _dt(g)>=now]
    if not future: return None
    nxt=sorted(future, key=_dt)[0]
    h,v=nxt.get("home_team",{}),nxt.get("visitor_team",{})
    is_home=(h.get("id")==team_id); opp=v if is_home else h
    dt_utc=_dt(nxt); dt_est=dt_utc.astimezone(ZoneInfo("America/New_York"))
    return {"opp_id":opp.get("id"),"opp_abbr":opp.get("abbreviation"),"is_home":is_home,"dt_utc":dt_utc,"dt_est":dt_est,"date":dt_est.date()}


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def compute_opponent_strength(df: pd.DataFrame) -> pd.DataFrame:
    opp = df.groupby("OPP_TEAM")[["PTS","REB","AST"]].mean().rename(columns={
        "PTS":"OPP_ALLOW_PTS","REB":"OPP_ALLOW_REB","AST":"OPP_ALLOW_AST"})
    return df.join(opp, on="OPP_TEAM")

def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    df=df.copy()
    df["IS_HOME"]=df["MATCHUP"].apply(lambda x: 1 if isinstance(x,str) and ("vs" in x) else 0)
    df["GAME_DATE"]=pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df=df.sort_values("GAME_DATE")
    df["REST_DAYS"]=df["GAME_DATE"].diff().dt.days.fillna(2)
    df["BACK_TO_BACK"]=(df["REST_DAYS"]==1).astype(int)
    return df

def attach_defense_pace(df: pd.DataFrame, defense_pace: pd.DataFrame) -> pd.DataFrame:
    if defense_pace is None or defense_pace.empty:
        for c in ["OPP_DEF_PPG","OPP_DEF_Z","OPP_PACE","OPP_PACE_Z","OPP_DEF_X_PACE","DEF_RTG","DEF_RTG_PCT"]:
            df[c]=np.nan
        return df
    m=defense_pace.rename(columns={"abbreviation":"OPP_TEAM"})
    df = df.merge(m[["OPP_TEAM","OPP_DEF_PPG","OPP_DEF_Z","PACE","PACE_Z","DEF_RTG","DEF_RTG_PCT"]], on="OPP_TEAM", how="left")
    df=df.rename(columns={"PACE":"OPP_PACE","PACE_Z":"OPP_PACE_Z"})
    df["OPP_DEF_X_PACE"]=df["OPP_DEF_Z"]*df["OPP_PACE_Z"]
    return df

def _ensure_training_base(df: pd.DataFrame, season: str) -> pd.DataFrame:
    df=df.copy()
    df["OPP_TEAM"]=df["MATCHUP"].astype(str).str.extract(r"(?:vs\.|@)\s(.+)$")
    df=compute_opponent_strength(df)
    df=add_context_features(df)
    df=df.dropna(subset=["PTS","REB","AST"])
    dp=load_team_defense_pace(season)
    df=attach_defense_pace(df, dp)
    return df.reset_index(drop=True)

@st.cache_data(show_spinner=False)
def build_all_features(df_in: pd.DataFrame, season: str) -> pd.DataFrame:
    df=_ensure_training_base(df_in, season)
    out=df.copy()
    for c in STAT_COLUMNS: out[c]=pd.to_numeric(out[c], errors="coerce").astype(float)
    for c in STAT_COLUMNS:
        s=out[c].to_numpy(dtype=float, copy=False)
        out[f"{c}_L1"]=np.roll(s,1); 
        if len(out.index)>0: out.loc[out.index[0], f"{c}_L1"]=np.nan
        out[f"{c}_L3"]=out[c].shift(3); out[f"{c}_L5"]=out[c].shift(5)
        out[f"{c}_AVG5"]=pd.Series(s).rolling(5, min_periods=1).mean().to_numpy()
        out[f"{c}_AVG10"]=pd.Series(s).rolling(10, min_periods=1).mean().to_numpy()
        # light trend proxy:
        out[f"{c}_TREND"]=pd.Series(s).rolling(5, min_periods=5).apply(lambda x: np.polyfit(range(5), x, 1)[0], raw=True)
    for bc in BASE_COLS:
        if bc in out.columns:
            med=float(out[bc].median()) if out[bc].notna().any() else 0.0
            out[bc]=out[bc].fillna(med)
    return out

def select_X_for_stat(features: pd.DataFrame, stat: str) -> pd.DataFrame:
    cols = PROP_MAP[stat] if isinstance(PROP_MAP[stat], list) else [PROP_MAP[stat]]
    bits=[]
    for c in cols: bits.extend([f"{c}_L1",f"{c}_L3",f"{c}_L5",f"{c}_AVG5",f"{c}_AVG10",f"{c}_TREND"])
    keep=[c for c in set(BASE_COLS)|set(bits) if c in features.columns]
    return features[keep].copy()

def build_target(df: pd.DataFrame, stat: str) -> pd.Series:
    tgt = df[PROP_MAP[stat]].sum(axis=1) if isinstance(PROP_MAP[stat], list) else df[PROP_MAP[stat]]
    return pd.to_numeric(tgt, errors="coerce").astype(float)

def _impute_features(X: pd.DataFrame) -> pd.DataFrame:
    X=X.copy(); X=X.ffill()
    med=X.median(numeric_only=True); X=X.fillna(med).fillna(0.0)
    return X


# =============================================================================
# MODELING + CONFIDENCE
# =============================================================================

def _apply_model_budget(manager: ModelManager, budget: str) -> None:
    if budget=="Full ensemble": return
    if budget=="Lite (3 models)": wanted={"elasticnet","hgb","stack"}
    else: wanted={"elasticnet"}  # Single (Lasso/EN)
    try:
        if hasattr(manager,"set_model_whitelist"): manager.set_model_whitelist(list(wanted)); return
        if hasattr(manager,"available_models") and isinstance(manager.available_models,list):
            manager.available_models=[m for m in manager.available_models if str(m).lower() in wanted]
        elif hasattr(manager,"models") and isinstance(manager.models,list):
            manager.models=[m for m in manager.models if str(m).lower() in wanted]
    except Exception: pass

def confidence_from_errors(mae: float, mse: float, y_std: float) -> int:
    # Normalize by target variability to avoid scale bias, then squash.
    if not (np.isfinite(mae) and np.isfinite(mse) and y_std>0): return 55
    rmse = math.sqrt(max(mse, 0.0))
    score = 1.0 - 0.5*(mae/y_std) - 0.5*(rmse/y_std)
    conf = int(clamp(score, 0.10, 0.97) * 100)
    return conf

def get_or_train_model_cached(player_id: int, season: str, stat: str, X: pd.DataFrame, y: np.ndarray, budget: str) -> ModelManager:
    key = _hash_frame_small(X, y, player_id, season, stat) + f"|{budget}"
    cache = st.session_state.setdefault("model_cache", {})
    if key in cache: return cache[key]
    manager = ModelManager(random_state=42)
    _apply_model_budget(manager, budget)
    manager.train(X, y)
    cache[key] = manager
    return manager

def train_predict_for_stat(
    player_id: int, season: str, stat: str, features: pd.DataFrame,
    fast_mode: bool, model_budget: str, upcoming_ctx: Optional[Dict[str,float]]=None,
) -> Dict[str, Any]:
    y_all = build_target(features, stat).to_numpy()
    X_all = select_X_for_stat(features, stat)
    df_join = pd.concat([pd.Series(y_all, name="TARGET", index=X_all.index), X_all], axis=1)
    df_join = df_join.loc[~df_join["TARGET"].isna()].copy()
    if df_join.empty: return {"Stat": stat, "Prediction": float("nan"), "Model": "N/A", "Confidence": 55}
    y_final = df_join["TARGET"].to_numpy(dtype=float)
    X_final = _impute_features(df_join.drop(columns=["TARGET"]))
    if len(X_final) > N_TRAIN:
        X_final = X_final.iloc[-N_TRAIN:].copy(); y_final = y_final[-N_TRAIN:].copy()
    X_next = X_final.tail(1).copy()
    if upcoming_ctx:
        for k,v in upcoming_ctx.items():
            if k in X_next.columns: X_next.loc[:,k] = v

    if fast_mode or len(X_final) < MIN_ROWS_FOR_MODEL:
        pred = float(np.nanmean(y_final[-10:])) if np.isfinite(y_final[-10:]).any() else float("nan")
        return {"Stat": stat, "Prediction": pred, "Model": "Baseline (10G mean)", "Confidence": 52}

    try:
        manager = get_or_train_model_cached(player_id, season, stat, X_final, y_final, model_budget)
        _ = manager.predict(X_next)
        best = manager.best_model()
        y_std = float(np.nanstd(y_final[-20:]) or np.nanstd(y_final) or 1.0)
        conf = confidence_from_errors(float(getattr(best,"mae",np.nan)), float(getattr(best,"mse",np.nan)), y_std)
        return {"Stat": stat, "Prediction": float(best.prediction), "Model": str(best.name), "Confidence": conf}
    except Exception:
        pred = float(np.nanmean(y_final[-10:])) if np.isfinite(y_final[-10:]).any() else float("nan")
        return {"Stat": stat, "Prediction": pred, "Model": "Baseline (10G mean)", "Confidence": 50}


# =============================================================================
# OPPONENT ADJUSTMENTS
# =============================================================================

def adjust_predictions(results: List[Dict], opp_row: Optional[pd.Series], position: str) -> List[Dict]:
    if opp_row is None or opp_row.empty: 
        return results
    z = float(opp_row.get("OPP_DEF_Z",0.0)); pace_z = float(opp_row.get("PACE_Z",0.0))
    alpha_def = 0.07; beta_pace = 0.03
    pos_key = "G" if "G" in (position or "") else ("C" if "C" in (position or "") else "F")
    w = POS_WEIGHTS.get(pos_key, POS_WEIGHTS["F"])

    base = {r["Stat"]:{ "pred":float(r["Prediction"]), "model":r.get("Model",""), "conf":int(r.get("Confidence",55)) } 
            for r in results if np.isfinite(r.get("Prediction",np.nan))}
    for stat in ["Points","Rebounds","Assists","3PM","Steals","Blocks","Turnovers","Minutes"]:
        if stat in base:
            weight = w.get(PROP_MAP[stat] if isinstance(PROP_MAP[stat], str) else stat, 1.0)
            mult = (1 - alpha_def * z * weight) * (1 + beta_pace * pace_z)
            base[stat]["pred"] = max(0.0, base[stat]["pred"] * mult)
            # small confidence haircut when strong adjustment
            base[stat]["conf"] = int(clamp(base[stat]["conf"] * (1 - 0.05*abs(z)), 30, 99))

    def derive(name, parts):
        if all(p in base for p in parts):
            p = sum(base[p]["pred"] for p in parts)
            c = min(base[p]["conf"] for p in parts)
            base[name] = {"pred": p, "model": "Composite", "conf": c}

    derive("PRA", ["Points","Rebounds","Assists"])
    derive("PR", ["Points","Rebounds"])
    derive("PA", ["Points","Assists"])
    derive("RA", ["Rebounds","Assists"])

    out=[]
    for s in BOX_SCORE_ORDER:
        if s in base:
            out.append({"Stat": s, "Prediction": base[s]["pred"], "Model": base[s]["model"], "Confidence": base[s]["conf"]})
    return out


# =============================================================================
# SHARE IMAGE & UI HELPERS
# =============================================================================

def get_font(size: int):
    paths=[
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for p in paths:
        try: return ImageFont.truetype(p, size=size)
        except Exception: continue
    return ImageFont.load_default()

def lighten_hex(hex_color: str, amount: float=0.35) -> str:
    hex_color=hex_color.lstrip("#")
    try: r=int(hex_color[0:2],16); g=int(hex_color[2:4],16); b=int(hex_color[4:6],16)
    except Exception: r,g,b=(96,165,250)
    r=int(r+(255-r)*amount); g=int(g+(255-g)*amount); b=int(b+(255-b)*amount)
    return f"#{r:02x}{g:02x}{b:02x}"

def results_to_table(results: List[Dict]) -> pd.DataFrame:
    ordered = sorted(results, key=lambda r: (BOX_SCORE_ORDER.index(r["Stat"]) if r["Stat"] in BOX_SCORE_ORDER else 999))
    df = pd.DataFrame(ordered)
    df["Pred"] = pd.to_numeric(df["Prediction"], errors="coerce").round(2)
    if "Confidence" in df.columns: df["Conf%"] = df["Confidence"].astype(int)
    if "Model" in df.columns: df["Model"] = df["Model"].astype(str)
    cols = ["Stat","Pred"] + ([ "Conf%","Model"] if "Confidence" in df.columns else [])
    return df[cols]

def render_metric_cards(results: List[Dict], team_color: str):
    base = team_color or "#60a5fa"; soft = lighten_hex(base, 0.55)
    ordered = sorted(results, key=lambda r: BOX_SCORE_ORDER.index(r["Stat"]) if r["Stat"] in BOX_SCORE_ORDER else 999)
    html=['<div class="metric-grid">']
    for i,r in enumerate(ordered):
        stat=str(r["Stat"])
        pred = f'{float(r["Prediction"]):.2f}' if np.isfinite(r["Prediction"]) else "—"
        model = r.get("Model","")
        conf = r.get("Confidence", None)
        sub = f"{model} • {conf}%" if conf is not None else model
        html.append(f"""
<div class="metric-card" style="background: linear-gradient(135deg, {base} 0%, {soft} 95%);">
  <div class="metric-title">{stat}</div>
  <div class="metric-value">{pred}</div>
  <div class="metric-sub">{sub}</div>
</div>""")
    html.append("</div>")
    st.markdown("\n".join(html), unsafe_allow_html=True)

def make_share_image_trading_card(
    player_name: str, team_abbr: str, team_name: str, team_color: str,
    season: str, next_info: str, photo_bytes: Optional[bytes], logo_bytes: Optional[bytes],
    df_table: pd.DataFrame, title_suffix: str="Projections",
) -> bytes:
    W,H=1080,1920
    col=team_color.lstrip("#") if team_color else "60a5fa"
    r=int(col[0:2],16); g=int(col[2:4],16); b=int(col[4:6],16)

    bg=Image.new("RGB",(W,H),(8,12,22))
    overlay=Image.new("RGB",(W,H),(r,g,b)).filter(ImageFilter.GaussianBlur(210))
    bg=Image.blend(bg, overlay, alpha=0.22)
    draw=ImageDraw.Draw(bg); draw.rectangle([0,0,W,180], fill=(20,26,40))
    draw.line([(0,180),(W,180)], fill=(255,255,255,44), width=1)

    if logo_bytes:
        try:
            logo=Image.open(io.BytesIO(logo_bytes)).convert("RGBA").resize((128,128))
            bg.paste(logo,(48,26),logo)
        except Exception: pass

    head=_safe_image_from_bytes(photo_bytes,(540,540)).convert("RGBA")
    mask=Image.new("L",(540,540),0); ImageDraw.Draw(mask).ellipse((0,0,540,540), fill=255)
    head.putalpha(mask); bg.paste(head,(270,160),head)

    f_title=get_font(72); f_sub=get_font(42); f_small=get_font(36)
    f_metric=get_font(84); f_metric_title=get_font(40)

    card=Image.new("RGBA",(980,1080),(16,22,36,245))
    cd=ImageDraw.Draw(card)
    cd.rounded_rectangle([0,0,980,1080], radius=34, outline=(255,255,255,)
