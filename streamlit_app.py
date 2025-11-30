Here's the complete, production-ready code with persistent SSE streaming that auto-starts after boot:
import os
import json
import threading
import queue
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import sseclient
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ======================================
# 🔑 CONFIG
# ======================================

OPTICODDS_BASE = "https://api.opticodds.com/api/v3"

SPORTS_CONFIG = {
    "nba":   {"sport": "basketball",    "league": "nba"},
    "ncaab": {"sport": "basketball",    "league": "ncaab"},
    "nfl":   {"sport": "football",      "league": "nfl"},
    "ncaaf": {"sport": "football",      "league": "ncaaf"},
    "mlb":   {"sport": "baseball",      "league": "mlb"},
    "nhl":   {"sport": "ice_hockey",    "league": "nhl"},
}

CORE_MARKETS = ["moneyline", "spread", "total_points"]
DEFAULT_SPORTSBOOKS = ["DraftKings", "FanDuel", "Caesars", "BetMGM"]
MAX_FIXTURES_PER_ODDS_CALL = 5

# ======================================
# 🎨 THEME
# ======================================

def inject_theme():
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top left, #020617 0, #000000 40%, #020617 100%);
            color: #e2f3ff;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1.5rem;
        }
        h1, h2, h3 {
            color: #e5f0ff;
            text-shadow: 0 0 12px rgba(96,165,250,0.85);
        }
        .fixture-card {
            border-radius: 16px;
            padding: 12px 14px;
            background: linear-gradient(135deg, rgba(15,118,110,0.15), rgba(59,130,246,0.1));
            border: 1px solid rgba(148,163,184,0.5);
            box-shadow: 0 0 18px rgba(56,189,248,0.25);
        }
        .fixture-header {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #a5b4fc;
            margin-bottom: 4px;
        }
        .team-name {
            font-weight: 600;
        }
        .odds-row {
            font-size: 0.85rem;
            color: #e2e8f0;
        }
        .cmd-user {
            color: #93c5fd;
        }
        .cmd-system {
            color: #a5b4fc;
        }
        .status-indicator {
            padding: 8px 12px;
            border-radius: 8px;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ======================================
# 🧮 MATH HELPERS
# ======================================

def american_to_decimal(odds: float) -> float:
    try:
        odds = float(odds)
    except Exception:
        return 1.0
    if odds > 0:
        return 1.0 + odds / 100.0
    elif odds < 0:
        return 1.0 + 100.0 / abs(odds)
    return 1.0

def implied_prob(odds: float) -> float:
    dec = american_to_decimal(odds)
    return 1.0 / dec if dec > 1.0 else 0.0

def minutes_until(start_iso: Optional[str]) -> Optional[float]:
    if not start_iso:
        return None
    try:
        dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return (dt - now).total_seconds() / 60.0
    except Exception:
        return None

# ======================================
# 🔐 SESSION STATE
# ======================================

def init_state():
    defaults = {
        "boot_done": False,
        "boot_log": [],
        "board_df": pd.DataFrame(),
        "messages": [],
        "sportsbooks": DEFAULT_SPORTSBOOKS.copy(),
        "api_key": os.getenv("OPTICODDS_API_KEY", ""),
        "sse_queue": queue.Queue(),
        "sse_running": False,
        "sse_thread": None,
        "last_update": None,
        "current_sport_filter": "nba",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def log_boot(msg: str):
    st.session_state["boot_log"].append(msg)

# ======================================
# 🌐 OPTICODDS HELPERS
# ======================================

def optic_get(path: str, params: Dict) -> Dict:
    params = dict(params)
    params["key"] = st.session_state["api_key"]
    url = f"{OPTICODDS_BASE}{path}"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_active_fixtures(sport_id: str, league_id: str, days_ahead: int = 2) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    after = now.isoformat().replace("+00:00", "Z")
    before = (now + timedelta(days=days_ahead)).isoformat().replace("+00:00", "Z")

    resp = optic_get(
        "/fixtures/active",
        {
            "sport": sport_id,
            "league": league_id,
            "start_date_after": after,
            "start_date_before": before,
        },
    )
    data = resp.get("data", [])
    rows = []

    for fx in data:
        try:
            competitors = fx.get("home_competitors") or []
            away_competitors = fx.get("away_competitors") or []
            rows.append(
                {
                    "fixture_id": fx["id"],
                    "sport": fx["sport"]["id"],
                    "league": fx["league"]["id"],
                    "start_date": fx.get("start_date"),
                    "home_name": fx.get("home_team_display"),
                    "away_name": fx.get("away_team_display"),
                    "home_logo": competitors[0].get("logo") if competitors else None,
                    "away_logo": away_competitors[0].get("logo") if away_competitors else None,
                    "status": fx.get("status"),
                }
            )
        except Exception as e:
            log_boot(f"Error processing fixture {fx.get('id')}: {e}")
            continue

    return pd.DataFrame(rows)

def fetch_odds_for_fixtures(
    fixture_ids: List[str],
    markets: List[str],
    sportsbooks: List[str],
) -> pd.DataFrame:
    if not fixture_ids:
        return pd.DataFrame()

    rows = []

    for i in range(0, len(fixture_ids), MAX_FIXTURES_PER_ODDS_CALL):
        chunk = fixture_ids[i : i + MAX_FIXTURES_PER_ODDS_CALL]
        resp = optic_get(
            "/fixtures/odds",
            {
                "fixture_id": chunk,
                "sportsbook": sportsbooks,
                "market": markets,
                "odds_format": "AMERICAN",
            },
        )
        data = resp.get("data", [])

        for fx in data:
            fid = fx["id"]
            start_date = fx.get("start_date")
            home_name = fx.get("home_team_display")
            away_name = fx.get("away_team_display")
            sport = fx.get("sport", {}).get("id")
            league = fx.get("league", {}).get("id")

            for od in fx.get("odds", []):
                rows.append(
                    {
                        "fixture_id": fid,
                        "sport": sport,
                        "league": league,
                        "start_date": start_date,
                        "home_name": home_name,
                        "away_name": away_name,
                        "sportsbook": od.get("sportsbook"),
                        "market_id": od.get("market_id") or od.get("market"),
                        "market_label": od.get("market"),
                        "selection": od.get("selection"),
                        "name": od.get("name"),
                        "price": od.get("price"),
                        "grouping_key": od.get("grouping_key"),
                        "points": od.get("points"),
                    }
                )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["price_decimal"] = df["price"].apply(american_to_decimal)
    df["implied_prob"] = df["price"].apply(implied_prob)
    
    # Store opening odds snapshot
    df["open_price"] = df["price"]
    df["open_implied"] = df["implied_prob"]
    
    return df

# ======================================
# 🧠 EFD SCORING + METRICS
# ======================================

def compute_efd_score_row(row) -> float:
    ev = float(row.get("no_vig_ev") or 0.0)
    open_imp = float(row.get("open_implied") or 0.0)
    curr_imp = float(row.get("curr_implied") or 0.0)
    steam = curr_imp - open_imp
    steam_abs = abs(steam)
    mins = row.get("minutes_to_start")
    
    try:
        mins = float(mins) if mins is not None else None
    except Exception:
        mins = None

    market_id = (row.get("market_id") or "").lower()

    # Math edge
    ev_cap = min(max(ev, -0.10), 0.12)
    math_component = (ev_cap + 0.10) / 0.22

    # Steam
    steam_cap = min(steam_abs, 0.10)
    steam_component = steam_cap / 0.10

    # Timing
    if mins is None:
        timing_component = 0.5
    else:
        if mins <= 0:
            timing_component = 1.0
        elif mins <= 360:
            timing_component = 1.0 - (mins / 360.0) * 0.5
        else:
            timing_component = 0.5

    # Market bias
    if "moneyline" in market_id:
        market_component = 1.0
    elif "spread" in market_id:
        market_component = 0.9
    else:
        market_component = 0.85

    base = (
        0.45 * math_component +
        0.35 * steam_component +
        0.20 * timing_component
    )

    score = base * 100.0 * market_component
    return float(round(score, 1))

def recompute_ev_and_efd(board_df: pd.DataFrame) -> pd.DataFrame:
    df = board_df.copy()
    if df.empty:
        return df

    # Current implied from latest price
    df["curr_implied"] = df["price"].apply(implied_prob)

    # Steam
    df["steam"] = df["curr_implied"] - df["open_implied"]

    # No-vig EV
    df["fair_prob"] = np.nan
    df["no_vig_ev"] = np.nan

    group_cols = ["fixture_id", "market_id", "grouping_key"]
    grouped = df.groupby(group_cols, as_index=False)

    updates = []

    for _, g in grouped:
        # Best prices per selection
        best = (
            g.sort_values("price_decimal", ascending=False)
             .groupby("selection", as_index=False)
             .first()
        )
        if len(best) < 2:
            continue

        inv_sum = (1.0 / best["price_decimal"]).sum()
        if inv_sum <= 0:
            continue

        best["fair_prob"] = (1.0 / best["price_decimal"]) / inv_sum
        best["no_vig_ev"] = best["price_decimal"] * best["fair_prob"] - 1.0
        updates.append(best)

    if updates:
        upd = pd.concat(updates, ignore_index=True)
        join_cols = ["fixture_id", "market_id", "grouping_key", "selection", "sportsbook"]
        df = df.merge(
            upd[join_cols + ["fair_prob", "no_vig_ev"]],
            on=join_cols,
            how="left",
            suffixes=("", "_upd"),
        )
        for col in ["fair_prob", "no_vig_ev"]:
            df[col] = df[col + "_upd"].combine_first(df[col])
            df.drop(columns=[col + "_upd"], inplace=True)

    # Minutes to start
    df["minutes_to_start"] = df["start_date"].apply(minutes_until)

    # EFD score
    df["efd_score"] = df.apply(compute_efd_score_row, axis=1)

    return df

# ======================================
# 🚀 BOOT SEQUENCE
# ======================================

def boot_backend():
    board_pieces = []
    books = st.session_state["sportsbooks"]

    total_sports = len(SPORTS_CONFIG)
    progress = st.progress(0.0)
    step = 0

    for alias, info in SPORTS_CONFIG.items():
        step += 1
        progress.progress(step / total_sports)
        log_boot(f"[{alias.upper()}] Fetching fixtures/active...")

        try:
            fixtures_df = fetch_active_fixtures(info["sport"], info["league"], days_ahead=2)
        except Exception as e:
            log_boot(f"[{alias.upper()}] Fixtures error: {e}")
            continue

        if fixtures_df.empty:
            log_boot(f"[{alias.upper()}] No active fixtures with odds.")
            continue

        fixture_ids = fixtures_df["fixture_id"].tolist()
        log_boot(f"[{alias.upper()}] Found {len(fixtures_df)} active fixtures.")

        try:
            odds_df = fetch_odds_for_fixtures(fixture_ids, CORE_MARKETS, books)
        except Exception as e:
            log_boot(f"[{alias.upper()}] Odds error: {e}")
            continue

        if odds_df.empty:
            log_boot(f"[{alias.upper()}] No odds returned for ML/Spread/Total.")
            continue

        merged = odds_df.merge(
            fixtures_df[
                ["fixture_id", "home_logo", "away_logo", "status"]
            ],
            on="fixture_id",
            how="left",
        )
        merged["alias"] = alias
        board_pieces.append(merged)
        log_boot(f"[{alias.upper()}] Added {len(merged)} board rows.")

    if not board_pieces:
        st.session_state["board_df"] = pd.DataFrame()
        st.session_state["boot_done"] = True
        log_boot("Boot finished, but no data was returned.")
        return

    board = pd.concat(board_pieces, ignore_index=True)
    log_boot("Computing fair probabilities, EV, and EFD scores...")
    board = recompute_ev_and_efd(board)
    st.session_state["board_df"] = board
    st.session_state["boot_done"] = True
    log_boot("Boot sequence complete. Board ready.")

# ======================================
# 🛰️ PERSISTENT SSE STREAMING
# ======================================

def sse_listener(sport_id: str, league_id: str, sportsbooks: List[str], markets: List[str]):
    """Background thread: continuously listens to SSE and puts events into queue."""
    while st.session_state["sse_running"]:
        try:
            params = {
                "league": league_id,
                "sportsbook": ",".join(sportsbooks),
                "market": ",".join(markets),
                "odds_format": "AMERICAN",
                "key": st.session_state["api_key"],
            }
            url = f"{OPTICODDS_BASE}/stream/odds/{sport_id}"
            
            with requests.get(
                url, 
                params=params, 
                headers={"Accept": "text/event-stream"}, 
                stream=True, 
                timeout=30
            ) as resp:
                resp.raise_for_status()
                client = sseclient.SSEClient(resp)
                
                for event in client.events():
                    if not st.session_state["sse_running"]:
                        return
                    
                    if event.data:
                        try:
                            data = json.loads(event.data)
                            st.session_state["sse_queue"].put(data)
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            st.session_state["sse_queue"].put({"type": "error", "message": str(e)})
            time.sleep(5)  # Backoff before retry

def start_sse_streaming(sport_alias: str):
    """Start SSE listener for a specific sport."""
    if st.session_state["sse_running"]:
        return
    
    info = SPORTS_CONFIG.get(sport_alias)
    if not info:
        return
    
    st.session_state["sse_running"] = True
    st.session_state["current_sport_filter"] = sport_alias
    
    thread = threading.Thread(
        target=sse_listener,
        args=(info["sport"], info["league"], st.session_state["sportsbooks"], CORE_MARKETS),
        daemon=True,
    )
    thread.start()
    st.session_state["sse_thread"] = thread

def stop_sse_streaming():
    """Stop SSE listener."""
    st.session_state["sse_running"] = False
    st.session_state["sse_thread"] = None

def process_pending_updates() -> bool:
    """Process any SSE updates in the queue. Returns True if updates were applied."""
    if st.session_state["board_df"].empty:
        return False
    
    updates = []
    while True:
        try:
            item = st.session_state["sse_queue"].get_nowait()
            if isinstance(item, dict) and item.get("type") == "error":
                st.warning(f"SSE error: {item.get('message')}", icon="⚠️")
            else:
                updates.append(item)
        except queue.Empty:
            break
    
    if updates:
        df = st.session_state["board_df"]
        df = apply_stream_events(df, updates)
        st.session_state["board_df"] = df
        st.session_state["last_update"] = datetime.now(timezone.utc)
        return True
    
    return False

# ======================================
# 🧵 LIVE STREAM (SINGLE CHUNK)
# ======================================

def stream_live_odds_chunk(
    sport_id: str,
    league_id: str,
    sportsbooks: List[str],
    markets: List[str],
    max_events: int = 80,
    timeout: int = 10,
) -> List[dict]:
    """Legacy function kept for fallback, but not used by persistent SSE."""
    params = {
        "league": league_id,
        "sportsbook": ",".join(sportsbooks),
        "market": ",".join(markets),
        "odds_format": "AMERICAN",
        "key": st.session_state["api_key"],
    }
    url = f"{OPTICODDS_BASE}/stream/odds/{sport_id}"

    headers = {"Accept": "text/event-stream"}

    with requests.get(url, params=params, headers=headers, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        client = sseclient.SSEClient(resp)
        events = []
        count = 0
        for event in client.events():
            if event.data:
                try:
                    data = json.loads(event.data)
                    events.append(data)
                except json.JSONDecodeError:
                    pass
            count += 1
            if count >= max_events:
                break
        return events

def apply_stream_events(board_df: pd.DataFrame, events: List[dict]) -> pd.DataFrame:
    """Map SSE events to board price updates, then recompute EV+EFD."""
    df = board_df.copy()
    if df.empty or not events:
        return df

    update_count = 0
    for ev in events:
        fixture_id = ev.get("fixture_id") or ev.get("fixture", {}).get("id")
        odds_list = ev.get("odds", [])


