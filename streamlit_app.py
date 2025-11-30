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

# Rate limiting delays (seconds)
SPORT_DELAY = 1.0      # Between sports
CHUNK_DELAY = 0.5      # Between odds chunks

# ======================================
# 🔐 API KEY MANAGEMENT
# ======================================

def get_api_key():
    """Get API key from secrets.toml first, then environment, then session state."""
    try:
        return st.secrets["opticodds"]["api_key"]
    except:
        pass
    
    env_key = os.getenv("OPTICODDS_API_KEY", "")
    if env_key:
        return env_key
    
    return st.session_state.get("api_key", "")

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
        .status-box {
            border-radius: 8px;
            padding: 8px 12px;
            margin: 4px 0;
            font-size: 0.85rem;
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
        "api_key": get_api_key(),
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
    api_key = st.session_state["api_key"]
    
    if not api_key:
        raise ValueError("❌ No API key configured. Please add your OpticOdds API key.")
    
    params["key"] = api_key
    url = f"{OPTICODDS_BASE}{path}"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_active_fixtures(sport_id: str, league_id: str, days_ahead: int = 2) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    after = now.isoformat().replace("+00:00", "Z")
    before = (now + timedelta(days=days_ahead)).isoformat().replace("+00:00", "Z")

    try:
        resp = optic_get(
            "/fixtures/active",
            {
                "sport": sport_id,
                "league": league_id,
                "start_date_after": after,
                "start_date_before": before,
            },
        )
    except ValueError as e:
        st.error(str(e))
        return pd.DataFrame()
        
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

def fetch_odds_for_fixtures_respectful(
    fixture_ids: List[str],
    markets: List[str],
    sportsbooks: List[str],
    status_text,
) -> pd.DataFrame:
    """
    Fetch odds with chunking and respectful delays between calls.
    Processes one chunk at a time to avoid rate limiting.
    """
    if not fixture_ids:
        return pd.DataFrame()

    rows = []
    total_chunks = (len(fixture_ids) + MAX_FIXTURES_PER_ODDS_CALL - 1) // MAX_FIXTURES_PER_ODDS_CALL
    
    for i in range(0, len(fixture_ids), MAX_FIXTURES_PER_ODDS_CALL):
        chunk = fixture_ids[i : i + MAX_FIXTURES_PER_ODDS_CALL]
        chunk_num = i // MAX_FIXTURES_PER_ODDS_CALL + 1
        
        status_text.text(f"📡 Processing odds chunk {chunk_num}/{total_chunks}...")
        
        try:
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
            
            # Respectful delay between chunks (only if more chunks remain)
            if i + MAX_FIXTURES_PER_ODDS_CALL < len(fixture_ids):
                time.sleep(CHUNK_DELAY)
                
        except Exception as e:
            log_boot(f"Chunk {chunk_num} error: {e}")
            # Continue with next chunk even if one fails

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["price_decimal"] = df["price"].apply(american_to_decimal)
    df["implied_prob"] = df["price"].apply(implied_prob)
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

    df["curr_implied"] = df["price"].apply(implied_prob)
    df["steam"] = df["curr_implied"] - df["open_implied"]

    df["fair_prob"] = np.nan
    df["no_vig_ev"] = np.nan

    group_cols = ["fixture_id", "market_id", "grouping_key"]
    grouped = df.groupby(group_cols, as_index=False)

    updates = []

    for _, g in grouped:
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

    df["minutes_to_start"] = df["start_date"].apply(minutes_until)
    df["efd_score"] = df.apply(compute_efd_score_row, axis=1)

    return df

# ======================================
# 🚀 BOOT SEQUENCE (ONE CALL AT A TIME)
# ======================================

def boot_backend():
    """
    Runs once on startup:
    - loops SPORTS_CONFIG one sport at a time
    - fetches fixtures/active with delay between sports
    - fetches odds in chunks with delays between chunks
    - builds unified board
    - computes EV + EFD
    """
    board_pieces = []
    books = st.session_state["sportsbooks"]

    total_sports = len(SPORTS_CONFIG)
    progress = st.progress(0.0)
    status_text = st.empty()  # Status indicator
    step = 0

    for alias, info in SPORTS_CONFIG.items():
        step += 1
        progress.progress(step / total_sports)
        status_text.text(f"📡 Processing {alias.upper()}... (Sport {step}/{total_sports})")
        
        # === FETCH FIXTURES (One call) ===
        try:
            fixtures_df = fetch_active_fixtures(info["sport"], info["league"], days_ahead=2)
            log_boot(f"[{alias.upper()}] ✓ Fixtures: {len(fixtures_df)} games")
        except Exception as e:
            log_boot(f"[{alias.upper()}] ✗ Fixtures error: {e}")
            time.sleep(SPORT_DELAY)
            continue

        if fixtures_df.empty:
            log_boot(f"[{alias.upper()}] No active fixtures with odds.")
            time.sleep(SPORT_DELAY)
            continue

        fixture_ids = fixtures_df["fixture_id"].tolist()

        # === FETCH ODDS (One chunk at a time) ===
        try:
            odds_df = fetch_odds_for_fixtures_respectful(
                fixture_ids, 
                CORE_MARKETS, 
                books, 
                status_text
            )
            log_boot(f"[{alias.upper()}] ✓ Odds: {len(odds_df)} lines")
        except Exception as e:
            log_boot(f"[{alias.upper()}] ✗ Odds error: {e}")
            time.sleep(SPORT_DELAY)
            continue

        if odds_df.empty:
            log_boot(f"[{alias.upper()}] No odds returned for ML/Spread/Total.")
            time.sleep(SPORT_DELAY)
            continue

        # === MERGE AND STORE ===
        merged = odds_df.merge(
            fixtures_df[
                ["fixture_id", "home_logo", "away_logo", "status"]
            ],
            on="fixture_id",
            how="left",
        )
        merged["alias"] = alias
        board_pieces.append(merged)
        log_boot(f"[{alias.upper()}] → Added {len(merged)} rows to board")
        
        # Respectful delay before next sport
        if step < total_sports:
            status_text.text(f"⏳ Pausing {SPORT_DELAY}s before next sport...")
            time.sleep(SPORT_DELAY)

    status_text.empty()  # Clear status
    progress.empty()

    if not board_pieces:
        st.session_state["board_df"] = pd.DataFrame()
        st.session_state["boot_done"] = True
        log_boot("Boot finished, but no data was returned.")
        return

    board = pd.concat(board_pieces, ignore_index=True)
    log_boot(f"Computing EV/EFD for {len(board)} total lines...")
    board = recompute_ev_and_efd(board)
    st.session_state["board_df"] = board
    st.session_state["boot_done"] = True
    log_boot("✅ Boot sequence complete. Board ready!")

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
            time.sleep(5)

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
# 🧠 COMMAND CONSOLE
# ======================================

def add_message(role: str, text: str):
    st.session_state["messages"].append({"role": role, "text": text})

def handle_command(cmd: str):
    cmd = cmd.strip()
    if not cmd:
        return
    add_message("user", cmd)

    parts = cmd.lower().split()
    if not parts:
        return

    verb = parts[0]
    board = st.session_state["board_df"]

    if verb in ["help", "?"]:
        add_message(
            "system",
            "Commands:\n"
            "- help\n"
            "- show all\n"
            "- show <sport> (nba, nfl, mlb, nhl, ncaab, ncaaf)\n"
            "- show market <ml|spread|total>",
        )
        return

    if board.empty:
        add_message("system", "Board is empty (boot may have failed).")
        return

    if verb == "show" and len(parts) >= 2:
        if parts[1] == "all":
            add_message("system", "Showing all sports (use filters above).")
            return
        if parts[1] == "market" and len(parts) >= 3:
            m = parts[2]
            if m == "ml":
                add_message("system", "Showing moneyline (filter Market = ML).")
            elif m == "spread":
                add_message("system", "Showing spreads.")
            elif m == "total":
                add_message("system", "Showing totals.")
            else:
                add_message("system", f"Unknown market {m}. Use ml/spread/total.")
            return

        sport_alias = parts[1]
        if sport_alias not in SPORTS_CONFIG:
            add_message("system", f"Unknown sport {sport_alias}.")
            return
        add_message("system", f"Showing {sport_alias.upper()} (filter Sport = {sport_alias}).")
        return

    add_message("system", f"Unknown command: {cmd}. Type 'help'.")

def render_console():
    st.subheader("⌨️ Command Console")

    for msg in st.session_state["messages"]:
        cls = "cmd-user" if msg["role"] == "user" else "cmd-system"
        who = "You" if msg["role"] == "user" else "System"
        st.markdown(
            f"<div class='{cls}'><b>{who}:</b> {msg['text']}</div>",
            unsafe_allow_html=True,
        )

    cmd = st.chat_input("Type a command (e.g. 'show nba', 'show market ml', 'help')")
    if cmd is not None:
        handle_command(cmd)

# ======================================
# 🎨 BOARD RENDER
# ======================================

def render_board():
    board = st.session_state["board_df"]
    if board.empty:
        st.error("Board is empty. Check boot log for API errors.")
        return

    # === STREAMING STATUS & CONTROLS ===
    col_status, col_ctrl, col_sport = st.columns([2, 1, 1])
    
    with col_status:
        if st.session_state["sse_running"]:
            st.success("🟢 LIVE – Streaming updates")
            if st.session_state["last_update"]:
                st.caption(f"Last: {st.session_state['last_update'].strftime('%H:%M:%S UTC')}")
        else:
            st.error("🔴 Offline – No live updates")
    
    with col_ctrl:
        if st.session_state["sse_running"]:
            if st.button("⏸️ Pause", use_container_width=True):
                stop_sse_streaming()
                st.rerun()
        else:
            if st.button("▶️ Start Live", use_container_width=True):
                start_sse_streaming(st.session_state["current_sport_filter"])
                st.rerun()
    
    with col_sport:
        sport_filter = st.selectbox(
            "Live Sport",
            list(SPORTS_CONFIG.keys()),
            index=list(SPORTS_CONFIG.keys()).index(st.session_state["current_sport_filter"]),
            key="live_sport_select",
        )
        if sport_filter != st.session_state["current_sport_filter"]:
            stop_sse_streaming()
            start_sse_streaming(sport_filter)
            st.rerun()
    
    st.markdown("---")

    # === FILTERS ===
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sport_filter = st.selectbox(
            "Sport",
            ["all"] + list(SPORTS_CONFIG.keys()),
            index=0,
        )
    with col2:
        market_filter = st.selectbox(
            "Market",
            ["all", "moneyline", "spread", "total"],
            index=0,
        )
    with col3:
        book_filter = st.selectbox(
            "Sportsbook",
            ["all"] + sorted(board["sportsbook"].dropna().unique().tolist()),
            index=0,
        )
    with col4:
        sort_mode = st.selectbox(
            "Sort by",
            ["EFD score", "EV", "Steam"],
            index=0,
        )

    # Process any pending SSE updates on every rerun
    if st.session_state["sse_running"]:
        process_pending_updates()

    # Apply filters
    df = board.copy()
    if sport_filter != "all":
        df = df[df["alias"] == sport_filter]
    if market_filter != "all":
        if market_filter == "moneyline":
            df = df[df["market_id"].str.contains("moneyline", case=False, na=False)]
        elif market_filter == "spread":
            df = df[df["market_id"].str.contains("spread", case=False, na=False)]
        elif market_filter == "total":
            df = df[df["market_id"].str.contains("total", case=False, na=False)]
    if book_filter != "all":
        df = df[df["sportsbook"] == book_filter]
    if df.empty:
        st.warning("No rows match the current filters.")
        return

    # Sorting
    if sort_mode == "EV":
        df = df.sort_values("no_vig_ev", ascending=False)
    elif sort_mode == "Steam":
        df = df.sort_values("steam", ascending=False)
    else:
        df = df.sort_values("efd_score", ascending=False)

    # --- Cards (top fixtures) ---
    st.subheader("🎮 Matchup Cards")
    by_fixture = (
        df.groupby("fixture_id")
        .agg(
            {
                "start_date": "first",
                "home_name": "first",
                "away_name": "first",
                "home_logo": "first",
                "away_logo": "first",
            }
        )
        .reset_index()
    ).head(8)

    for _, row in by_fixture.iterrows():
        fid = row["fixture_id"]
        subset = df[df["fixture_id"] == fid]

        colA, colB = st.columns([2, 3])

        with colA:
            st.markdown('<div class="fixture-card">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="fixture-header">{row["start_date"]}</div>',
                unsafe_allow_html=True,
            )
            logo_cols = st.columns(2)
            with logo_cols[0]:
                if isinstance(row["home_logo"], str):
                    st.image(row["home_logo"], width=48)
                st.markdown(
                    f'<span class="team-name">{row["home_name"]}</span>',
                    unsafe_allow_html=True,
                )
            with logo_cols[1]:
                if isinstance(row["away_logo"], str):
                    st.image(row["away_logo"], width=48)
                st.markdown(
                    f'<span class="team-name">{row["away_name"]}</span>',
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        with colB:
            st.write("Top prices (Moneyline / Spread / Total):")

            ml = subset[subset["market_id"].str.contains("moneyline", case=False, na=False)]
            sp = subset[subset["market_id"].str.contains("spread", case=False, na=False)]
            tot = subset[subset["market_id"].str.contains("total", case=False, na=False)]

            def summarize(market_df: pd.DataFrame, label: str):
                if market_df.empty:
                    return f"**{label}:** —"
                best = market_df.sort_values("price_decimal", ascending=False).iloc[0]
                return f"**{label}:** {best['sportsbook']} – {best['name']} ({best['price']}, EFD {best['efd_score']:.1f})"

            st.markdown(
                "<div class='odds-row'>"
                + summarize(ml, "Moneyline")
                + "<br>"
                + summarize(sp, "Spread")
                + "<br>"
                + summarize(tot, "Total")
                + "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

    # --- Full board table ---
    st.subheader("📊 Full Board (top 250 rows)")
    show_cols = [
        "alias",
        "start_date",
        "home_name",
        "away_name",
        "sportsbook",
        "market_label",
        "name",
        "price",
        "no_vig_ev",
        "steam",
        "efd_score",
    ]
    tmp = df[show_cols].copy()
    tmp["EV%"] = tmp["no_vig_ev"].apply(lambda v: f"{(v or 0)*100:.1f}%")
    tmp["Steam%"] = tmp["steam"].apply(lambda v: f"{(v or 0)*100:.1f}%")

    tmp = tmp.rename(
        columns={
            "alias": "Sport",
            "start_date": "Start",
            "home_name": "Home",
            "away_name": "Away",
            "sportsbook": "Book",
            "market_label": "Market",
            "name": "Bet",
            "price": "Odds",
            "efd_score": "EFD",
        }
    )

    st.dataframe(
        tmp[
            [
                "Sport",
                "Start",
                "Home",
                "Away",
                "Book",
                "Market",
                "Bet",
                "Odds",
                "EV%",
                "Steam%",
                "EFD",
            ]
        ].head(250),
        use_container_width=True,
    )

# ======================================
# 🧵 MAIN
# ======================================

def main():
    init_state()
    inject_theme()
    
    st.title("⚡ Edge Force Dominion – Global Odds Engine")

    # Auto-refresh every 3 seconds when app is running
    if st.session_state["boot_done"]:
        st_autorefresh(interval=3000, key="oddsrefresher")

    # Check for API key and show setup instructions if missing
    if not st.session_state["api_key"]:
        st.warning("⚠️ No API key found!")
        with st.expander("🔑 How to add your OpticOdds API key", expanded=True):
            st.markdown("""
            ### For Local Development:
            1. Create a `.env` file in your project root:
            ```
            OPTICODDS_API_KEY=your_key_here
            ```
            2. Or enter it directly in the sidebar

            ### For Streamlit Cloud:
            1. Go to your app dashboard
            2. Click "Settings"
            3. Add secret: `opticodds.api_key` = your_key_here
            4. Click "Save"
            5. Redeploy your app
            """)
        return  # Don't proceed without API key

    # Sidebar: API key & sportsbooks
    with st.sidebar:
        st.markdown("### 🔐 OpticOdds API Key")
        key_input = st.text_input(
            "API key",
            value=st.session_state["api_key"],
            type="password",
            help="Enter your OpticOdds API key",
        )
        if key_input:
            st.session_state["api_key"] = key_input

        st.markdown("### 📚 Sportsbooks")
        current_books = st.session_state["sportsbooks"]
        books_text = st.text_input(
            "Comma separated sportsbooks",
            value=",".join(current_books),
        )
        st.session_state["sportsbooks"] = [
            b.strip() for b in books_text.split(",") if b.strip()
        ]

        st.markdown("---")
        st.markdown(
            "**How it works:**\n"
            "1. App boots and loads all static data\n"
            "2. SSE streaming auto-starts for default sport\n"
            "3. Updates every 3 seconds automatically\n"
            "4. Use Pause/Resume to control streaming"
        )

    # Boot sequence
    if not st.session_state["boot_done"]:
        st.subheader("🧩 Booting Edge Force Dominion Board")
        with st.spinner("Pulling fixtures + odds for all sports..."):
            boot_backend()

        st.markdown("#### Boot log")
        for line in st.session_state["boot_log"]:
            st.markdown(f"- {line}")
        st.stop()

    # Post-boot: Auto-start SSE streaming
    if st.session_state["boot_done"] and not st.session_state["sse_running"]:
        start_sse_streaming(st.session_state["current_sport_filter"])
        st.rerun()

    # Quick view of boot log
    with st.expander("View boot log", expanded=False):
        for line in st.session_state["boot_log"]:
            st.markdown(f"- {line}")

    # Board
    render_board()
    st.markdown("---")
    render_console()

if __name__ == "__main__":
    main()
