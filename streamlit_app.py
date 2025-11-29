import time
import json
import itertools
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import requests
import pandas as pd
import numpy as np
import streamlit as st

# ==============================
#  CONFIG & CONSTANTS
# ==============================

OPTIC_BASE = "https://api.opticodds.com/api/v3"

# Map UI names to API keys
SPORT_MAP = {
    "NBA": "nba",
    "NFL": "nfl",
    "NHL": "nhl",
    "NCAAB": "ncaab",
    "NCAAF": "ncaaf",
    "MLB": "mlb",
}

DEFAULT_BOOKS = ["FanDuel", "DraftKings", "BetMGM", "Caesars", "Pinnacle", "LowVig"]
DEFAULT_MARKETS = ["moneyline", "point_spread", "total_points"]

# ==============================
#  UTILS
# ==============================

def _chunk_list(lst, size):
    """Yield successive chunks of length `size` from list."""
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

def american_to_decimal(odds):
    """Safe American -> Decimal conversion."""
    try:
        o = float(odds)
    except Exception:
        return 1.0
    if o == 0: return 1.0
    if o > 0:
        return 1.0 + (o / 100.0)
    return 1.0 + (100.0 / abs(o))

def now_utc_iso():
    return datetime.now(timezone.utc).isoformat()

# ==============================
#  OPTICODDS API: SNAPSHOT
# ==============================

def fetch_fixtures_for_sport(api_key: str, sport_key: str, max_events: int = 50):
    """Get list of upcoming fixture IDs."""
    params = {
        "key": api_key,
        "sport": sport_key,
        "event_status": "upcoming",
        "page_size": max_events,
    }
    try:
        r = requests.get(f"{OPTIC_BASE}/fixtures", params=params, timeout=10)
        r.raise_for_status()
    except Exception as e:
        st.error(f"❌ {sport_key.upper()}: Failed to pull fixtures. {e}")
        return []

    data = r.json().get("data", [])
    # Return unique IDs
    return list({f.get("id") for f in data if f.get("id")})

def fetch_odds_for_fixtures(api_key, fixture_ids, sportsbooks, markets):
    """
    Batch fetch odds for specific fixtures.
    Chunks requests to avoid 414 URI Too Long or Rate Limits.
    """
    if not fixture_ids or not sportsbooks or not markets:
        return []

    all_rows = []
    # Deduplicate inputs
    fixture_ids = list(dict.fromkeys(fixture_ids))
    sportsbooks = list(dict.fromkeys(sportsbooks))

    # Progress bar for long fetches
    prog_bar = st.progress(0)
    total_chunks = (len(fixture_ids) // 5) + 1
    chunk_idx = 0

    # OpticOdds Limit: Max 5 fixtures per call recommended for odds endpoint
    for fixture_chunk in _chunk_list(fixture_ids, 5):
        # OpticOdds Limit: Max 5 sportsbooks per call recommended
        for book_chunk in _chunk_list(sportsbooks, 5):
            
            params = [("key", api_key)]
            for fid in fixture_chunk: params.append(("fixture_id", fid))
            for sb in book_chunk: params.append(("sportsbook", sb))
            for m in markets: params.append(("market", m))
            params.append(("is_main", "True"))

            try:
                r = requests.get(f"{OPTIC_BASE}/fixtures/odds", params=params, timeout=10)
                r.raise_for_status()
                payload = r.json().get("data", [])
            except Exception as e:
                # Log error but keep going
                print(f"Error fetching chunk: {e}")
                continue

            for f in payload:
                # Common fixture data
                base_data = {
                    "fixture_id": f.get("id"),
                    "sport": f.get("sport"),
                    "league": f.get("league"),
                    "home_team": f.get("home_team"),
                    "away_team": f.get("away_team"),
                    "start_time": f.get("start_time"),
                    "source": "snapshot",
                    "timestamp": now_utc_iso()
                }

                for o in f.get("odds", []):
                    # Flatten the odds object
                    row = base_data.copy()
                    row.update({
                        "sportsbook": o.get("sportsbook"),
                        "market": o.get("market"),
                        "selection": o.get("selection"),
                        "price_american": o.get("price_american"),
                        "price_decimal": o.get("price_decimal") or american_to_decimal(o.get("price_american")),
                    })
                    all_rows.append(row)
            
            # RATE LIMIT PROTECTION: Sleep slightly between chunks
            time.sleep(0.15)
        
        chunk_idx += 1
        prog_bar.progress(min(chunk_idx / total_chunks, 1.0))
    
    prog_bar.empty()
    return all_rows

# ==============================
#  OPTICODDS API: STREAM
# ==============================

def parse_sse_stream(resp, max_messages=200):
    """Parses SSE 'data:' lines into JSON objects."""
    message_buf = []
    count = 0

    for raw_line in resp.iter_lines(decode_unicode=True):
        if raw_line is None: continue
        line = raw_line.strip()
        
        if not line:
            # End of event
            if not message_buf: continue
            
            data_lines = [l[5:].strip() for l in message_buf if l.startswith("data:")]
            if data_lines:
                try:
                    payload = json.loads("\n".join(data_lines))
                    yield payload
                    count += 1
                    if count >= max_messages: return
                except Exception:
                    pass # Skip malformed packets
            message_buf = []
        else:
            message_buf.append(line)

def stream_burst_for_sport(api_key, sport_key, sportsbooks, markets, is_live, max_events, max_messages=200):
    """Connects to SSE stream, gathers a burst of updates, then disconnects."""
    params = [("key", api_key)]
    for sb in sportsbooks: params.append(("sportsbook", sb))
    for m in markets: params.append(("market", m))
    params.append(("is_main", "True"))

    if is_live is True: params.append(("event_status", "live"))
    elif is_live is False: params.append(("event_status", "upcoming"))

    url = f"{OPTIC_BASE}/stream/odds/{sport_key}"
    rows = []

    try:
        with requests.get(url, params=params, stream=True, timeout=25) as resp:
            resp.raise_for_status()
            for msg in parse_sse_stream(resp, max_messages=max_messages):
                # Handle inconsistent wrapping (sometimes 'data', sometimes 'fixtures')
                data_items = msg.get("data") or msg.get("fixtures") or []
                if not isinstance(data_items, list): data_items = [data_items]

                for f in data_items:
                    base_data = {
                        "fixture_id": f.get("id") or f.get("fixture_id"),
                        "sport": f.get("sport"),
                        "league": f.get("league"),
                        "home_team": f.get("home_team"),
                        "away_team": f.get("away_team"),
                        "start_time": f.get("start_time"),
                        "source": "stream",
                        "timestamp": now_utc_iso()
                    }

                    for o in f.get("odds", []):
                        row = base_data.copy()
                        row.update({
                            "sportsbook": o.get("sportsbook"),
                            "market": o.get("market"),
                            "selection": o.get("selection"),
                            "price_american": o.get("price_american"),
                            "price_decimal": o.get("price_decimal") or american_to_decimal(o.get("price_american")),
                        })
                        rows.append(row)
                        
                        # Stop early if we have enough rows
                        if len(rows) >= max_events * 5: return rows
            return rows
    except Exception as e:
        st.warning(f"⚠️ {sport_key}: Stream disconnected or empty. ({e})")
        return rows

# ==============================
#  LOGIC ENGINES
# ==============================

def merge_into_history(history_df: pd.DataFrame, new_rows: list) -> pd.DataFrame:
    """Concatenates new data into the session history."""
    if not new_rows: return history_df
    new_df = pd.DataFrame(new_rows)
    if history_df is None or history_df.empty: return new_df
    return pd.concat([history_df, new_df], ignore_index=True)

def compute_line_moves(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    THE STEAM ENGINE
    Groups by unique Selection+Book and compares First Timestamp vs Last Timestamp.
    """
    if history_df is None or history_df.empty: return pd.DataFrame()

    # Filter out bad data
    df = history_df.dropna(subset=["fixture_id", "sportsbook", "selection", "market"]).copy()
    df["price_decimal"] = df["price_decimal"].astype(float)

    agg_rows = []
    # We group by these 4 keys to track a specific line on a specific book
    group_cols = ["fixture_id", "sportsbook", "selection", "market"]

    for key, grp in df.groupby(group_cols):
        # Sort by time to find Start and End
        grp = grp.sort_values("timestamp")
        
        first = grp.iloc[0]
        last = grp.iloc[-1]

        open_dec = float(first["price_decimal"])
        curr_dec = float(last["price_decimal"])
        
        # Calculate diff
        move = curr_dec - open_dec
        
        # Only keep if there is non-zero movement (optional, but cleaner)
        if abs(move) < 0.001: continue 

        agg_rows.append({
            "fixture_id": key[0],
            "sportsbook": key[1],
            "selection": key[2],
            "market": key[3],
            "league": last.get("league"),
            "home_team": last.get("home_team"),
            "away_team": last.get("away_team"),
            "open_odds": open_dec,
            "current_odds": curr_dec,
            "line_move": move,
            "abs_move": abs(move),
            "move_direction": "Steam to DOG 🐶" if move > 0 else "Steam to FAV 🔥"
        })

    if not agg_rows: return pd.DataFrame()
    
    res = pd.DataFrame(agg_rows)
    return res.sort_values("abs_move", ascending=False).reset_index(drop=True)

def detect_arbitrage(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    THE ARBITRAGE ENGINE
    Finds instances where implied probability sum < 100% across different books.
    """
    if odds_df is None or odds_df.empty: return pd.DataFrame()

    # Use only the LATEST odds for every selection
    # 1. Sort by time
    df = odds_df.sort_values("timestamp")
    # 2. Group by unique key and take the last one
    latest_state = df.groupby(["fixture_id", "sportsbook", "selection", "market"]).tail(1)
    
    arb_rows = []
    
    # Analyze by Market (e.g. Moneyline for Lakers vs Celtics)
    for (fixture_id, market), grp in latest_state.groupby(["fixture_id", "market"]):
        
        # Find best price for each selection
        best_prices = [] # (Selection, BestDecimal, Sportsbook)
        
        for selection, sub_grp in grp.groupby("selection"):
            best_idx = sub_grp["price_decimal"].idxmax()
            best_row = sub_grp.loc[best_idx]
            best_prices.append({
                "selection": selection,
                "price": best_row["price_decimal"],
                "book": best_row["sportsbook"]
            })
        
        # Calculate Implied Prob
        total_prob = sum(1/item["price"] for item in best_prices)
        
        if total_prob < 1.0:
            roi = (1.0 / total_prob) - 1.0
            
            # Formatting for display
            details = " | ".join([f"{x['selection']} @ {x['price']} ({x['book']})" for x in best_prices])
            
            arb_rows.append({
                "fixture_id": fixture_id,
                "market": market,
                "league": grp.iloc[0]["league"],
                "matchup": f"{grp.iloc[0]['home_team']} vs {grp.iloc[0]['away_team']}",
                "ROI_Pct": round(roi * 100, 2),
                "Details": details
            })

    if not arb_rows: return pd.DataFrame()
    return pd.DataFrame(arb_rows).sort_values("ROI_Pct", ascending=False)

# ==============================
#  UI MAIN
# ==============================

def main():
    st.set_page_config(page_title="EDGE|FORCE DOMINION", layout="wide", page_icon="⚡")

    # Initialize Session State (The "Database")
    if "odds_history" not in st.session_state:
        st.session_state.odds_history = pd.DataFrame()
    if "data_log" not in st.session_state:
        st.session_state.data_log = []

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚡ EDGE|FORCE Config")
        api_key = st.text_input("OpticOdds API Key", type="password")
        
        st.subheader("Filters")
        selected_sports = st.multiselect("Leagues", list(SPORT_MAP.keys()), default=["NBA"])
        selected_books = st.multiselect("Books", DEFAULT_BOOKS, default=DEFAULT_BOOKS[:4])
        selected_markets = st.multiselect("Markets", DEFAULT_MARKETS, default=["moneyline"])
        
        st.subheader("Data Depth")
        max_snap = st.slider("Snapshot Size (Fixtures)", 10, 100, 25)
        max_burst = st.slider("Stream Burst (Messages)", 50, 500, 100)

        st.markdown("---")
        if st.button("📡 RUN ENGINE (Snap + Stream)", type="primary"):
            if not api_key:
                st.error("Enter API Key first.")
            else:
                run_ingestion(api_key, selected_sports, selected_books, selected_markets, max_snap, max_burst)

    # --- MAIN DASHBOARD ---
    st.title("EDGE|FORCE DOMINION")
    st.caption("Live Sports Analytics & Arbitrage Engine")

    # Metrics Row
    history_len = len(st.session_state.odds_history)
    col1, col2, col3 = st.columns(3)
    col1.metric("Data Points Ingested", history_len)
    
    if history_len > 0:
        latest_ts = st.session_state.odds_history["timestamp"].max()
        col2.metric("Last Update (UTC)", latest_ts.split("T")[1][:8])
    else:
        col2.metric("Status", "Waiting for Data")

    # TABS
    tab_arb, tab_steam, tab_raw = st.tabs(["💰 Arbitrage", "🔥 Steam (Line Moves)", "📊 Raw Data"])

    # 1. ARBITRAGE TAB
    with tab_arb:
        arb_df = detect_arbitrage(st.session_state.odds_history)
        if arb_df.empty:
            st.info("No Arbitrage opportunities found in current data.")
        else:
            st.success(f"Found {len(arb_df)} Arbitrage Opportunities!")
            st.dataframe(arb_df, use_container_width=True)

    # 2. STEAM TAB
    with tab_steam:
        steam_df = compute_line_moves(st.session_state.odds_history)
        if steam_df.empty:
            st.info("No line movement detected yet. Run the engine again to capture changes.")
        else:
            st.warning(f"Detected {len(steam_df)} Line Movements!")
            
            # Add filter for big moves only
            min_move = st.slider("Min Decimal Move Filter", 0.01, 0.5, 0.05)
            filtered_steam = steam_df[steam_df["abs_move"] >= min_move]
            
            st.dataframe(
                filtered_steam[[
                    "league", "home_team", "away_team", "market", "selection", 
                    "sportsbook", "open_odds", "current_odds", "line_move", "move_direction"
                ]], 
                use_container_width=True
            )

    # 3. RAW DATA TAB
    with tab_raw:
        st.dataframe(st.session_state.odds_history.sort_values("timestamp", ascending=False).head(500))

# ==============================
#  EXECUTION LOGIC
# ==============================

def run_ingestion(api_key, sports, books, markets, max_snap, max_burst):
    """Orchestrates the Snapshot -> Stream -> Merge workflow."""
    
    new_data = []
    status_box = st.status("Initializing Engine...", expanded=True)

    for sport in sports:
        sport_key = SPORT_MAP[sport]
        
        # 1. SNAPSHOT
        status_box.write(f"📸 {sport}: Fetching Snapshot...")
        fixtures = fetch_fixtures_for_sport(api_key, sport_key, max_events=max_snap)
        if fixtures:
            snap_data = fetch_odds_for_fixtures(api_key, fixtures, books, markets)
            new_data.extend(snap_data)
            status_box.write(f"✅ {sport}: Snapshot complete ({len(snap_data)} rows)")
        
        # 2. STREAM BURST
        status_box.write(f"📡 {sport}: Opening Live Stream...")
        stream_data = stream_burst_for_sport(api_key, sport_key, books, markets, is_live=None, max_events=10, max_messages=max_burst)
        new_data.extend(stream_data)
        status_box.write(f"✅ {sport}: Stream burst complete ({len(stream_data)} rows)")

    status_box.update(label="Processing Data...", state="running")
    
    # 3. MERGE
    if new_data:
        st.session_state.odds_history = merge_into_history(st.session_state.odds_history, new_data)
        status_box.update(label="Engine Cycle Complete!", state="complete", expanded=False)
        st.rerun()
    else:
        status_box.update(label="No Data Found", state="error")

if __name__ == "__main__":
    main()
