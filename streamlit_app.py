import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple

import pandas as pd
import requests
import streamlit as st

# ============================================================
#  CONFIG
# ============================================================
st.set_page_config(page_title="Edge Force Dominion – Live Stream", layout="wide")

OPTICODDS_API_BASE = "https://api.opticodds.com/api/v3"

SPORT_MAP = {
    "🏀 NBA": {"sport": "basketball", "leagues": ["NBA"]},
    "🏀 NCAAB": {"sport": "basketball", "leagues": ["NCAAB"]},
    "🏈 NFL": {"sport": "football", "leagues": ["NFL"]},
    "🏈 NCAAF": {"sport": "football", "leagues": ["NCAAF"]},
    "🏒 NHL": {"sport": "hockey", "leagues": ["NHL"]},
    "⚽ Soccer": {"sport": "soccer", "leagues": []},
}

BOOKS_ALL = ["FanDuel", "DraftKings", "BetMGM", "Caesars", "Pinnacle", "LowVig"]

# ============================================================
#  STYLE
# ============================================================
st.markdown(
    """
<style>
section.main {
    background: radial-gradient(circle at top, #111827 0%, #020617 55%);
    color: #f9fafb;
    font-family: -apple-system, system-ui, BlinkMacSystemFont, "SF Pro Display";
}

.efd-card {
    border-radius: 18px;
    padding: 1.25rem 1.5rem;
    border: 1px solid rgba(96, 165, 250, 0.65);
    background: rgba(15, 23, 42, 0.94);
    box-shadow: 0 0 32px rgba(37, 99, 235, 0.55);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #020617);
    border-right: 1px solid rgba(55, 65, 81, 0.8);
}

.dataframe tbody tr:hover {
    background-color: rgba(59, 130, 246, 0.22) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
#  UTILS
# ============================================================

def american_to_decimal(odds: float) -> float:
    try:
        o = float(odds)
    except Exception:
        return 1.0
    return (o / 100.0 + 1.0) if o > 0 else (100.0 / abs(o) + 1.0)


def parse_sse_stream(
    resp: requests.Response,
    max_events: int = 200,
    max_seconds: int = 8,
) -> List[Tuple[str, str]]:
    """
    Minimal SSE parser (no extra deps).
    Returns list of (event_name, data_json).
    """
    events: List[Tuple[str, str]] = []
    event_name = None
    data_lines: List[str] = []
    start = time.time()

    for raw in resp.iter_lines(decode_unicode=True):
        if time.time() - start > max_seconds:
            break
        if raw is None:
            continue
        line = raw.strip()
        if not line:
            # dispatch
            if event_name and data_lines:
                events.append((event_name, "\n".join(data_lines)))
                if len(events) >= max_events:
                    break
            event_name = None
            data_lines = []
            continue
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())

    return events


def stream_odds_once(
    api_key: str,
    sport: str,
    leagues: List[str],
    sportsbooks: List[str],
    markets: List[str],
    max_events: int = 150,
) -> pd.DataFrame:
    """
    Hit /stream/odds/{sport} once, collect a burst of odds + locked-odds events,
    and return a DataFrame with open/current price + line movement per selection.
    """
    if not api_key:
        st.error("Paste your OpticOdds key in the sidebar and hit 'Test & Use Key'.")
        return pd.DataFrame()

    params: Dict[str, Any] = {
        "key": api_key,
        "sportsbook": sportsbooks or [],
        "market": markets or ["Moneyline"],
    }
    if leagues:
        params["league"] = leagues
    url = f"{OPTICODDS_API_BASE}/stream/odds/{sport}"

    try:
        resp = requests.get(url, params=params, stream=True, timeout=12)
        resp.raise_for_status()
    except Exception as e:
        st.error(f"Error opening stream: {e}")
        return pd.DataFrame()

    events = parse_sse_stream(resp, max_events=max_events)
    resp.close()

    rows: List[Dict[str, Any]] = []
    for ev_name, data_str in events:
        if ev_name not in ("odds", "locked-odds"):
            continue
        try:
            payload = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        entry_type = payload.get("type", ev_name)
        for odd in payload.get("data", []):
            rows.append(
                dict(
                    event_type=entry_type,
                    fixture_id=odd.get("fixture_id"),
                    game_id=odd.get("game_id"),
                    sport=odd.get("sport"),
                    league=odd.get("league"),
                    market=odd.get("market"),
                    selection=odd.get("name") or odd.get("selection"),
                    sportsbook=odd.get("sportsbook"),
                    price=odd.get("price"),
                    timestamp=odd.get("timestamp"),
                )
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    # aggregate to open/current per selection within this burst
    key_cols = ["fixture_id", "sportsbook", "market", "selection"]
    df = df.dropna(subset=["price", "timestamp"])
    df = df.sort_values("timestamp")

    agg_rows: List[Dict[str, Any]] = []
    for key, grp in df.groupby(key_cols):
        first = grp.iloc[0]
        last = grp.iloc[-1]
        open_price = float(first["price"])
        curr_price = float(last["price"])
        move = curr_price - open_price
        agg_rows.append(
            dict(
                fixture_id=key[0],
                sportsbook=key[1],
                market=key[2],
                selection=key[3],
                league=str(last["league"]),
                sport=str(last["sport"]),
                open_odds=open_price,
                current_odds=curr_price,
                line_move=move,
                move_abs=abs(move),
                last_timestamp=float(last["timestamp"]),
            )
        )

    if not agg_rows:
        return pd.DataFrame()
    out = pd.DataFrame(agg_rows)
    out["move_direction"] = out["line_move"].apply(
        lambda x: "Steam to dog" if x > 0 else ("Steam to fav" if x < 0 else "Flat")
    )
    return out.sort_values("move_abs", ascending=False)


def detect_arbitrage(df: pd.DataFrame, min_edge_pct: float) -> pd.DataFrame:
    """Very simple two-way arb finder off latest prices."""
    if df.empty:
        return df

    records: List[Dict[str, Any]] = []
    grouped = df.groupby(["fixture_id", "market"])

    for (fixture_id, market), grp in grouped:
        best = grp.sort_values("current_odds", ascending=False).groupby("selection").head(1)
        if best["selection"].nunique() != 2:
            continue
        recs = list(best.to_dict("records"))
        a, b = recs[0], recs[1]
        d1 = american_to_decimal(a["current_odds"])
        d2 = american_to_decimal(b["current_odds"])
        inv_sum = 1.0 / d1 + 1.0 / d2
        if inv_sum >= 1.0:
            continue
        edge = (1.0 - inv_sum) * 100.0
        if edge < min_edge_pct:
            continue
        records.append(
            dict(
                fixture_id=fixture_id,
                market=market,
                selection1=a["selection"],
                book1=a["sportsbook"],
                odds1=a["current_odds"],
                selection2=b["selection"],
                book2=b["sportsbook"],
                odds2=b["current_odds"],
                arb_edge_pct=round(edge, 2),
            )
        )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("arb_edge_pct", ascending=False)


def test_key_once(api_key: str) -> Tuple[bool, str]:
    """Quick smoke test: open NBA moneyline stream."""
    try:
        url = f"{OPTICODDS_API_BASE}/stream/odds/basketball"
        params = {
            "key": api_key,
            "sportsbook": ["DraftKings"],
            "market": ["Moneyline"],
            "league": ["NBA"],
        }
        resp = requests.get(url, params=params, stream=True, timeout=6)
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}: {resp.text[:180]}"
        _ = next(resp.iter_lines(decode_unicode=True), None)
        resp.close()
        return True, "Stream opened successfully."
    except Exception as e:
        return False, str(e)

# ============================================================
#  SIDEBAR – CONTROLS
# ============================================================

st.sidebar.markdown("## ⚙️ Edge Force Control")

if "optic_key" not in st.session_state:
    st.session_state["optic_key"] = ""
if "key_status" not in st.session_state:
    st.session_state["key_status"] = None
if "key_message" not in st.session_state:
    st.session_state["key_message"] = ""

key_input = st.sidebar.text_input("OpticOdds API Key", type="password")

if st.sidebar.button("🔌 Test & Use Key"):
    st.session_state["optic_key"] = key_input.strip()
    ok, msg = test_key_once(st.session_state["optic_key"])
    st.session_state["key_status"] = ok
    st.session_state["key_message"] = msg

mode = st.sidebar.radio("Mode", ["🔥 Steam / Line Move", "♟️ Arbitrage Radar"])

books = st.sidebar.multiselect("Sportsbooks", BOOKS_ALL, default=BOOKS_ALL)
min_move = st.sidebar.slider("Min line move (cents)", 5, 100, 20)
min_arb_edge = st.sidebar.slider("Min arb edge (%)", 0.1, 5.0, 0.5)
burst_events = st.sidebar.slider("Events per refresh (per sport)", 40, 400, 160, step=40)

if st.sidebar.button("🔄 Pull fresh stream burst"):
    st.experimental_rerun()

active_key = st.session_state["optic_key"]

# ============================================================
#  HEADER
# ============================================================

st.markdown(
    """
<div class="efd-card">
  <h1 style="margin-bottom:0.25rem;">🏆 Edge Force Dominion — Live OpticOdds Stream</h1>
  <p style="opacity:0.9;margin-bottom:0;">
    Straight from <code>/stream/odds/{sport}</code>. No demos, no fake rows. 
    Prices update based on the last burst of SSE events.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

if st.session_state["key_status"] is True:
    st.success(f"OpticOdds stream: {st.session_state['key_message']}")
elif st.session_state["key_status"] is False:
    st.error(f"OpticOdds stream: {st.session_state['key_message']}")
else:
    st.info("Paste your key and hit '🔌 Test & Use Key' to verify the stream.")

st.markdown("---")

# ============================================================
#  TABS PER SPORT
# ============================================================

tabs = st.tabs(list(SPORT_MAP.keys()))

for tab_label, tab in zip(SPORT_MAP.keys(), tabs):
    with tab:
        cfg = SPORT_MAP[tab_label]
        sport = cfg["sport"]
        leagues = cfg["leagues"]

        st.markdown(f"### {tab_label} — {mode}")

        if not active_key:
            st.warning("No API key set.")
            continue

        # pull a burst of events for this sport
        df = stream_odds_once(
            api_key=active_key,
            sport=sport,
            leagues=leagues,
            sportsbooks=books,
            markets=["Moneyline"],
            max_events=burst_events,
        )

        if df.empty:
            st.info("No odds came through in this burst (for this sport / filters).")
            continue

        if mode.startswith("🔥"):  # steam view
            steam_df = df[df["move_abs"] >= min_move]
            if steam_df.empty:
                st.success("No steam above your threshold in this burst.")
            else:
                view = steam_df[
                    [
                        "league",
                        "market",
                        "selection",
                        "sportsbook",
                        "open_odds",
                        "current_odds",
                        "line_move",
                        "move_direction",
                    ]
                ]
                st.dataframe(view, use_container_width=True, hide_index=True)
        else:
            arb_df = detect_arbitrage(df, min_arb_edge)
            if arb_df.empty:
                st.success("No clear 2-way arbitrage in this burst.")
            else:
                st.dataframe(arb_df, use_container_width=True, hide_index=True)