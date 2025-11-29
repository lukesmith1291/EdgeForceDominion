import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

import pandas as pd
import requests
import streamlit as st

# ============================================================
#  GLOBAL CONFIG
# ============================================================
st.set_page_config(
    page_title="Edge Force Dominion – Live Deck",
    layout="wide",
)

OPTICODDS_API_BASE = "https://api.opticodds.com/api/v3"

SPORTS = {
    "🏀 NBA": "nba",
    "🏈 NFL": "nfl",
    "🏒 NHL": "nhl",
    "🏀 NCAAB": "ncaab",
    "🏈 NCAAF": "ncaaf",
    "⚽ Soccer": "soccer",
}

BOOKS_ALL = ["FanDuel", "DraftKings", "BetMGM", "Caesars", "Pinnacle", "LowVig"]

# ============================================================
#  STYLE – FUTURISTIC UI
# ============================================================
st.markdown(
    """
<style>
section.main {
    background: radial-gradient(circle at top, #131a2f 0%, #05060a 60%);
    color: #f5f7ff;
    font-family: -apple-system, system-ui, BlinkMacSystemFont, "SF Pro Display";
}

.efd-card {
    border-radius: 16px;
    padding: 1.25rem;
    border: 1px solid rgba(120, 180, 255, 0.45);
    background: rgba(10, 14, 30, 0.92);
    box-shadow: 0 0 24px rgba(0,150,255,0.30);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b0f1a, #05060a);
    border-right: 1px solid rgba(100,150,255,0.3);
}

.dataframe tbody tr:hover {
    background-color: rgba(41, 98, 255, 0.22) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
#  HELPERS
# ============================================================

def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal."""
    try:
        odds = float(odds)
    except Exception:
        return 1.0
    return (odds / 100.0 + 1.0) if odds > 0 else (100.0 / abs(odds) + 1.0)


def normalize_opticodds_json(raw: Dict[str, Any], sport_code: str) -> pd.DataFrame:
    """
    Normalize OpticOdds JSON into a tabular frame.
    You may need to tweak this once you see real JSON.
    """
    fixtures = raw.get("fixtures") or raw.get("data") or []
    rows: List[Dict[str, Any]] = []

    for f in fixtures:
        fixture_id = f.get("id") or f.get("fixture_id")
        league = f.get("league", sport_code.upper())
        start_time = f.get("start_time") or f.get("commence_time")
        home_team = f.get("home_team") or f.get("home")
        away_team = f.get("away_team") or f.get("away")

        markets = f.get("markets") or f.get("odds") or []
        for m in markets:
            m_key = m.get("key") or m.get("market_key") or "moneyline"
            m_name = m.get("name") or "Moneyline"

            outcomes = m.get("outcomes") or m.get("lines") or []
            for o in outcomes:
                open_val = (
                    o.get("open_odds")
                    or o.get("opening_line")
                    or o.get("opening_odds")
                    or o.get("open")
                    or o.get("odds")
                )
                current_val = (
                    o.get("current_odds")
                    or o.get("current_line")
                    or o.get("live_line")
                    or o.get("line")
                    or o.get("odds")
                )
                rows.append(
                    dict(
                        fixture_id=fixture_id,
                        league=league,
                        start_time=start_time,
                        home_team=home_team,
                        away_team=away_team,
                        market_key=m_key,
                        market_name=m_name,
                        outcome_key=o.get("key") or o.get("outcome_key"),
                        outcome_name=o.get("name") or o.get("outcome_name"),
                        bookmaker=o.get("bookmaker") or o.get("book") or "Unknown",
                        open_odds=open_val,
                        current_odds=current_val,
                        last_updated=o.get("last_updated") or f.get("last_update"),
                    )
                )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["current_odds"] = pd.to_numeric(df["current_odds"], errors="coerce")
    df["open_odds"] = pd.to_numeric(df["open_odds"], errors="coerce")
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce", utc=True)
    df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce", utc=True)
    return df


def test_opticodds_connection(api_key: str) -> Tuple[bool, str, Dict[str, Any]]:
    """Test NBA once and report status + raw JSON (for debugging)."""
    if not api_key:
        return False, "No API key provided.", {}

    try:
        url = f"{OPTICODDS_API_BASE}/fixtures/odds/nba"
        params = {"markets": "moneyline", "event_status": "upcoming"}
        headers = {"x-api-key": api_key}
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}: {resp.text[:200]}", {}
        raw = resp.json()
        return True, "OpticOdds connection OK.", raw
    except Exception as e:
        return False, f"Error: {e}", {}


def fetch_opticodds_snapshot(
    api_key: str,
    sport_code: str,
    sportsbooks: List[str],
    min_odds: int,
    max_odds: int,
) -> pd.DataFrame:
    """
    Pull a snapshot from OpticOdds.
    No demos, no fake rows.
    On any error or empty result: returns an empty DataFrame.
    """
    if not api_key:
        st.error("No OpticOdds API key set. Paste it in the sidebar and press 'Test & Use This Key'.")
        return pd.DataFrame()

    try:
        url = f"{OPTICODDS_API_BASE}/fixtures/odds/{sport_code}"
        params = {
            "sportsbooks": ",".join(sportsbooks) if sportsbooks else None,
            "markets": "moneyline",
            "event_status": "upcoming",
        }
        headers = {"x-api-key": api_key}

        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        raw = resp.json()

        df = normalize_opticodds_json(raw, sport_code)
        if df.empty:
            st.warning("OpticOdds returned no usable rows for this sport / filters.")
            return pd.DataFrame()

        df = df[(df["current_odds"] >= min_odds) & (df["current_odds"] <= max_odds)]
        if df.empty:
            st.info("Live data returned, but nothing inside your odds window.")
            return pd.DataFrame()

        return df

    except Exception as e:
        st.error(f"Error while calling OpticOdds: {e}")
        return pd.DataFrame()


def compute_line_movement(df: pd.DataFrame) -> pd.DataFrame:
    """Add line_move (cents), move_abs, and move_direction columns."""
    if df.empty:
        return df
    df = df.copy()
    df["line_move"] = df["current_odds"] - df["open_odds"]
    df["move_abs"] = df["line_move"].abs()

    def _direction(x: float) -> str:
        if pd.isna(x):
            return "Unknown"
        if x > 0:
            return "Steam toward plus side"
        if x < 0:
            return "Steam toward favorite"
        return "No move"

    df["move_direction"] = df["line_move"].apply(_direction)
    return df


def detect_steam(df: pd.DataFrame, min_move: int) -> pd.DataFrame:
    """Filter rows with >= min_move cents of line movement."""
    if df.empty:
        return df
    df = compute_line_movement(df)
    steam = df[df["move_abs"] >= min_move]
    return steam.sort_values("move_abs", ascending=False)


def detect_two_way_arbitrage(df: pd.DataFrame, min_edge_pct: float) -> pd.DataFrame:
    """Simple 2-way arb finder based on best price each side."""
    if df.empty:
        return df

    records = []
    grouped = df.groupby(["fixture_id", "market_key"])

    for (fixture_id, market_key), group in grouped:
        best = (
            group.sort_values("current_odds", ascending=False)
            .groupby("outcome_key")
            .head(1)
        )

        if best["outcome_key"].nunique() != 2:
            continue

        recs = list(best.to_dict("records"))
        r1, r2 = recs[0], recs[1]

        d1 = american_to_decimal(r1["current_odds"])
        d2 = american_to_decimal(r2["current_odds"])

        inv_sum = (1.0 / d1) + (1.0 / d2)
        if inv_sum >= 1.0:
            continue

        edge = (1.0 - inv_sum) * 100.0
        if edge < min_edge_pct:
            continue

        records.append(
            dict(
                fixture_id=fixture_id,
                league=r1["league"],
                start_time=r1["start_time"],
                home_team=r1["home_team"],
                away_team=r1["away_team"],
                market_key=market_key,
                outcome1=r1["outcome_name"],
                book1=r1["bookmaker"],
                odds1=r1["current_odds"],
                outcome2=r2["outcome_name"],
                book2=r2["bookmaker"],
                odds2=r2["current_odds"],
                arb_edge_pct=round(edge, 2),
            )
        )

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("arb_edge_pct", ascending=False)


# ============================================================
#  SIDEBAR – CONTROL PANEL
# ============================================================

st.sidebar.markdown("## ⚙️ Edge Force Control Panel")

# Session state for key + test result
if "optic_key" not in st.session_state:
    st.session_state["optic_key"] = ""
if "test_ok" not in st.session_state:
    st.session_state["test_ok"] = None
if "test_message" not in st.session_state:
    st.session_state["test_message"] = ""
if "test_raw" not in st.session_state:
    st.session_state["test_raw"] = {}

api_key_input = st.sidebar.text_input("OpticOdds API Key", type="password")

if st.sidebar.button("🔌 Test & Use This Key"):
    key = api_key_input.strip()
    st.session_state["optic_key"] = key
    ok, msg, raw = test_opticodds_connection(key)
    st.session_state["test_ok"] = ok
    st.session_state["test_message"] = msg
    st.session_state["test_raw"] = raw

mode = st.sidebar.radio(
    "Radar Mode",
    ["🔥 Steam / Line Movement", "♟️ Arbitrage Radar"],
)

books = st.sidebar.multiselect(
    "Sportsbooks",
    BOOKS_ALL,
    default=BOOKS_ALL,
)

min_odds, max_odds = st.sidebar.slider(
    "Odds window (American)",
    -1000,
    1000,
    (-250, 250),
)

min_move = st.sidebar.slider(
    "Steam threshold (cents)",
    5,
    80,
    20,
)

min_arb_edge = st.sidebar.slider(
    "Min arbitrage edge (%)",
    0.1,
    5.0,
    0.5,
)

if st.sidebar.button("🔄 Refresh data"):
    st.experimental_rerun()

active_key = st.session_state["optic_key"]

# ============================================================
#  HEADER + CONNECTION STATUS
# ============================================================

st.markdown(
    """
<div class="efd-card">
  <h1 style="margin-bottom:0.25rem;">🏆 Edge Force Dominion – Live Deck</h1>
  <p style="opacity:0.88;margin-bottom:0.35rem;">
    Real-time steam, line movement and cross-book arbitrage radar. No fake data, ever.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

if st.session_state["test_ok"] is True:
    st.success(f"OpticOdds: {st.session_state['test_message']}")
elif st.session_state["test_ok"] is False:
    st.error(f"OpticOdds: {st.session_state['test_message']}")
else:
    st.info("Paste your OpticOdds key in the sidebar and press '🔌 Test & Use This Key'.")

if st.session_state["test_raw"]:
    with st.expander("DEBUG: Raw OpticOdds JSON from test call"):
        st.json(st.session_state["test_raw"])

st.markdown("---")

# ============================================================
#  TABS – ONE PER SPORT
# ============================================================

tabs = st.tabs(list(SPORTS.keys()))

for (label, sport_code), tab in zip(SPORTS.items(), tabs):
    with tab:
        left, right = st.columns([1.6, 1.0])

        with left:
            st.markdown(f"### {label} – {mode}")

        df_snapshot = fetch_opticodds_snapshot(
            api_key=active_key,
            sport_code=sport_code,
            sportsbooks=books,
            min_odds=min_odds,
            max_odds=max_odds,
        )

        with left:
            if df_snapshot.empty:
                st.info("No live rows for this sport with current filters.")
            else:
                if "Steam" in mode:
                    steam_df = detect_steam(df_snapshot, min_move)
                    if steam_df.empty:
                        st.success("No steam above your threshold.")
                    else:
                        view = steam_df[
                            [
                                "league",
                                "start_time",
                                "home_team",
                                "away_team",
                                "market_name",
                                "bookmaker",
                                "open_odds",
                                "current_odds",
                                "line_move",
                                "move_direction",
                                "last_updated",
                            ]
                        ]
                        st.dataframe(view, use_container_width=True, hide_index=True)
                else:
                    arb_df = detect_two_way_arbitrage(df_snapshot, min_arb_edge)
                    if arb_df.empty:
                        st.success("No 2-way arbitrage edges above threshold.")
                    else:
                        st.dataframe(arb_df, use_container_width=True, hide_index=True)

        with right:
            st.markdown("### 📊 Raw odds snapshot")
            if df_snapshot.empty:
                st.write("No odds to show.")
            else:
                base_view = df_snapshot[
                    [
                        "league",
                        "start_time",
                        "home_team",
                        "away_team",
                        "market_name",
                        "outcome_name",
                        "bookmaker",
                        "open_odds",
                        "current_odds",
                        "last_updated",
                    ]
                ].sort_values("start_time")
                st.dataframe(base_view, use_container_width=True, hide_index=True)