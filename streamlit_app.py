import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

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
/* Page background */
section.main {
    background: radial-gradient(circle at top, #131a2f 0%, #05060a 60%);
    color: #f5f7ff;
}

/* Card style */
.efd-card {
    border-radius: 16px;
    padding: 1.25rem;
    border: 1px solid rgba(120, 180, 255, 0.45);
    background: rgba(10, 14, 30, 0.85);
    box-shadow: 0 0 20px rgba(0,150,255,0.25);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b0f1a, #05060a);
    border-right: 1px solid rgba(100,150,255,0.3);
}

/* Highlight rows */
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
    return (odds / 100 + 1.0) if odds > 0 else (100 / abs(odds) + 1.0)


def build_demo_snapshot(sport_code: str) -> pd.DataFrame:
    """Fallback demo so UI always renders."""
    now = datetime.utcnow()
    rows = [
        dict(
            fixture_id=f"{sport_code}-EX",
            league=sport_code.upper(),
            start_time=now + timedelta(hours=2),
            home_team="Edge City Titans",
            away_team="Quantum Sharks",
            market_key="moneyline",
            market_name="Moneyline",
            outcome_key="home",
            outcome_name="Edge City Titans",
            bookmaker="FanDuel",
            open_odds=-140,
            current_odds=-165,
            last_updated=now,
        ),
        dict(
            fixture_id=f"{sport_code}-EX",
            league=sport_code.upper(),
            start_time=now + timedelta(hours=2),
            home_team="Edge City Titans",
            away_team="Quantum Sharks",
            market_key="moneyline",
            market_name="Moneyline",
            outcome_key="away",
            outcome_name="Quantum Sharks",
            bookmaker="DraftKings",
            open_odds=120,
            current_odds=140,
            last_updated=now,
        ),
    ]
    return pd.DataFrame(rows)


def normalize_opticodds_json(raw: Dict[str, Any], sport_code: str) -> pd.DataFrame:
    """Normalizes OpticOdds JSON. Adjust once you see your real JSON."""
    fixtures = raw.get("fixtures") or raw.get("data") or []
    rows = []

    for f in fixtures:
        fixture_id = f.get("id")
        league = f.get("league", sport_code.upper())
        start_time = f.get("start_time") or f.get("commence_time")
        home_team = f.get("home_team")
        away_team = f.get("away_team")

        markets = f.get("markets", [])
        for m in markets:
            m_key = m.get("key", "moneyline")
            m_name = m.get("name", "Moneyline")

            for o in m.get("outcomes", []):
                rows.append(
                    dict(
                        fixture_id=fixture_id,
                        league=league,
                        start_time=start_time,
                        home_team=home_team,
                        away_team=away_team,
                        market_key=m_key,
                        market_name=m_name,
                        outcome_key=o.get("key"),
                        outcome_name=o.get("name"),
                        bookmaker=o.get("bookmaker"),
                        open_odds=o.get("open_odds") or o.get("odds"),
                        current_odds=o.get("current_odds") or o.get("odds"),
                        last_updated=o.get("last_updated"),
                    )
                )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["current_odds"] = pd.to_numeric(df["current_odds"], errors="coerce")
    df["open_odds"] = pd.to_numeric(df["open_odds"], errors="coerce")
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")

    return df


def fetch_opticodds_snapshot(api_key, sport_code, sportsbooks, min_odds, max_odds):
    """API fetcher (falls back to demo if key missing)."""
    if not api_key:
        return build_demo_snapshot(sport_code)

    try:
        url = f"{OPTICODDS_API_BASE}/fixtures/odds/{sport_code}"
        params = {
            "sportsbooks": ",".join(sportsbooks),
            "markets": "moneyline",
            "event_status": "upcoming",
        }
        headers = {"x-api-key": api_key}

        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        json_raw = resp.json()

        df = normalize_opticodds_json(json_raw, sport_code)
        if df.empty:
            return build_demo_snapshot(sport_code)

        df = df[(df["current_odds"] >= min_odds) & (df["current_odds"] <= max_odds)]
        return df if not df.empty else pd.DataFrame()

    except:
        return build_demo_snapshot(sport_code)


def compute_line_movement(df: pd.DataFrame):
    df = df.copy()
    df["line_move"] = df["current_odds"] - df["open_odds"]
    df["move_abs"] = df["line_move"].abs()
    return df


def detect_steam(df, min_move):
    df = compute_line_movement(df)
    steam = df[df["move_abs"] >= min_move]
    return steam.sort_values("move_abs", ascending=False)


# FIXED FUNCTION (no syntax errors)
def detect_two_way_arbitrage(df: pd.DataFrame, min_edge_pct: float = 0.5):
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

        r = list(best.to_dict("records"))
        d1 = american_to_decimal(r[0]["current_odds"])
        d2 = american_to_decimal(r[1]["current_odds"])

        inv_sum = (1 / d1) + (1 / d2)

        if inv_sum < 1.0:
            edge = (1.0 - inv_sum) * 100.0
            if edge >= min_edge_pct:
                records.append(
                    dict(
                        fixture_id=fixture_id,
                        league=r[0]["league"],
                        start_time=r[0]["start_time"],
                        home_team=r[0]["home_team"],
                        away_team=r[0]["away_team"],
                        market_key=market_key,
                        outcome1=r[0]["outcome_name"],
                        book1=r[0]["bookmaker"],
                        odds1=r[0]["current_odds"],
                        outcome2=r[1]["outcome_name"],
                        book2=r[1]["bookmaker"],
                        odds2=r[1]["current_odds"],
                        arb_edge_pct=round(edge, 2),
                    )
                )

    return pd.DataFrame(records)


# ============================================================
#  SIDEBAR – CONTROL PANEL
# ============================================================
st.sidebar.markdown("## ⚙️ Control Panel")

api_key = st.sidebar.text_input("OpticOdds API Key", type="password")

mode = st.sidebar.radio(
    "Mode",
    ["🔥 Steam / Line Movement", "♟️ Arbitrage Radar"]
)

books = st.sidebar.multiselect("Sportsbooks", BOOKS_ALL, BOOKS_ALL)

min_odds, max_odds = st.sidebar.slider(
    "Odds Window",
    -1000, 1000,
    (-250, 250)
)

min_move = st.sidebar.slider("Steam threshold (cents)", 5, 80, 20)
min_arb_edge = st.sidebar.slider("Minimum arb edge (%)", 0.1, 5.0, 0.5)

# ============================================================
#  HEADER
# ============================================================
st.markdown(
    """
<div class="efd-card">
  <h1>🏆 Edge Force Dominion – Live Deck</h1>
  <p>Real-time steam, line movement, arbitrage, and market intelligence.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
#  TABS FOR SPORTS
# ============================================================
tabs = st.tabs(list(SPORTS.keys()))

for (label, code), tab in zip(SPORTS.items(), tabs):
    with tab:
        left, right = st.columns([1.5, 1])

        # SNAPSHOT
        df = fetch_opticodds_snapshot(api_key, code, books, min_odds, max_odds)

        with left:
            if "Steam" in mode:
                st.markdown(f"### 🔥 Steam – {label}")
                if df.empty:
                    st.info("No data.")
                else:
                    steam = detect_steam(df, min_move)
                    st.dataframe(steam)
            else:
                st.markdown(f"### ♟️ Arbitrage Radar – {label}")
                if df.empty:
                    st.info("No data.")
                else:
                    arb = detect_two_way_arbitrage(df, min_arb_edge)
                    st.dataframe(arb)

        with right:
            st.markdown("### 📊 Raw Odds")
            st.dataframe(df)