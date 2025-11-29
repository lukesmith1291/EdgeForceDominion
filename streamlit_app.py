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
#  STYLE – SPORTY / FUTURISTIC
# ============================================================
st.markdown(
    """
<style>
/* Global dark theme tweaks */
body {
    background-color: #05060a;
}

section.main {
    background: radial-gradient(circle at top, #141b33 0, #05060a 45%);
    color: #f5f7ff;
}

/* Headers */
h1, h2, h3 {
    font-family: -apple-system, system-ui, BlinkMacSystemFont, "SF Pro Display";
    letter-spacing: 0.03em;
}

/* Cards */
.efd-card {
    border-radius: 16px;
    padding: 1rem 1.25rem;
    border: 1px solid rgba(120, 180, 255, 0.45);
    background: linear-gradient(135deg, rgba(17, 24, 48, 0.98), rgba(8, 11, 22, 0.98));
    box-shadow: 0 0 25px rgba(0, 180, 255, 0.18);
}

/* Tables */
.dataframe tbody tr:hover {
    background-color: rgba(41, 98, 255, 0.18) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: radial-gradient(circle at top left, #111628 0, #05060a 55%);
    border-right: 1px solid rgba(100, 150, 255, 0.3);
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
    if odds > 0:
        return odds / 100.0 + 1.0
    else:
        return 100.0 / abs(odds) + 1.0


def build_demo_snapshot(sport_code: str) -> pd.DataFrame:
    """Fallback demo data so the UI always renders."""
    now = datetime.utcnow()
    rows = []

    # Single demo game per sport
    rows.append(
        dict(
            fixture_id=f"{sport_code.upper()}-EX1",
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
        )
    )
    rows.append(
        dict(
            fixture_id=f"{sport_code.upper()}-EX1",
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
        )
    )

    return pd.DataFrame(rows)


def normalize_opticodds_json(raw: Dict[str, Any], sport_code: str) -> pd.DataFrame:
    """
    🔧 THIS IS THE ONLY PART YOU'LL LIKELY NEED TO TWEAK ONCE
    YOU SEE REAL OPTICODDS JSON.

    Right now this is a generic normalizer that expects something like:

    {
      "fixtures": [
        {
          "id": "...",
          "league": "...",
          "home_team": "...",
          "away_team": "...",
          "start_time": "...",
          "markets": [
            {
              "key": "moneyline",
              "name": "Moneyline",
              "outcomes": [
                {
                  "key": "home",
                  "name": "...",
                  "bookmaker": "FanDuel",
                  "open_odds": -140,
                  "current_odds": -165,
                  "last_updated": "..."
                },
                ...
              ]
            }
          ]
        }
      ]
    }

    If your JSON shape is different, just adjust the traversal and keep
    the output columns the same.
    """
    fixtures = raw.get("fixtures") or raw.get("data") or []
    rows = []

    for f in fixtures:
        fixture_id = f.get("id") or f.get("fixture_id")
        league = f.get("league", {}).get("name") if isinstance(f.get("league"), dict) else f.get("league", sport_code.upper())
        start_time = f.get("start_time") or f.get("commence_time")
        home_team = f.get("home_team") or f.get("home")
        away_team = f.get("away_team") or f.get("away")

        markets = f.get("markets", []) or f.get("odds", [])
        for m in markets:
            m_key = m.get("key") or m.get("market_key") or "moneyline"
            m_name = m.get("name") or m_key.title()
            outcomes = m.get("outcomes") or m.get("lines") or []
            for o in outcomes:
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
                        open_odds=o.get("open_odds") or o.get("opening_line") or o.get("odds"),
                        current_odds=o.get("current_odds") or o.get("line") or o.get("odds"),
                        last_updated=o.get("last_updated") or f.get("last_update"),
                    )
                )

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    # Basic cleanup
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce", utc=True)
    df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce", utc=True)
    df["current_odds"] = pd.to_numeric(df["current_odds"], errors="coerce")
    df["open_odds"] = pd.to_numeric(df["open_odds"], errors="coerce")

    return df


def fetch_opticodds_snapshot(
    api_key: str,
    sport_code: str,
    sportsbooks: List[str],
    min_odds: int,
    max_odds: int,
    markets: List[str],
) -> pd.DataFrame:
    """
    Core fetcher. Ready for OpticOdds.

    - If api_key is empty OR request fails -> returns demo data.
    - Once you see real JSON in logs, tweak normalize_opticodds_json().
    """
    if not api_key:
        return build_demo_snapshot(sport_code)

    try:
        url = f"{OPTICODDS_API_BASE}/fixtures/odds/{sport_code}"
        params = {
            "sportsbooks": ",".join(sportsbooks) if sportsbooks else None,
            "markets": ",".join(markets) if markets else None,
            "event_status": "upcoming",
        }
        headers = {"x-api-key": api_key}

        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        raw = resp.json()

        df = normalize_opticodds_json(raw, sport_code)
        if df.empty:
            st.warning("OpticOdds returned no rows – falling back to demo data.")
            return build_demo_snapshot(sport_code)

        # Odds filter
        df = df[(df["current_odds"] >= min_odds) & (df["current_odds"] <= max_odds)]
        if df.empty:
            return df

        return df

    except Exception as e:
        st.warning(f"OpticOdds error: {e}. Showing demo data instead.")
        return build_demo_snapshot(sport_code)


def compute_line_movement(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["line_move"] = df["current_odds"] - df["open_odds"]
    df["move_abs"] = df["line_move"].abs()
    return df


def detect_steam(df: pd.DataFrame, min_move: int = 20) -> pd.DataFrame:
    if df.empty:
        return df
    df = compute_line_movement(df)
    steam = df[df["move_abs"] >= min_move]
    return steam.sort_values("move_abs", ascending=False)


def detect_two_way_arbitrage(df: pd.DataFrame, min_edge_pct: float = 0.5) -> pd.DataFrame:
    if df.empty:
        return df

    records = []
    for (fixture_id, market_key), group in df.groupby(["fixture_id", "market_key"]):
        best_by_outcome = (
            group.sort_values("current_odds", ascending=False)
            .groupby("outcome_key")
            .head(1)
        )

        if best_by_outcome["outcome_key"].nunique() != 2:
            continue

        sides = list(best_by_outcome.to_dict("records"))
        d1 = american_to_decimal(sides[0]["current_odds"])
        d2 = american_to_decimal(sides[1]["current_odds"])
        inv_sum = 1.0 / d1 + 1.0 / d2

        if inv_sum < 1.0:
            edge = (1.0 -