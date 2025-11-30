import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ======================================
# 🔑 CONFIG
# ======================================

OPTICODDS_API_KEY = os.getenv("OPTICODDS_API_KEY", "")
OPTICODDS_BASE = "https://api.opticodds.com/api/v3"

# What we auto-build at startup
SPORTS_CONFIG = {
    "nba":  {"sport": "basketball", "league": "nba"},
    "ncaab": {"sport": "basketball", "league": "ncaab"},
    "nfl":  {"sport": "football", "league": "nfl"},
    "ncaaf": {"sport": "football", "league": "ncaaf"},
    "mlb":  {"sport": "baseball", "league": "mlb"},
    "nhl":  {"sport": "ice_hockey", "league": "nhl"},
}

CORE_MARKETS = ["moneyline", "spread", "total_points"]  # adjust if OpticOdds uses different IDs
DEFAULT_SPORTSBOOKS = ["DraftKings", "FanDuel", "Caesars", "BetMGM"]

# Max fixture_ids per /fixtures/odds call (per docs it’s typically 5)
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
        </style>
        """,
        unsafe_allow_html=True,
    )


# ======================================
# 🧮 HELPER FUNCTIONS
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


def optic_get(path: str, params: Dict) -> Dict:
    params = dict(params)
    params["key"] = OPTICODDS_API_KEY
    url = f"{OPTICODDS_BASE}{path}"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


# ======================================
# 🌐 STATIC DATA FETCHERS
# ======================================

def fetch_active_fixtures(sport_id: str, league_id: str, days_ahead: int = 2) -> pd.DataFrame:
    """
    /fixtures/active → only fixtures that have odds.
    """
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
        rows.append(
            {
                "fixture_id": fx["id"],
                "sport": fx["sport"]["id"],
                "league": fx["league"]["id"],
                "start_date": fx.get("start_date"),
                "home_name": fx.get("home_team_display"),
                "away_name": fx.get("away_team_display"),
                "home_logo": (fx.get("home_competitors") or [{}])[0].get("logo"),
                "away_logo": (fx.get("away_competitors") or [{}])[0].get("logo"),
                "status": fx.get("status"),
            }
        )
    return pd.DataFrame(rows)


def fetch_odds_for_fixtures(
    fixture_ids: List[str],
    markets: List[str],
    sportsbooks: List[str],
) -> pd.DataFrame:
    """
    /fixtures/odds → static snapshot for ML / Spread / Total.
    Assumes odds payload is a flat list per fixture with fields:
      - sportsbook
      - market_id or market
      - selection / name
      - price (American)
      - grouping_key
      - points (for spreads/totals)
    You may need to tweak field names based on your account response.
    """
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
    return df


def compute_no_vig_ev(board_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute fair probabilities & EV (no-vig) per (fixture, market_id, grouping_key).
    """
    if board_df.empty:
        return board_df

    board_df = board_df.copy()
    board_df["fair_prob"] = np.nan
    board_df["no_vig_ev"] = np.nan

    group_cols = ["fixture_id", "market_id", "grouping_key"]
    grouped = board_df.groupby(group_cols, as_index=False)

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
        all_updates = pd.concat(updates, ignore_index=True)
        join_cols = ["fixture_id", "market_id", "grouping_key", "selection", "sportsbook"]
        board_df = board_df.merge(
            all_updates[join_cols + ["fair_prob", "no_vig_ev"]],
            on=join_cols,
            how="left",
            suffixes=("", "_upd"),
        )
        # In case of nulls, keep original
        for col in ["fair_prob", "no_vig_ev"]:
            board_df[col] = board_df[col + "_upd"].combine_first(board_df[col])
            board_df.drop(columns=[col + "_upd"], inplace=True)

    return board_df


# ======================================
# 🔒 SESSION STATE
# ======================================

def init_state():
    if "boot_done" not in st.session_state:
        st.session_state["boot_done"] = False
    if "boot_log" not in st.session_state:
        st.session_state["boot_log"] = []
    if "board_df" not in st.session_state:
        st.session_state["board_df"] = pd.DataFrame()
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "sportsbooks" not in st.session_state:
        st.session_state["sportsbooks"] = DEFAULT_SPORTSBOOKS.copy()


def log_boot(msg: str):
    st.session_state["boot_log"].append(msg)


# ======================================
# 🚀 BOOT SEQUENCE (RUNS BEFORE DASHBOARD)
# ======================================

def boot_backend():
    """
    Full startup build:
      - loop all SPORTS_CONFIG entries
      - fetch fixtures/active
      - fetch odds (ML/Spread/Total)
      - compute implied, fair_prob, no_vig_ev
      - build unified board_df
    