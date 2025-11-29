# streamlit_app.py
# EDGE FORCE DOMINION – Multi-sport OpticOdds dashboard (poll + history)
# NOTE: you may need to tweak a few parameter names to match the live OpticOdds docs.

import time
import json
from datetime import datetime, timezone

import requests
import pandas as pd
import numpy as np
import streamlit as st


# ======================================
#  CONFIG
# ======================================

OPTIC_BASE = "https://api.opticodds.com/api/v3"

# Tabs = high-level sports. We’ll show real leagues from the data.
SPORT_KEYS = {
    "🏀 Basketball": "basketball",
    "🏈 Football": "football",
    "⚾ Baseball": "baseball",
    "🏒 Hockey": "hockey",
}

DEFAULT_BOOKS = ["FanDuel", "DraftKings", "BetMGM", "Caesars", "Pinnacle", "LowVig"]
DEFAULT_MARKETS = ["moneyline", "point_spread", "total_points"]


# ======================================
#  UTILS
# ======================================

def now_iso():
    return datetime.now(timezone.utc).isoformat()


def american_to_decimal(odds):
    """Safe American → decimal."""
    try:
        o = float(odds)
    except Exception:
        return 1.0
    if o == 0:
        return 1.0
    if o > 0:
        return 1.0 + (o / 100.0)
    return 1.0 + (100.0 / abs(o))


def ensure_df(obj):
    if isinstance(obj, pd.DataFrame):
        return obj
    if not obj:
        return pd.DataFrame()
    return pd.DataFrame(obj)


def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# ======================================
#  OPTICODDS HELPERS
# ======================================

def fetch_fixtures(api_key: str, sport_slug: str, max_events: int = 100,
                   event_status: str = "upcoming") -> list:
    """
    Fetch fixtures (schedule) for a given high-level sport.

    ⚠️ TODO: Check OpticOdds docs:
    - If 'league' is required, add it.
    - If sport is named 'american_football' instead of 'football', adjust SPORT_KEYS.
    """
    params = {
        "key": api_key,
        "sport": sport_slug,
        "event_status": event_status,   # 'upcoming', 'live', etc.
        "page_size": max_events,
    }
    try:
        r = requests.get(f"{OPTIC_BASE}/fixtures", params=params, timeout=12)
        r.raise_for_status()
    except Exception as e:
        st.error(f"{sport_slug}: error fetching fixtures: {e}")
        return []

    return r.json().get("data", [])


def fetch_odds_snapshot(api_key: str,
                        fixture_ids: list,
                        sportsbooks: list,
                        markets: list) -> list:
    """
    Snapshot odds for a list of fixtures.

    ⚠️ THIS IS THE PLACE MOST LIKELY TO NEED A SMALL ADJUSTMENT.

    Two common styles:
    1) Repeated params:
       ?fixture_id=A&fixture_id=B&sportsbook=FanDuel&sportsbook=DK&market=moneyline
    2) Comma-separated:
       ?fixture_ids=A,B&sportsbooks=FanDuel,DraftKings&markets=moneyline,total_points

    I’ll implement the COMMA style as it’s less likely to 400; if you still see 400,
    swap to style (1) using multiple tuples instead.
    """

    if not fixture_ids or not sportsbooks or not markets:
        return []

    fixture_ids = list(dict.fromkeys(fixture_ids))
    sportsbooks = list(dict.fromkeys(sportsbooks))
    markets = list(dict.fromkeys(markets))

    rows = []

    # OpticOdds will usually have some limit; chunk to be safe.
    for fixture_chunk in chunk(fixture_ids, 25):
        params = {
            "key": api_key,
            # ⚠️ TODO: confirm param names; try 'fixture_ids' / 'sportsbooks' / 'markets'
            "fixture_ids": ",".join(fixture_chunk),
            "sportsbooks": ",".join(sportsbooks),
            "markets": ",".join(markets),
            "is_main": "True",
        }
        try:
            r = requests.get(f"{OPTIC_BASE}/fixtures/odds", params=params, timeout=12)
            r.raise_for_status()
        except Exception as e:
            st.error(f"Snapshot odds error: {e}")
            continue

        data = r.json().get("data", [])
        for f in data:
            fixture_id = f.get("id")
            sport = f.get("sport")
            league = f.get("league")
            home = f.get("home_team")
            away = f.get("away_team")
            start_time = f.get("start_time")

            for o in f.get("odds", []):
                price_dec = o.get("price_decimal")
                if price_dec is None:
                    price_dec = american_to_decimal(o.get("price_american"))

                rows.append(
                    {
                        "source": "snapshot",
                        "ingest_ts": now_iso(),
                        "fixture_id": fixture_id,
                        "sport": sport,
                        "league": league,
                        "home_team": home,
                        "away_team": away,
                        "start_time": start_time,
                        "sportsbook": o.get("sportsbook"),
                        "market": o.get("market"),
                        "selection": o.get("selection"),
                        "price_american": o.get("price_american"),
                        "price_decimal": price_dec,
                    }
                )

    return rows


# ======================================
#  EFD SCORING & ARBITRAGE
# ======================================

def compute_efd_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Very simple EFD v1: higher score = more mispricing + more dispersion across books.

    Inputs: all odds rows for a set of fixtures.
    """
    if df.empty:
        return pd.DataFrame()

    scored_rows = []
    # Group at fixture + market + selection level
    for (fixture_id, league, market, selection), grp in df.groupby(
        ["fixture_id", "league", "market", "selection"]
    ):
        dec = grp["price_decimal"].astype(float)
        best = dec.max()
        worst = dec.min()
        spread = best - worst
        # "Edge" as best vs average
        edge_pct = (best / dec.mean()) - 1.0 if dec.mean() > 0 else 0.0

        # Map into 0–100: baseline 40, plus boosts capped
        score = 40 + 300 * edge_pct + 20 * spread
        score = max(0, min(100, score))

        scored_rows.append(
            {
                "fixture_id": fixture_id,
                "league": league,
                "market": market,
                "selection": selection,
                "EFD_score": score,
                "best_price": best,
                "worst_price": worst,
                "books": ",".join(sorted(set(grp["sportsbook"]))),
            }
        )

    return pd.DataFrame(scored_rows).sort_values("EFD_score", ascending=False)


def detect_arbitrage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple cross-book arbitrage detector.

    For each fixture + market, compute best decimal price for each unique selection.
    If sum(1/best_price) < 1 → theoretical arbitrage.

    This works nicely for 2-way (moneyline, spread) and 3-way (soccer) markets.
    """
    if df.empty:
        return pd.DataFrame()

    arb_rows = []
    for (fixture_id, league, market), grp in df.groupby(["fixture_id", "league", "market"]):
        # Best odds by outcome
        outcome_best = (
            grp.groupby("selection")["price_decimal"]
            .max()
            .reset_index()
        )

        if outcome_best.empty:
            continue

        inv_sum = (1.0 / outcome_best["price_decimal"]).sum()
        if inv_sum < 1.0:
            profit_pct = (1.0 - inv_sum) * 100.0
            arb_rows.append(
                {
                    "fixture_id": fixture_id,
                    "league": league,
                    "market": market,
                    "num_outcomes": len(outcome_best),
                    "inv_sum": inv_sum,
                    "arb_yield_pct": profit_pct,
                    "outcomes": "; ".join(
                        f'{row.selection}@{row.price_decimal:.2f}'
                        for _, row in outcome_best.iterrows()
                    ),
                }
            )

    return pd.DataFrame(arb_rows).sort_values("arb_yield_pct", ascending=False)


def compute_line_moves(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Use all historical snapshots in session_state to compute open vs current odds.
    """
    if history_df.empty:
        return pd.DataFrame()

    history_df = history_df.copy()
    history_df["ingest_ts"] = pd.to_datetime(history_df["ingest_ts"])

    records = []
    for (fixture_id, league, market, selection, sportsbook), grp in history_df.groupby(
        ["fixture_id", "league", "market", "selection", "sportsbook"]
    ):
        grp_sorted = grp.sort_values("ingest_ts")
        open_price = float(grp_sorted.iloc[0]["price_decimal"])
        curr_price = float(grp_sorted.iloc[-1]["price_decimal"])
        move = curr_price - open_price

        records.append(
            {
                "fixture_id": fixture_id,
                "league": league,
                "market": market,
                "selection": selection,
                "sportsbook": sportsbook,
                "open_odds": open_price,
                "current_odds": curr_price,
                "line_move": move,
                "move_abs": abs(move),
                "direction": (
                    "Steam to dog" if move > 0
                    else ("Steam to fav" if move < 0 else "Flat")
                ),
            }
        )

    return pd.DataFrame(records).sort_values("move_abs", ascending=False)


# ======================================
#  STREAMLIT LAYOUT & STATE
# ======================================

st.set_page_config(
    page_title="EDGE FORCE DOMINION – OpticOdds Live",
    page_icon="🏆",
    layout="wide",
)

# --- Neon-dark theme styling
st.markdown(
    """
    <style>
    body {
        background-color: #050816;
        color: #f8fafc;
    }
    .stApp {
        background: radial-gradient(circle at top, #172554 0, #020617 55%, #000 100%);
    }
    .efd-card {
        border-radius: 16px;
        padding: 1rem 1.25rem;
        background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(15,118,110,0.35));
        box-shadow: 0 0 30px rgba(34,211,238,0.25);
        border: 1px solid rgba(56,189,248,0.45);
    }
    .efd-header {
        font-size: 1.4rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Session state
if "odds_history" not in st.session_state:
    st.session_state.odds_history = pd.DataFrame()

if "last_snapshot_ts" not in st.session_state:
    st.session_state.last_snapshot_ts = None

# ======================================
#  SIDEBAR CONFIG
# ======================================

st.sidebar.title("⚙️ Configuration")

api_key = st.sidebar.text_input("OpticOdds API Key", type="password")

selected_sports_labels = st.sidebar.multiselect(
    "Sports",
    list(SPORT_KEYS.keys()),
    default=list(SPORT_KEYS.keys()),
)

selected_books = st.sidebar.multiselect(
    "Sportsbooks",
    DEFAULT_BOOKS,
    default=DEFAULT_BOOKS,
)

selected_markets = st.sidebar.multiselect(
    "Markets",
    DEFAULT_MARKETS,
    default=["moneyline"],
)

max_events = st.sidebar.slider(
    "Max fixtures per sport (snapshot)",
    min_value=10, max_value=200, value=60, step=10,
)

include_live = st.sidebar.checkbox("Include live games", value=True)

run_snapshot = st.sidebar.button("Run Snapshot Pull Now")

st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("Auto-refresh every 30s", value=False)
if auto_refresh:
    st.sidebar.write("App will re-run itself every ~30 seconds.")


# ======================================
#  HEADER
# ======================================

st.markdown(
    """
    <div class="efd-card">
      <div class="efd-header">🏆 EDGE FORCE DOMINION – OpticOdds Live Engine</div>
      <div>Multi-sport odds ingestion, EFD scoring, arbitrage radar, and line-move tracking over your OpticOdds feed.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")


# ======================================
#  DATA PULL
# ======================================

if auto_refresh:
    # this triggers a silent periodic rerun
    st.experimental_rerun()

odds_all = st.session_state.odds_history.copy()

if api_key and run_snapshot:
    total_new_rows = 0
    all_fixture_ids = []

    # 1) fixtures per selected sport
    for label in selected_sports_labels:
        sport_slug = SPORT_KEYS[label]
        status = "live" if include_live else "upcoming"
        fixtures = fetch_fixtures(api_key, sport_slug, max_events, event_status=status)
        if not fixtures:
            continue

        fixture_ids = [f.get("id") for f in fixtures if f.get("id")]
        all_fixture_ids.extend(fixture_ids)

    all_fixture_ids = list(dict.fromkeys(all_fixture_ids))

    # 2) odds snapshot for those fixtures
    snapshot_rows = fetch_odds_snapshot(
        api_key=api_key,
        fixture_ids=all_fixture_ids,
        sportsbooks=selected_books,
        markets=selected_markets,
    )

    if snapshot_rows:
        new_df = pd.DataFrame(snapshot_rows)
        total_new_rows = len(new_df)
        st.session_state.odds_history = pd.concat(
            [st.session_state.odds_history, new_df], ignore_index=True
        )
        st.session_state.last_snapshot_ts = now_iso()
        odds_all = st.session_state.odds_history.copy()

    st.success(f"Snapshot complete. New rows ingested: {total_new_rows}")

# --- derived views
odds_all = ensure_df(odds_all)
unique_sports = sorted(odds_all["sport"].dropna().unique()) if not odds_all.empty else []
unique_leagues = sorted(odds_all["league"].dropna().unique()) if not odds_all.empty else []


# ======================================
#  TOP-LEVEL TABS
# ======================================

tab_dash, tab_arb, tab_moves, tab_analytics = st.tabs(
    ["🏠 Dashboard", "💰 Arbitrage", "📈 Line Moves", "📊 Analytics"]
)

# ------------------ DASHBOARD ------------------ #

with tab_dash:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Sports w/ data",
            len(unique_sports),
        )
    with col2:
        st.metric(
            "Leagues detected",
            len(unique_leagues),
        )
    with col3:
        st.metric(
            "Total odds rows",
            len(odds_all),
        )
    with col4:
        st.metric(
            "Last snapshot",
            st.session_state.last_snapshot_ts or "–",
        )

    st.write("")

    if odds_all.empty:
        st.info("No odds data yet. Enter your key and click **Run Snapshot Pull Now**.")
    else:
        # Sport / league filters
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            sport_filter = st.multiselect(
                "Filter by sport",
                options=unique_sports,
                default=unique_sports,
            )
        with fcol2:
            league_filter = st.multiselect(
                "Filter by league",
                options=unique_leagues,
                default=unique_leagues,
            )

        view_df = odds_all.copy()
        if sport_filter:
            view_df = view_df[view_df["sport"].isin(sport_filter)]
        if league_filter:
            view_df = view_df[view_df["league"].isin(league_filter)]

        efd_df = compute_efd_scores(view_df)

        st.markdown("### 🔬 Top EFD Edges (selection-level)")
        if efd_df.empty:
            st.info("No edges yet – pull another snapshot.")
        else:
            merged = efd_df.merge(
                view_df[
                    ["fixture_id", "home_team", "away_team", "start_time"]
                ].drop_duplicates(),
                on="fixture_id",
                how="left",
            )
            display_cols = [
                "EFD_score",
                "league",
                "market",
                "selection",
                "home_team",
                "away_team",
                "start_time",
                "best_price",
                "worst_price",
                "books",
            ]
            st.dataframe(
                merged[display_cols].sort_values("EFD_score", ascending=False).head(50),
                use_container_width=True,
            )

# ------------------ ARBITRAGE ------------------ #

with tab_arb:
    if odds_all.empty:
        st.info("No odds yet. Run a snapshot first.")
    else:
        arb_df = detect_arbitrage(odds_all)
        st.markdown("### 💰 Live Arbitrage Opportunities (across all ingested data)")

        if arb_df.empty:
            st.info("No pure arbitrage detected yet.")
        else:
            show_cols = [
                "arb_yield_pct",
                "league",
                "market",
                "fixture_id",
                "num_outcomes",
                "outcomes",
            ]
            st.dataframe(
                arb_df[show_cols].head(100),
                use_container_width=True,
            )

# ------------------ LINE MOVES ------------------ #

with tab_moves:
    if odds_all.empty:
        st.info("No odds yet. Run a snapshot first.")
    else:
        moves_df = compute_line_moves(odds_all)
        st.markdown("### 📈 Line Movement (open vs current – this session)")

        if moves_df.empty:
            st.info("No movement yet – pull multiple snapshots over time.")
        else:
            st.dataframe(
                moves_df[
                    [
                        "league",
                        "market",
                        "selection",
                        "sportsbook",
                        "open_odds",
                        "current_odds",
                        "line_move",
                        "direction",
                    ]
                ].head(200),
                use_container_width=True,
            )

# ------------------ ANALYTICS ------------------ #

with tab_analytics:
    st.markdown("### 📊 Session-level analytics")

    if odds_all.empty:
        st.info("No data yet.")
    else:
        # Simple summary by league & market
        summary = (
            odds_all.groupby(["league", "market"])
            .agg(
                num_rows=("fixture_id", "count"),
                num_fixtures=("fixture_id", "nunique"),
                num_books=("sportsbook", "nunique"),
            )
            .reset_index()
            .sort_values("num_rows", ascending=False)
        )
        st.dataframe(summary, use_container_width=True)

        st.write("-----")
        st.caption(
            "This v1 is poll-based (snapshots). To get more movement/arbs, keep running "
            "snapshots while games and markets are active."
        )