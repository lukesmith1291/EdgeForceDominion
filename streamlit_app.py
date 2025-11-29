import os
import time
import math
import json
from typing import List, Dict, Any, Tuple

import requests
import pandas as pd
import streamlit as st

# =========================
#  BASIC CONFIG
# =========================

BASE_URL = "https://api.opticodds.com/api/v3"

US_SPORTS = {
    "NBA": "basketball_nba",
    "NCAAB": "basketball_ncaab",
    "NFL": "americanfootball_nfl",
    "NCAAF": "americanfootball_ncaaf",
    "MLB": "baseball_mlb",
    "NHL": "icehockey_nhl",
}

DEFAULT_SPORTBOOKS = [
    "FanDuel",
    "DraftKings",
    "BetMGM",
    "Caesars",
    "Pinnacle",
    "LowVig",
]

DEFAULT_MARKETS = [
    "moneyline",
    "point_spread",
    "total_points",
]

SESSION_KEY_SNAPSHOT = "efd_snapshot_df"
SESSION_KEY_FIXTURES = "efd_fixtures_df"

# =========================
#  HELPERS
# =========================

def american_to_decimal(odds: Any) -> float:
    """Convert American odds to decimal, safe against junk."""
    try:
        o = float(odds)
    except Exception:
        return 1.0
    if o == 0:
        return 1.0
    if o > 0:
        return 1.0 + (o / 100.0)
    else:
        return 1.0 + (100.0 / abs(o))


def ev_from_implied_vs_offer(implied_prob: float, price_dec: float) -> float:
    """Expected value given implied 'true' prob and offered decimal price."""
    offer_prob = 1.0 / price_dec if price_dec != 0 else 0.0
    edge = (implied_prob * price_dec) - 1.0
    return edge


def safe_get(url: str, params: Dict[str, Any], timeout: int = 15) -> Dict[str, Any]:
    """GET with basic error handling – returns JSON or raises."""
    resp = requests.get(url, params=params, timeout=timeout)
    if not resp.ok:
        raise RuntimeError(f"{resp.status_code} {resp.text}")
    return resp.json()


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


# =========================
#  OPTICODDS – SNAPSHOT
# =========================

def fetch_active_fixtures(
    api_key: str,
    sports: List[str],
    max_per_sport: int = 50,
) -> pd.DataFrame:
    """
    Hit /fixtures/active for each selected sport.
    Returns a combined fixtures DataFrame.
    """
    all_rows = []

    for label, sport_code in US_SPORTS.items():
        if label not in sports:
            continue

        try:
            params = {
                "key": api_key,
                "sport": sport_code,
                "event_status": "upcoming",  # upcoming / live / closed
            }
            data = safe_get(f"{BASE_URL}/fixtures/active", params=params)
        except Exception as e:
            st.error(f"{label}: error fetching fixtures – {e}")
            continue

        fixtures = data.get("fixtures", data)  # depending on API shape
        if not fixtures:
            continue

        for fx in fixtures[:max_per_sport]:
            all_rows.append(
                {
                    "sport_label": label,
                    "sport_code": sport_code,
                    "fixture_id": fx.get("fixture_id"),
                    "home_team": fx.get("home_team"),
                    "away_team": fx.get("away_team"),
                    "commence_time": fx.get("commence_time"),
                    "league": fx.get("league"),
                }
            )

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame(all_rows)


def fetch_odds_for_fixtures(
    api_key: str,
    sport_label: str,
    sport_code: str,
    fixture_ids: List[str],
    sportsbooks: List[str],
    markets: List[str],
    chunk_size: int = 10,
) -> List[Dict[str, Any]]:
    """
    For a list of fixture_ids, call /fixtures/odds in chunks.
    Returns list of rows, one row per (fixture, book, market, selection).
    """
    all_rows: List[Dict[str, Any]] = []

    for fixture_chunk in chunk_list(fixture_ids, chunk_size):
        params = {
            "key": api_key,
            "sport": sport_code,
            "fixture_ids": ",".join(fixture_chunk),
            "sportsbooks": ",".join(sportsbooks),
            "markets": ",".join(markets),
            "is_main": "True",
        }
        try:
            data = safe_get(f"{BASE_URL}/fixtures/odds", params=params)
        except Exception as e:
            st.error(
                f"{sport_label}: snapshot odds error (fixtures {fixture_chunk[0]}..): {e}"
            )
            continue

        # Shape is typically list of fixtures, each with markets → outcomes
        fixtures = data.get("fixtures", data)
        for fx in fixtures:
            fx_id = fx.get("fixture_id")
            league = fx.get("league")
            home = fx.get("home_team")
            away = fx.get("away_team")
            commence = fx.get("commence_time")

            books = fx.get("sportsbooks", [])
            for book in books:
                book_name = book.get("key") or book.get("sportsbook") or "Unknown"

                mkts = book.get("markets", [])
                for mkt in mkts:
                    mkt_key = mkt.get("key") or mkt.get("market")
                    outcomes = mkt.get("outcomes", [])

                    for out in outcomes:
                        sel_name = out.get("name")
                        price_american = out.get("price")
                        price_dec = american_to_decimal(price_american)

                        all_rows.append(
                            {
                                "sport_label": sport_label,
                                "sport_code": sport_code,
                                "fixture_id": fx_id,
                                "league": league,
                                "home_team": home,
                                "away_team": away,
                                "commence_time": commence,
                                "sportsbook": book_name,
                                "market": mkt_key,
                                "selection": sel_name,
                                "price_american": price_american,
                                "price_decimal": price_dec,
                            }
                        )

    return all_rows


def build_snapshot(
    api_key: str,
    sports: List[str],
    sportsbooks: List[str],
    markets: List[str],
    max_per_sport: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full snapshot:
    - fixtures_df: every fixture we found
    - odds_df: every odds row (fixture/book/market/selection)
    """
    fixtures_df = fetch_active_fixtures(api_key, sports, max_per_sport=max_per_sport)
    if fixtures_df.empty:
        return fixtures_df, pd.DataFrame()

    odds_rows: List[Dict[str, Any]] = []

    for sport_label in fixtures_df["sport_label"].unique():
        sport_fixture_df = fixtures_df[fixtures_df["sport_label"] == sport_label]
        sport_code = sport_fixture_df["sport_code"].iloc[0]

        fixture_ids = sport_fixture_df["fixture_id"].dropna().unique().tolist()
        if not fixture_ids:
            continue

        rows = fetch_odds_for_fixtures(
            api_key=api_key,
            sport_label=sport_label,
            sport_code=sport_code,
            fixture_ids=fixture_ids,
            sportsbooks=sportsbooks,
            markets=markets,
        )
        odds_rows.extend(rows)

    odds_df = pd.DataFrame(odds_rows)
    return fixtures_df, odds_df


# =========================
#  EDGE FORCE – SCORING
# =========================

def compute_efd_scores(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Very simple v1 EFD scoring:
    - For each fixture & side:
        * compute avg price across books (decimal).
        * compute implied "consensus" probability.
        * compute book dispersion (max-min price across books).
        * EFD_score = combo of advantage vs worst line & dispersion.
    """
    if odds_df.empty:
        return pd.DataFrame()

    # Moneyline only for EFD v1
    df = odds_df[odds_df["market"] == "moneyline"].copy()
    if df.empty:
        return pd.DataFrame()

    # Basic aggregation per fixture & selection
    grouped = df.groupby(["fixture_id", "sport_label", "selection", "home_team", "away_team"])

    rows = []
    for (fixture_id, sport_label, sel, home, away), grp in grouped:
        prices = grp["price_decimal"].astype(float)
        if prices.empty:
            continue

        avg_price = prices.mean()
        best_price = prices.max()
        worst_price = prices.min()
        implied_consensus = 1.0 / avg_price if avg_price != 0 else 0.0

        # "Value" – how much better the best price is vs consensus expectation
        ev_edge = ev_from_implied_vs_offer(implied_consensus, best_price)

        dispersion = best_price - worst_price

        # Map to 0–100 (rough heuristic)
        base_score = (ev_edge * 1000.0) + (dispersion * 10.0)
        base_score = max(-20.0, min(base_score, 80.0))
        efd_score = 50.0 + base_score

        rows.append(
            {
                "fixture_id": fixture_id,
                "sport_label": sport_label,
                "home_team": home,
                "away_team": away,
                "selection": sel,
                "avg_price": round(avg_price, 3),
                "best_price": round(best_price, 3),
                "worst_price": round(worst_price, 3),
                "implied_prob": round(implied_consensus, 3),
                "ev_edge": round(ev_edge, 4),
                "dispersion": round(dispersion, 3),
                "EFD_score": round(efd_score, 1),
            }
        )

    scored = pd.DataFrame(rows)
    if scored.empty:
        return scored

    return scored.sort_values("EFD_score", ascending=False)


# =========================
#  ARBITRAGE ENGINE
# =========================

def detect_arbitrage(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect crude multi-book arb on 2-way & 3-way moneylines.
    """
    if odds_df.empty:
        return pd.DataFrame()

    df = odds_df[odds_df["market"] == "moneyline"].copy()
    if df.empty:
        return pd.DataFrame()

    # We assume selection names like "Home", "Away", "Draw" or team names.
    rows = []
    for fixture_id, grp in df.groupby("fixture_id"):
        # map selection -> best (highest) decimal price & book
        sel_best = {}
        for sel, sgrp in grp.groupby("selection"):
            idx = sgrp["price_decimal"].astype(float).idxmax()
            row = sgrp.loc[idx]
            sel_best[sel] = (row["price_decimal"], row["sportsbook"])

        if len(sel_best) < 2:
            continue

        inv_sum = sum(1.0 / price for price, _ in sel_best.values() if price > 0)
        if inv_sum <= 0:
            continue

        arb_pct = 1.0 - inv_sum
        if arb_pct <= 0:
            continue  # not an arb

        # Build human-friendly description
        legs = []
        for sel, (price, book) in sel_best.items():
            legs.append(f"{sel} @ {book} ({round(price,3)})")

        any_row = grp.iloc[0]
        sport_label = any_row["sport_label"]
        home_team = any_row["home_team"]
        away_team = any_row["away_team"]

        rows.append(
            {
                "fixture_id": fixture_id,
                "sport_label": sport_label,
                "matchup": f"{away_team} @ {home_team}",
                "legs": " | ".join(legs),
                "edge_percent": round(arb_pct * 100.0, 2),
            }
        )

    arb_df = pd.DataFrame(rows)
    if arb_df.empty:
        return arb_df

    return arb_df.sort_values("edge_percent", ascending=False)


# =========================
#  UI – THEME & LAYOUT
# =========================

def init_session():
    if SESSION_KEY_SNAPSHOT not in st.session_state:
        st.session_state[SESSION_KEY_SNAPSHOT] = pd.DataFrame()
    if SESSION_KEY_FIXTURES not in st.session_state:
        st.session_state[SESSION_KEY_FIXTURES] = pd.DataFrame()


def sidebar_controls() -> Dict[str, Any]:
    st.sidebar.title("⚙️ Edge Force Config")

    api_key = st.sidebar.text_input(
        "OpticOdds API key",
        value=os.getenv("OPTICODDS_API_KEY", ""),
        type="password",
        help="Paste the key Abe sent you.",
    )

    st.sidebar.markdown("---")

    # Sports
    sport_labels = list(US_SPORTS.keys())
    sports_choice = st.sidebar.multiselect(
        "Sports",
        options=["ALL"] + sport_labels,
        default=["ALL"],
        help="Choose which US sports you want to ingest.",
    )
    if "ALL" in sports_choice:
        sports_selected = sport_labels
    else:
        sports_selected = sports_choice or sport_labels

    # Sportsbooks
    books_choice = st.sidebar.multiselect(
        "Sportsbooks",
        options=["ALL"] + DEFAULT_SPORTBOOKS,
        default=["ALL"],
        help="Books to request from OpticOdds.",
    )
    if "ALL" in books_choice or not books_choice:
        books_selected = DEFAULT_SPORTBOOKS
    else:
        books_selected = books_choice

    # Markets
    markets_choice = st.sidebar.multiselect(
        "Markets",
        options=["ALL"] + DEFAULT_MARKETS,
        default=["ALL"],
        help="You can start with Moneyline only if you want it lean.",
    )
    if "ALL" in markets_choice or not markets_choice:
        markets_selected = DEFAULT_MARKETS
    else:
        markets_selected = markets_choice

    max_per_sport = st.sidebar.slider(
        "Max fixtures per sport (snapshot)",
        min_value=5,
        max_value=100,
        value=40,
        step=5,
    )

    st.sidebar.markdown("---")
    run_snapshot = st.sidebar.button("🚀 Run Snapshot", use_container_width=True)

    return {
        "api_key": api_key,
        "sports": sports_selected,
        "sports_display": sports_choice,
        "books": books_selected,
        "markets": markets_selected,
        "max_per_sport": max_per_sport,
        "run_snapshot": run_snapshot,
    }


# =========================
#  MAIN APP
# =========================

def main():
    st.set_page_config(
        page_title="EDGE FORCE Dominion – OpticOdds",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session()

    # Simple "neon dark" vibe using markdown + CSS
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top, #0b1535 0, #050810 45%, #000000 100%);
            color: #f2f4ff;
        }
        .efd-card {
            border-radius: 18px;
            padding: 18px 22px;
            background: linear-gradient(135deg,#0b1023,#050814);
            border: 1px solid rgba(0,210,255,0.35);
            box-shadow: 0 0 25px rgba(0,210,255,0.18);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="efd-card">
          <h1>🏆 EDGE FORCE Dominion — OpticOdds Live Engine</h1>
          <p>Multi-sport odds ingestion, EFD scoring, and arbitrage radar over your OpticOdds feed.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cfg = sidebar_controls()
    api_key = cfg["api_key"]

    if not api_key:
        st.warning("Paste your OpticOdds API key in the sidebar to start.")
        return

    # Run snapshot when user clicks button
    if cfg["run_snapshot"]:
        with st.spinner("Pulling fixtures and odds snapshot from OpticOdds…"):
            fixtures_df, odds_df = build_snapshot(
                api_key=api_key,
                sports=cfg["sports"],
                sportsbooks=cfg["books"],
                markets=cfg["markets"],
                max_per_sport=cfg["max_per_sport"],
            )
            st.session_state[SESSION_KEY_FIXTURES] = fixtures_df
            st.session_state[SESSION_KEY_SNAPSHOT] = odds_df

    fixtures_df: pd.DataFrame = st.session_state[SESSION_KEY_FIXTURES]
    odds_df: pd.DataFrame = st.session_state[SESSION_KEY_SNAPSHOT]

    tabs = st.tabs(["🏠 Dashboard", "💰 Arbitrage", "📈 Line Moves", "📊 Analytics"])

    # --------------- Dashboard ---------------
    with tabs[0]:
        st.subheader("Live Snapshot")

        if odds_df.empty:
            st.info("No odds data yet. Click **Run Snapshot** in the sidebar.")
        else:
            # Small KPIs
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Active Sports (snapshot)", len(odds_df["sport_label"].unique()))
            with col2:
                st.metric("Tracked Sportsbooks", len(odds_df["sportsbook"].unique()))
            with col3:
                st.metric("Rows ingested", len(odds_df))

            # Filters
            colf1, colf2, colf3 = st.columns(3)
            with colf1:
                sport_filter = st.selectbox(
                    "Filter by sport",
                    options=["ALL"] + sorted(odds_df["sport_label"].unique().tolist()),
                    index=0,
                )
            with colf2:
                book_filter = st.selectbox(
                    "Filter by book",
                    options=["ALL"] + sorted(odds_df["sportsbook"].unique().tolist()),
                    index=0,
                )
            with colf3:
                market_filter = st.selectbox(
                    "Filter by market",
                    options=["ALL"] + sorted(odds_df["market"].unique().tolist()),
                    index=0,
                )

            df_view = odds_df.copy()
            if sport_filter != "ALL":
                df_view = df_view[df_view["sport_label"] == sport_filter]
            if book_filter != "ALL":
                df_view = df_view[df_view["sportsbook"] == book_filter]
            if market_filter != "ALL":
                df_view = df_view[df_view["market"] == market_filter]

            # Build matchup label
            df_view["matchup"] = df_view["away_team"].fillna("") + " @ " + df_view[
                "home_team"
            ].fillna("")

            st.markdown("### Snapshot Odds")
            st.dataframe(
                df_view[
                    [
                        "sport_label",
                        "league",
                        "matchup",
                        "commence_time",
                        "sportsbook",
                        "market",
                        "selection",
                        "price_american",
                        "price_decimal",
                    ]
                ].sort_values(["sport_label", "commence_time"]),
                use_container_width=True,
                height=450,
            )

            # Game detail selector
            st.markdown("---")
            st.markdown("#### Game Detail")
            unique_games = (
                df_view.groupby("fixture_id")
                .agg(
                    matchup=("matchup", "first"),
                    sport_label=("sport_label", "first"),
                )
                .reset_index()
            )
            if not unique_games.empty:
                fixture_ids_list = unique_games["fixture_id"].tolist()
                labels = [
                    f"{row['sport_label']} – {row['matchup']} ({row['fixture_id']})"
                    for _, row in unique_games.iterrows()
                ]
                selected = st.selectbox(
                    "Pick a matchup",
                    options=["None"] + labels,
                    index=0,
                )
                if selected != "None":
                    sel_idx = labels.index(selected)
                    sel_fixture_id = fixture_ids_list[sel_idx]
                    gdf = df_view[df_view["fixture_id"] == sel_fixture_id].copy()

                    st.markdown(f"##### {selected}")
                    st.dataframe(
                        gdf[
                            [
                                "sportsbook",
                                "market",
                                "selection",
                                "price_american",
                                "price_decimal",
                            ]
                        ].sort_values(["market", "sportsbook"]),
                        use_container_width=True,
                    )

    # --------------- Arbitrage ---------------
    with tabs[1]:
        st.subheader("Arbitrage Radar")

        if odds_df.empty:
            st.info("No odds snapshot yet.")
        else:
            arb_df = detect_arbitrage(odds_df)
            if arb_df.empty:
                st.info("No clear arbitrage edges detected in this snapshot.")
            else:
                st.dataframe(
                    arb_df[
                        [
                            "sport_label",
                            "matchup",
                            "edge_percent",
                            "legs",
                        ]
                    ],
                    use_container_width=True,
                    height=450,
                )

    # --------------- Line Moves (placeholder) ---------------
    with tabs[2]:
        st.subheader("Line Movement (Session)")

        st.info(
            "This v1 focuses on snapshot odds. "
            "Next step is wiring the OpticOdds /stream/odds/{sport} endpoint "
            "into this structure so we can track open vs current for each side."
        )

    # --------------- Analytics ---------------
    with tabs[3]:
        st.subheader("EFD Scoring & Analytics")

        if odds_df.empty:
            st.info("No odds snapshot yet.")
        else:
            efd_df = compute_efd_scores(odds_df)
            if efd_df.empty:
                st.info("No EFD scores yet (check that moneyline data is present).")
            else:
                st.markdown("### Top EFD Edges (Moneyline Only)")
                efd_df["matchup"] = (
                    efd_df["away_team"].fillna("") + " @ " + efd_df["home_team"].fillna("")
                )
                st.dataframe(
                    efd_df[
                        [
                            "sport_label",
                            "matchup",
                            "selection",
                            "EFD_score",
                            "best_price",
                            "avg_price",
                            "implied_prob",
                            "ev_edge",
                            "dispersion",
                        ]
                    ].sort_values("EFD_score", ascending=False),
                    use_container_width=True,
                    height=450,
                )


if __name__ == "__main__":
    main()