# streamlit_app.py
# EDGE FORCE DOMINION – OpticOdds Live Engine (v1)

import time
import json
import itertools
from datetime import datetime, timezone

import requests
import pandas as pd
import numpy as np
import streamlit as st


# ==============================
#  BASIC CONFIG
# ==============================

OPTIC_BASE = "https://api.opticodds.com/api/v3"

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
    """Safe American → decimal conversion."""
    try:
        o = float(odds)
    except Exception:
        return 1.0
    if o == 0:
        return 1.0
    if o > 0:
        return 1.0 + (o / 100.0)
    return 1.0 + (100.0 / abs(o))


def now_utc_iso():
    return datetime.now(timezone.utc).isoformat()


# ==============================
#  OPTICODDS – SNAPSHOT HELPERS
# ==============================

def fetch_fixtures_for_sport(api_key: str, sport_key: str, max_events: int = 50):
    """
    Get upcoming fixtures for a sport.
    This uses: /fixtures?sport=nba&event_status=upcoming
    """
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
        st.error(f"{sport_key.upper()}: failed to pull fixtures: {e}")
        return []

    data = r.json().get("data", [])
    fixture_ids = []
    for f in data:
        fid = f.get("id")
        if fid:
            fixture_ids.append(fid)
    return fixture_ids


def fetch_odds_for_fixtures(api_key, fixture_ids, sportsbooks, markets):
    """
    AUTOMATED AGGREGATOR
    Respects typical OpticOdds limits: <=5 fixture_ids & <=5 sportsbooks per call.
    Returns list of flat odds rows.
    """
    if not fixture_ids or not sportsbooks or not markets:
        return []

    all_rows = []
    fixture_ids = list(dict.fromkeys(fixture_ids))
    sportsbooks = list(dict.fromkeys(sportsbooks))

    for fixture_chunk in _chunk_list(fixture_ids, 5):
        for book_chunk in _chunk_list(sportsbooks, 5):
            params = [("key", api_key)]
            for fid in fixture_chunk:
                params.append(("fixture_id", fid))
            for sb in book_chunk:
                params.append(("sportsbook", sb))
            for m in markets:
                params.append(("market", m))
            params.append(("is_main", "True"))

            try:
                r = requests.get(f"{OPTIC_BASE}/fixtures/odds",
                                 params=params, timeout=10)
                r.raise_for_status()
                payload = r.json().get("data", [])
            except Exception as e:
                st.error(
                    f"Failed to pull odds snapshot "
                    f"(fixtures={len(fixture_chunk)}, books={len(book_chunk)}): {e}"
                )
                continue

            for f in payload:
                fixture_id = f.get("id")
                sport = f.get("sport")
                league = f.get("league")
                home = f.get("home_team")
                away = f.get("away_team")
                start_time = f.get("start_time")

                for o in f.get("odds", []):
                    row = {
                        "source": "snapshot",
                        "timestamp": now_utc_iso(),
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
                        "price_decimal": o.get("price_decimal")
                            or american_to_decimal(o.get("price_american")),
                    }
                    all_rows.append(row)

    return all_rows


# ==============================
#  OPTICODDS – STREAM (SSE)
# ==============================

def parse_sse_stream(resp, max_messages=200):
    """
    Simple SSE parser. Yields JSON payloads for 'data:' events.
    Stops after `max_messages` messages (defensive).
    """
    message_buf = []
    count = 0

    for raw_line in resp.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        line = raw_line.strip()
        if not line:
            # end of one SSE event
            if not message_buf:
                continue
            # gather data: lines
            data_lines = [l[5:].strip() for l in message_buf if l.startswith("data:")]
            if data_lines:
                try:
                    payload_str = "\n".join(data_lines)
                    payload = json.loads(payload_str)
                    yield payload
                    count += 1
                    if count >= max_messages:
                        return
                except Exception:
                    pass
            message_buf = []
        else:
            message_buf.append(line)


def stream_burst_for_sport(api_key, sport_key, sportsbooks, markets,
                           is_live: bool, max_events: int, max_messages: int = 200):
    """
    Connects briefly to /stream/odds/{sport} and returns flat odds rows.
    This is a "burst", not an infinite loop.
    """
    markets = list(dict.fromkeys(markets))
    sportsbooks = list(dict.fromkeys(sportsbooks))

    params = [("key", api_key)]
    for sb in sportsbooks:
        params.append(("sportsbook", sb))
    for m in markets:
        params.append(("market", m))

    params.append(("is_main", "True"))

    # game type filter
    if is_live is True:
        params.append(("event_status", "live"))
    elif is_live is False:
        params.append(("event_status", "upcoming"))

    url = f"{OPTIC_BASE}/stream/odds/{sport_key}"

    try:
        with requests.get(url, params=params, stream=True, timeout=25) as resp:
            resp.raise_for_status()
            rows = []
            for msg in parse_sse_stream(resp, max_messages=max_messages):
                # OpticOdds stream payload is usually similar to fixtures/odds.
                # We'll be defensive and handle a couple of shapes.
                data_fixtures = msg.get("data") or msg.get("fixtures") or []
                if not isinstance(data_fixtures, list):
                    data_fixtures = [data_fixtures]

                for f in data_fixtures:
                    fixture_id = f.get("id") or f.get("fixture_id")
                    sport = f.get("sport")
                    league = f.get("league")
                    home = f.get("home_team")
                    away = f.get("away_team")
                    start_time = f.get("start_time")

                    odds_list = f.get("odds", [])
                    for o in odds_list:
                        row = {
                            "source": "stream",
                            "timestamp": now_utc_iso(),
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
                            "price_decimal": o.get("price_decimal")
                                or american_to_decimal(o.get("price_american")),
                        }
                        rows.append(row)

                        if len(rows) >= max_events * len(sportsbooks) * len(markets):
                            return rows
            return rows
    except Exception as e:
        st.error(f"{sport_key.upper()}: error opening stream: {e}")
        return []


# ==============================
#  CORE ENGINES
# ==============================

def merge_into_history(history_df: pd.DataFrame, new_rows: list) -> pd.DataFrame:
    """Append new_rows to history_df (per session)."""
    if not new_rows:
        return history_df
    add = pd.DataFrame(new_rows)
    if history_df is None or history_df.empty:
        return add
    return pd.concat([history_df, add], ignore_index=True)


def compute_line_moves(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (fixture, sportsbook, selection, market), compare OPEN vs CURRENT.
    Returns one row per combo with line_move.
    """
    if history_df is None or history_df.empty:
        return pd.DataFrame()

    # Drop obvious junk
    df = history_df.dropna(subset=["fixture_id", "sportsbook", "selection", "market"])
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    # ensure numeric
    df["price_decimal"] = df["price_decimal"].astype(float)

    agg_rows = []
    group_cols = ["fixture_id", "sportsbook", "selection", "market"]

    for key, grp in df.groupby(group_cols):
        grp = grp.sort_values("timestamp")
        first = grp.iloc[0]
        last = grp.iloc[-1]

        open_price = float(first["price_decimal"])
        current_price = float(last["price_decimal"])
        move = current_price - open_price

        fixture_id, sportsbook, selection, market = key
        agg_rows.append({
            "fixture_id": fixture_id,
            "sportsbook": sportsbook,
            "selection": selection,
            "market": market,
            "sport": last.get("sport"),
            "league": last.get("league"),
            "home_team": last.get("home_team"),
            "away_team": last.get("away_team"),
            "start_time": last.get("start_time"),
            "open_odds": open_price,
            "current_odds": current_price,
            "line_move": move,
            "abs_move": abs(move),
            "move_direction": (
                "Steam to dog" if move > 0 else
                ("Steam to fav" if move < 0 else "Flat")
            ),
        })

    if not agg_rows:
        return pd.DataFrame()

    res = pd.DataFrame(agg_rows)
    res = res.sort_values("abs_move", ascending=False).reset_index(drop=True)
    return res


def compute_efd_scores(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple EFD-style score per fixture/outcome:
    - For each fixture + selection:
      * best_decimal = max price over all books
      * fair_decimal = average over all books
      * edge_pct = (best_decimal - fair_decimal) / fair_decimal
      * EFD_score = clip(edge_pct * 100, 0, 100)
    """
    if odds_df is None or odds_df.empty:
        return pd.DataFrame()

    df = odds_df.copy()
    df["price_decimal"] = df["price_decimal"].astype(float)

    agg_rows = []
    group_cols = ["fixture_id", "sport", "league",
                  "home_team", "away_team", "selection"]

    for key, grp in df.groupby(group_cols):
        best = grp["price_decimal"].max()
        avg = grp["price_decimal"].mean()
        if avg <= 0:
            continue
        edge_pct = (best - avg) / avg
        efd_score = float(np.clip(edge_pct * 100.0, 0, 100))

        fixture_id, sport, league, home, away, sel = key
        agg_rows.append({
            "fixture_id": fixture_id,
            "sport": sport,
            "league": league,
            "home_team": home,
            "away_team": away,
            "selection": sel,
            "best_decimal": best,
            "avg_decimal": avg,
            "edge_pct": edge_pct,
            "EFD_score": efd_score,
        })

    if not agg_rows:
        return pd.DataFrame()

    res = pd.DataFrame(agg_rows)
    res = res.sort_values("EFD_score", ascending=False).reset_index(drop=True)
    return res


def detect_arbitrage(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple arbitrage finder:
    - For each fixture + market:
      * for each selection: best decimal across books
      * sum implied probs; if < 1, arbitrage.
    """
    if odds_df is None or odds_df.empty:
        return pd.DataFrame()

    df = odds_df.dropna(subset=["fixture_id", "market", "selection"]).copy()
    df["price_decimal"] = df["price_decimal"].astype(float)

    arb_rows = []
    group_cols = ["fixture_id", "market", "sport", "league",
                  "home_team", "away_team"]

    for key, grp in df.groupby(group_cols):
        best_by_sel = grp.groupby("selection")["price_decimal"].max()
        if best_by_sel.empty:
            continue

        implied_probs = 1.0 / best_by_sel
        total_prob = implied_probs.sum()
        if total_prob < 1.0:
            profit_pct = (1.0 - total_prob) * 100.0
            fixture_id, market, sport, league, home, away = key
            arb_rows.append({
                "fixture_id": fixture_id,
                "sport": sport,
                "league": league,
                "home_team": home,
                "away_team": away,
                "market": market,
                "n_sides": len(best_by_sel),
                "total_implied_prob": total_prob,
                "arbitrage_edge_pct": profit_pct,
            })

    if not arb_rows:
        return pd.DataFrame()

    res = pd.DataFrame(arb_rows)
    res = res.sort_values("arbitrage_edge_pct", ascending=False).reset_index(drop=True)
    return res


# ==============================
#  STREAMLIT UI
# ==============================

def setup_state():
    if "odds_history" not in st.session_state:
        # odds_history[league_or_sport] = DataFrame of all rows
        st.session_state.odds_history = {}
    if "last_update" not in st.session_state:
        st.session_state.last_update = None
    if "last_snapshot_count" not in st.session_state:
        st.session_state.last_snapshot_count = 0
    if "last_burst_count" not in st.session_state:
        st.session_state.last_burst_count = 0


def main():
    st.set_page_config(
        page_title="Edge Force Dominion – Live Odds Engine",
        layout="wide",
        page_icon="🏆",
    )
    setup_state()

    # ------------ SIDEBAR CONFIG ------------
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        api_key = st.text_input(
            "OpticOdds API Key",
            type="password",
            help="Paste the key Abe sent you.",
        )

        all_sports_list = list(SPORT_MAP.keys())
        sport_choice = st.multiselect(
            "Sports",
            options=["All"] + all_sports_list,
            default=["NBA"],
            help="Choose which sports to pull. 'All' = all supported.",
        )
        if "All" in sport_choice or not sport_choice:
            effective_sports = all_sports_list
        else:
            effective_sports = sport_choice

        sportsbook_choice = st.multiselect(
            "Sportsbooks",
            options=DEFAULT_BOOKS,
            default=DEFAULT_BOOKS[:5],
        )
        if not sportsbook_choice:
            sportsbook_choice = DEFAULT_BOOKS[:3]

        market_choice = st.multiselect(
            "Markets",
            options=DEFAULT_MARKETS,
            default=["moneyline"],
        )
        if not market_choice:
            market_choice = ["moneyline"]

        game_type = st.selectbox(
            "Game Type",
            options=["All", "Pre-game only", "Live only"],
            index=0,
        )
        if game_type == "Pre-game only":
            is_live = False
        elif game_type == "Live only":
            is_live = True
        else:
            is_live = None

        max_events = st.slider(
            "Max fixtures per sport (snapshot)",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
        )

        max_stream_messages = st.slider(
            "Max stream messages per sport (burst)",
            min_value=20,
            max_value=400,
            value=150,
            step=10,
        )

        st.markdown("---")
        run_snapshot = st.button("📡 Run Snapshot + Burst")

    st.markdown(
        "<h1 style='color:#f5f5f5;text-shadow:0 0 18px #00e6ff;'>"
        "Edge Force Dominion – Live OpticOdds Engine"
        "</h1>",
        unsafe_allow_html=True,
    )
    st.caption("Live odds ingestion, EFD scoring, and arbitrage radar over OpticOdds.")

    if not api_key:
        st.warning("Paste your OpticOdds API key in the sidebar to start.")
        return

    # ------------ DATA PULL ------------
    if run_snapshot:
        snapshot_rows = []
        burst_rows = []

        with st.spinner("Pulling snapshots + short streams for selected sports..."):
            for sport_label in effective_sports:
                sport_key = SPORT_MAP[sport_label]

                # 1) Fixtures snapshot
                fixture_ids = fetch_fixtures_for_sport(api_key, sport_key, max_events)
                if not fixture_ids:
                    continue

                rows_snap = fetch_odds_for_fixtures(
                    api_key, fixture_ids, sportsbook_choice, market_choice
                )
                snapshot_rows.extend(rows_snap)

                # 2) Short live stream burst
                rows_stream = stream_burst_for_sport(
                    api_key,
                    sport_key,
                    sportsbook_choice,
                    market_choice,
                    is_live=is_live,
                    max_events=max_events,
                    max_messages=max_stream_messages,
                )
                burst_rows.extend(rows_stream)

        total_rows = snapshot_rows + burst_rows

        # Merge into per-league history
        if total_rows:
            df_total = pd.DataFrame(total_rows)
            st.session_state.last_update = now_utc_iso()
            st.session_state.last_snapshot_count = len(snapshot_rows)
            st.session_state.last_burst_count = len(burst_rows)

            # group by league (fallback to sport)
            for (sport, league), grp in df_total.groupby(["sport", "league"], dropna=False):
                key = league or sport or "Unknown"
                existing = st.session_state.odds_history.get(key)
                merged = merge_into_history(existing, grp.to_dict("records"))
                st.session_state.odds_history[key] = merged

    # Flatten all history into one big DF for global engines
    if st.session_state.odds_history:
        combined_history = pd.concat(
            st.session_state.odds_history.values(),
            ignore_index=True,
        )
    else:
        combined_history = pd.DataFrame()

    # Global derived tables
    line_move_df = compute_line_moves(combined_history)
    efd_df = compute_efd_scores(combined_history)
    arb_df = detect_arbitrage(combined_history)

    # ------------ TABS ------------
    tab_dash, tab_arb, tab_line, tab_analytics = st.tabs(
        ["🏠 Dashboard", "💰 Arbitrage", "📈 Line Moves", "📊 Analytics"]
    )

    # -------- DASHBOARD TAB --------
    with tab_dash:
        col1, col2, col3, col4 = st.columns(4)

        active_sports = len(
            {k for k, v in st.session_state.odds_history.items() if not v.empty}
        )
        total_books = len(set(combined_history["sportsbook"])) if not combined_history.empty else 0

        with col1:
            st.metric("Active Sports (this session)", active_sports)
        with col2:
            st.metric("Tracked Sportsbooks", total_books)
        with col3:
            st.metric("Snapshot rows ingested", st.session_state.last_snapshot_count)
        with col4:
            st.metric("Stream rows ingested", st.session_state.last_burst_count)

        st.markdown("---")

        if combined_history.empty:
            st.info("No odds data yet. Click **Run Snapshot + Burst** in the sidebar.")
        else:
            st.subheader("Top EFD Edges (All Sports)")
            if efd_df.empty:
                st.caption("No EFD scores yet for current data.")
            else:
                view_cols = [
                    "sport", "league", "home_team", "away_team",
                    "selection", "EFD_score", "edge_pct",
                    "best_decimal", "avg_decimal",
                ]
                st.dataframe(
                    efd_df[view_cols].head(50),
                    use_container_width=True,
                )

    # -------- ARBITRAGE TAB --------
    with tab_arb:
        st.subheader("Real-time Arbitrage Radar")

        if arb_df.empty:
            st.info("No arbitrage edges detected yet for current markets.")
        else:
            st.dataframe(
                arb_df[[
                    "sport", "league", "home_team", "away_team",
                    "market", "n_sides",
                    "total_implied_prob", "arbitrage_edge_pct",
                ]].head(100),
                use_container_width=True,
            )

    # -------- LINE MOVES TAB --------
    with tab_line:
        st.subheader("Cumulative Line Movement (This Session)")

        if line_move_df.empty:
            st.info("No line movement yet. Run another snapshot/burst and watch this grow.")
        else:
            # League filter
            leagues = sorted(
                list({l for l in line_move_df["league"].dropna().unique()})
            )
            league_filter = st.selectbox(
                "League filter",
                options=["All"] + leagues,
                index=0,
            )
            df_view = line_move_df
            if league_filter != "All":
                df_view = df_view[df_view["league"] == league_filter]

            st.dataframe(
                df_view[[
                    "sport", "league",
                    "home_team", "away_team",
                    "selection", "sportsbook",
                    "market",
                    "open_odds", "current_odds",
                    "line_move", "move_direction",
                ]].head(200),
                use_container_width=True,
            )

    # -------- ANALYTICS TAB --------
    with tab_analytics:
        st.subheader("Session Analytics")

        if combined_history.empty:
            st.info("No data yet to analyze.")
        else:
            st.markdown("**Raw history sample (latest 200 rows)**")
            st.dataframe(
                combined_history.sort_values("timestamp", ascending=False).head(200),
                use_container_width=True,
            )

            st.markdown("---")
            st.markdown("**EFD Score Distribution**")
            if efd_df.empty:
                st.caption("No EFD scores yet.")
            else:
                st.bar_chart(
                    efd_df["EFD_score"]
                    .clip(0, 100)
                    .round()
                    .value_counts()
                    .sort_index()
                )

            st.markdown("---")
            st.markdown("**Arbitrage Edge Distribution**")
            if arb_df.empty:
                st.caption("No arbitrage edges yet.")
            else:
                st.bar_chart(
                    arb_df["arbitrage_edge_pct"]
                    .round()
                    .value_counts()
                    .sort_index()
                )


if __name__ == "__main__":
    main()