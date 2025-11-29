import os
import time
import json
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import requests
import pandas as pd
import numpy as np
import streamlit as st


# -----------------------------
# Config
# -----------------------------

OPTICODDS_BASE = "https://api.opticodds.com/api/v3"

SPORT_PATHS = {
    "NBA": "basketball_nba",
    "NCAAB": "basketball_ncaab",
    "NFL": "football_nfl",
    "NCAAF": "football_ncaaf",
    "NHL": "ice_hockey_nhl",
    "MLB": "baseball_mlb",
}

DEFAULT_SPORTS = ["NCAAF", "NFL", "NBA"]

DEFAULT_BOOKS = [
    "FanDuel",
    "DraftKings",
    "BetMGM",
    "Caesars",
    "Pinnacle",
    "LowVig",
]

DEFAULT_MARKETS = ["moneyline", "point_spread", "total_points"]


# -----------------------------
# Helpers
# -----------------------------

def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal; safe against junk/0 values."""
    try:
        o = float(odds)
    except Exception:
        return 1.0
    if o == 0:
        return 1.0
    if o > 0:
        return 1.0 + (o / 100.0)
    return 1.0 + (100.0 / abs(o))


def safe_get(url: str, params=None, timeout: int = 10):
    """GET wrapper that returns JSON or {} and captures errors."""
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"HTTP error calling {url}: {e}")
        return {}


def init_session_state():
    if "fixtures" not in st.session_state:
        st.session_state.fixtures = pd.DataFrame()
    if "odds_snapshot" not in st.session_state:
        st.session_state.odds_snapshot = pd.DataFrame()
    if "odds_stream" not in st.session_state:
        st.session_state.odds_stream = pd.DataFrame()
    if "last_ingest" not in st.session_state:
        st.session_state.last_ingest = None


# -----------------------------
# Fixtures & Snapshot Odds
# -----------------------------

def fetch_active_fixtures(
    api_key: str,
    sports: List[str],
    max_per_sport: int = 100,
) -> pd.DataFrame:
    """Pull upcoming/live fixtures for each sport. Returns combined DataFrame."""
    rows = []

    for sport in sports:
        sport_path = SPORT_PATHS.get(sport)
        if not sport_path:
            continue

        url = f"{OPTICODDS_BASE}/fixtures"
        params = {
            "key": api_key,
            "sport": sport_path,
            "event_status": "upcoming,live",
        }

        data = safe_get(url, params=params)

        # ---- robust extraction of fixture list ----
        fixtures = []
        if isinstance(data, list):
            fixtures = data
        elif isinstance(data, dict):
            if isinstance(data.get("fixtures"), list):
                fixtures = data["fixtures"]
            elif isinstance(data.get("data"), list):
                fixtures = data["data"]
        # If still not a list, we just skip this sport
        if not isinstance(fixtures, list) or len(fixtures) == 0:
            st.warning(f"No fixtures list returned for {sport}.")
            continue

        if max_per_sport is not None:
            fixtures = fixtures[:max_per_sport]

        for fx in fixtures:
            fixture_id = fx.get("fixture_id") or fx.get("id")
            if not fixture_id:
                continue

            home = fx.get("home_team") or fx.get("home")
            away = fx.get("away_team") or fx.get("away")
            league = fx.get("league") or fx.get("competition")
            ts_raw = fx.get("start_time") or fx.get("start_timestamp")

            try:
                if isinstance(ts_raw, (int, float)):
                    start_dt = datetime.fromtimestamp(ts_raw, tz=timezone.utc)
                elif isinstance(ts_raw, str):
                    # try ISO
                    start_dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                else:
                    start_dt = None
            except Exception:
                start_dt = None

            rows.append(
                {
                    "sport": sport,
                    "sport_path": sport_path,
                    "fixture_id": fixture_id,
                    "home_team": home,
                    "away_team": away,
                    "league": league,
                    "start_time": start_dt,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "sport",
                "sport_path",
                "fixture_id",
                "home_team",
                "away_team",
                "league",
                "start_time",
            ]
        )

    df = pd.DataFrame(rows).drop_duplicates(subset=["fixture_id"])
    return df


def fetch_odds_for_fixtures(
    api_key: str,
    fixture_ids: List[str],
    sportsbooks: List[str],
    markets: List[str],
    chunk_size: int = 20,
) -> pd.DataFrame:
    """
    Call /fixtures/odds with repeated fixture_id parameters.
    This is designed to avoid the 'fixture_ids' 400s you were seeing.
    """
    all_rows = []

    if not fixture_ids:
        return pd.DataFrame()

    for i in range(0, len(fixture_ids), chunk_size):
        chunk = fixture_ids[i : i + chunk_size]

        url = f"{OPTICODDS_BASE}/fixtures/odds"

        # We need repeated fixture_id=...&fixture_id=...
        params = [("key", api_key)]
        for fid in chunk:
            params.append(("fixture_id", fid))

        if sportsbooks:
            params.append(("sportsbooks", ",".join(sportsbooks)))
        if markets:
            params.append(("markets", ",".join(markets)))
        params.append(("is_main", "True"))

        data = safe_get(url, params=params)

        # Robust extraction
        odds_list = []
        if isinstance(data, list):
            odds_list = data
        elif isinstance(data, dict):
            if isinstance(data.get("odds"), list):
                odds_list = data["odds"]
            elif isinstance(data.get("data"), list):
                odds_list = data["data"]

        if not odds_list:
            st.warning("Snapshot odds call returned no odds (chunk).")
            continue

        for fx in odds_list:
            fixture_id = fx.get("fixture_id")
            league = fx.get("league")
            sport = fx.get("sport")

            for book in fx.get("sportsbooks", []):
                book_name = book.get("sportsbook")
                for mkt in book.get("markets", []):
                    market = mkt.get("market")
                    for sel in mkt.get("selections", []):
                        selection = sel.get("selection")
                        odds = sel.get("odds_american")
                        try:
                            dec = american_to_decimal(odds)
                        except Exception:
                            dec = 1.0

                        all_rows.append(
                            {
                                "fixture_id": fixture_id,
                                "league": league,
                                "sport": sport,
                                "sportsbook": book_name,
                                "market": market,
                                "selection": selection,
                                "odds_american": odds,
                                "odds_decimal": dec,
                                "timestamp": datetime.now(timezone.utc),
                                "source": "snapshot",
                            }
                        )

    if not all_rows:
        return pd.DataFrame(
            columns=[
                "fixture_id",
                "league",
                "sport",
                "sportsbook",
                "market",
                "selection",
                "odds_american",
                "odds_decimal",
                "timestamp",
                "source",
            ]
        )

    return pd.DataFrame(all_rows)


def build_snapshot(
    api_key: str,
    sports: List[str],
    sportsbooks: List[str],
    markets: List[str],
    max_per_sport: int = 100,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """High-level: fixtures + odds snapshot."""
    fixtures_df = fetch_active_fixtures(api_key, sports, max_per_sport=max_per_sport)
    if fixtures_df.empty:
        return fixtures_df, pd.DataFrame()

    fixture_ids = fixtures_df["fixture_id"].dropna().astype(str).tolist()

    odds_df = fetch_odds_for_fixtures(
        api_key,
        fixture_ids,
        sportsbooks,
        markets,
        chunk_size=20,
    )

    return fixtures_df, odds_df


# -----------------------------
# Streaming (bursts)
# -----------------------------

def parse_sse_stream(resp) -> List[dict]:
    """
    Very small SSE parser: returns list of JSON events (dicts).
    """
    events = []
    data_lines = []

    for raw_line in resp.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        line = raw_line.strip()
        if not line:
            if data_lines:
                payload = "\n".join(data_lines)
                data_lines = []
                # Expect "data: {...json...}"
                prefix = "data:"
                for p in payload.splitlines():
                    if p.startswith(prefix):
                        try:
                            obj = json.loads(p[len(prefix) :].strip())
                            events.append(obj)
                        except Exception:
                            pass
            continue

        if line.startswith("data:"):
            data_lines.append(line)
        # ignore id:, event:, retry:

    return events


def stream_odds_burst(
    api_key: str,
    sport_path: str,
    sportsbooks: List[str],
    markets: List[str],
    is_live: bool,
    seconds: int = 8,
) -> pd.DataFrame:
    """
    Connect to /stream/odds/{sport_path} for a short burst and collect odds updates.
    """
    url = f"{OPTICODDS_BASE}/stream/odds/{sport_path}"

    params = {
        "key": api_key,
    }
    if sportsbooks:
        params["sportsbooks"] = ",".join(sportsbooks)
    if markets:
        params["markets"] = ",".join(markets)
    if is_live:
        params["event_status"] = "live"
    else:
        params["event_status"] = "upcoming,live"

    try:
        with requests.get(url, params=params, stream=True, timeout=seconds + 3) as resp:
            resp.raise_for_status()
            start = time.time()
            rows = []

            for raw_line in resp.iter_lines(decode_unicode=True):
                if time.time() - start > seconds:
                    break
                if raw_line is None:
                    continue
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                try:
                    payload = json.loads(line[len("data:") :].strip())
                except Exception:
                    continue

                fixture_id = payload.get("fixture_id")
                league = payload.get("league")
                sport = payload.get("sport")
                ts_raw = payload.get("timestamp")

                try:
                    ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                except Exception:
                    ts = datetime.now(timezone.utc)

                for book in payload.get("sportsbooks", []):
                    book_name = book.get("sportsbook")
                    for mkt in book.get("markets", []):
                        market = mkt.get("market")
                        for sel in mkt.get("selections", []):
                            selection = sel.get("selection")
                            odds = sel.get("odds_american")
                            dec = american_to_decimal(odds)
                            rows.append(
                                {
                                    "fixture_id": fixture_id,
                                    "league": league,
                                    "sport": sport,
                                    "sportsbook": book_name,
                                    "market": market,
                                    "selection": selection,
                                    "odds_american": odds,
                                    "odds_decimal": dec,
                                    "timestamp": ts,
                                    "source": "stream",
                                }
                            )

            if not rows:
                return pd.DataFrame(
                    columns=[
                        "fixture_id",
                        "league",
                        "sport",
                        "sportsbook",
                        "market",
                        "selection",
                        "odds_american",
                        "odds_decimal",
                        "timestamp",
                        "source",
                    ]
                )
            return pd.DataFrame(rows)

    except Exception as e:
        st.error(f"Error opening stream for {sport_path}: {e}")
        return pd.DataFrame(
            columns=[
                "fixture_id",
                "league",
                "sport",
                "sportsbook",
                "market",
                "selection",
                "odds_american",
                "odds_decimal",
                "timestamp",
                "source",
            ]
        )


# -----------------------------
# EFD scoring & arbitrage
# -----------------------------

def compute_efd_scores(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple but consistent EFD score:
      - For each fixture/selection:
        * Take best price across books.
        * Compute implied prob from that best price.
        * Compare to avg implied prob across books.
        * Edge = avg_prob - best_prob.
      - Scale edge into 0–100 and classify Tier.
    """
    if odds_df is None or odds_df.empty:
        return pd.DataFrame(
            columns=[
                "fixture_id",
                "selection",
                "best_odds",
                "best_sportsbook",
                "avg_implied_prob",
                "best_implied_prob",
                "edge",
                "EFD_score",
                "efd_tier",
            ]
        )

    df = odds_df.copy()
    # moneyline only for now
    df = df[df["market"] == "moneyline"].copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "fixture_id",
                "selection",
                "best_odds",
                "best_sportsbook",
                "avg_implied_prob",
                "best_implied_prob",
                "edge",
                "EFD_score",
                "efd_tier",
            ]
        )

    # implied prob from decimal
    df["implied_prob"] = 1.0 / df["odds_decimal"]

    rows = []
    for (fixture_id, selection), grp in df.groupby(["fixture_id", "selection"]):
        grp = grp.copy()
        grp = grp.sort_values("timestamp")

        # best price (highest decimal) and its book
        idx_best = grp["odds_decimal"].idxmax()
        best_row = grp.loc[idx_best]

        best_odds = best_row["odds_american"]
        best_dec = best_row["odds_decimal"]
        best_book = best_row["sportsbook"]

        avg_prob = grp["implied_prob"].mean()
        best_prob = 1.0 / best_dec if best_dec > 0 else avg_prob

        edge = avg_prob - best_prob  # higher = better for us

        # Scale to 0–100 (cap at 0–0.15 edge)
        edge_clamped = max(0.0, min(edge, 0.15))
        score = int(round((edge_clamped / 0.15) * 100))

        if score >= 80:
            tier = "Tier A"
        elif score >= 60:
            tier = "Tier B"
        elif score >= 40:
            tier = "Tier C"
        else:
            tier = "Watchlist"

        rows.append(
            {
                "fixture_id": fixture_id,
                "selection": selection,
                "best_odds": best_odds,
                "best_sportsbook": best_book,
                "avg_implied_prob": avg_prob,
                "best_implied_prob": best_prob,
                "edge": edge,
                "EFD_score": score,
                "efd_tier": tier,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "fixture_id",
                "selection",
                "best_odds",
                "best_sportsbook",
                "avg_implied_prob",
                "best_implied_prob",
                "edge",
                "EFD_score",
                "efd_tier",
            ]
        )
    return out.sort_values("EFD_score", ascending=False)


def find_arbitrage(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple 2-outcome arbitrage scan on moneylines.
    """
    if odds_df is None or odds_df.empty:
        return pd.DataFrame(
            columns=[
                "fixture_id",
                "side_1",
                "side_1_book",
                "side_1_odds",
                "side_2",
                "side_2_book",
                "side_2_odds",
                "inv_sum",
                "arb_yield_pct",
            ]
        )

    df = odds_df.copy()
    df = df[df["market"] == "moneyline"].copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "fixture_id",
                "side_1",
                "side_1_book",
                "side_1_odds",
                "side_2",
                "side_2_book",
                "side_2_odds",
                "inv_sum",
                "arb_yield_pct",
            ]
        )

    out_rows = []

    for fixture_id, grp in df.groupby("fixture_id"):
        # Map selection -> best price
        best = {}
        for selection, s_grp in grp.groupby("selection"):
            idx = s_grp["odds_decimal"].idxmax()
            r = s_grp.loc[idx]
            best[selection] = r

        if len(best) < 2:
            continue

        # Just take top 2 selections by name for now
        sel_names = list(best.keys())[:2]
        a, b = sel_names[0], sel_names[1]
        ra, rb = best[a], best[b]

        dec_a = ra["odds_decimal"]
        dec_b = rb["odds_decimal"]

        inv_sum = (1.0 / dec_a) + (1.0 / dec_b)
        if inv_sum < 0.999:
            arb_yield = (1.0 - inv_sum) * 100.0

            out_rows.append(
                {
                    "fixture_id": fixture_id,
                    "side_1": a,
                    "side_1_book": ra["sportsbook"],
                    "side_1_odds": ra["odds_american"],
                    "side_2": b,
                    "side_2_book": rb["sportsbook"],
                    "side_2_odds": rb["odds_american"],
                    "inv_sum": inv_sum,
                    "arb_yield_pct": arb_yield,
                }
            )

    if not out_rows:
        return pd.DataFrame(
            columns=[
                "fixture_id",
                "side_1",
                "side_1_book",
                "side_1_odds",
                "side_2",
                "side_2_book",
                "side_2_odds",
                "inv_sum",
                "arb_yield_pct",
            ]
        )

    return pd.DataFrame(out_rows).sort_values("arb_yield_pct", ascending=False)


# -----------------------------
# UI + Theme
# -----------------------------

def inject_theme():
    st.markdown(
        """
        <style>
        body { background-color: #050816; }
        .stApp { background: radial-gradient(circle at top, #101828 0, #050816 60%); }
        .metric-box {
            border-radius: 16px;
            padding: 10px 16px;
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid rgba(56, 189, 248, 0.3);
        }
        .big-card {
            border-radius: 22px;
            padding: 18px 22px;
            background: linear-gradient(135deg, #020617 0%, #020617 55%, #0f172a 100%);
            border: 1px solid rgba(56, 189, 248, 0.35);
            box-shadow: 0 0 32px rgba(8, 47, 73, 0.85);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    inject_theme()
    init_session_state()

    st.markdown(
        "## 🏆 EDGE FORCE DOMINION – OpticOdds Live Engine\n"
        "Multi-sport odds ingestion, EFD scoring, arbitrage radar, and line-move tracking on your OpticOdds feed."
    )

    # ----------------- SIDEBAR CONFIG -----------------
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        api_key = st.text_input(
            "OpticOdds API Key",
            type="password",
            value="",
            help="Paste the key Abe gave you.",
        )

        # Sports (with All)
        sport_choices = list(SPORT_PATHS.keys())
        sport_label = st.multiselect(
            "Select Sports",
            options=["All"] + sport_choices,
            default=["All"],
            help="Choose which sports to ingest.",
        )
        if "All" in sport_label:
            selected_sports = sport_choices
        else:
            selected_sports = sport_label

        # Books (with All)
        book_choices = DEFAULT_BOOKS
        book_label = st.multiselect(
            "Sportsbooks",
            options=["All"] + book_choices,
            default=["All"],
        )
        if "All" in book_label or not book_label:
            selected_books = book_choices
        else:
            selected_books = book_label

        # Markets (with All)
        mkt_choices = DEFAULT_MARKETS
        mkt_label = st.multiselect(
            "Markets",
            options=["All"] + mkt_choices,
            default=["All"],
        )
        if "All" in mkt_label or not mkt_label:
            selected_markets = mkt_choices
        else:
            selected_markets = mkt_label

        is_live_only = st.checkbox("Live games only", value=False)

        max_per_sport = st.slider(
            "Max fixtures per sport (snapshot)",
            min_value=10,
            max_value=150,
            value=80,
            step=10,
        )

        st.markdown("---")

        run_ingest = st.button("🚀 Run Snapshot + Burst", use_container_width=True)

    # ----------------- DATA INGEST -----------------
    if run_ingest:
        if not api_key:
            st.error("Paste your OpticOdds API key in the sidebar first.")
        else:
            with st.spinner("Pulling fixtures & snapshot odds…"):
                fixtures_df, snap_df = build_snapshot(
                    api_key,
                    selected_sports,
                    selected_books,
                    selected_markets,
                    max_per_sport=max_per_sport,
                )

            st.session_state.fixtures = fixtures_df
            st.session_state.odds_snapshot = snap_df

            stream_frames = []
            with st.spinner("Running live burst for each sport…"):
                for sp in selected_sports:
                    path = SPORT_PATHS.get(sp)
                    if not path:
                        continue
                    burst_df = stream_odds_burst(
                        api_key,
                        path,
                        selected_books,
                        selected_markets,
                        is_live=is_live_only,
                        seconds=8,
                    )
                    stream_frames.append(burst_df)

            if stream_frames:
                burst_all = pd.concat(stream_frames, ignore_index=True)
            else:
                burst_all = pd.DataFrame()

            st.session_state.odds_stream = pd.concat(
                [st.session_state.odds_stream, burst_all],
                ignore_index=True,
            )

            st.session_state.last_ingest = datetime.now(timezone.utc)

    # Combine snapshot + stream
    odds_all = pd.concat(
        [st.session_state.odds_snapshot, st.session_state.odds_stream],
        ignore_index=True,
    )

    # ----------------- TABS -----------------
    tabs = st.tabs(["🏠 Dashboard", "💰 Arbitrage", "📈 Line Moves", "📊 Analytics"])

    # -------- Dashboard --------
    with tabs[0]:
        st.markdown("### 🏠 Dashboard")

        fixtures_df = st.session_state.fixtures
        num_sports = fixtures_df["sport"].nunique() if not fixtures_df.empty else 0
        num_books = odds_all["sportsbook"].nunique() if not odds_all.empty else 0
        snap_rows = len(st.session_state.odds_snapshot)
        stream_rows = len(st.session_state.odds_stream)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f'<div class="metric-box"><b>Active Sports (this session)</b><br><span style="font-size:32px;">{num_sports}</span></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="metric-box"><b>Tracked Sportsbooks</b><br><span style="font-size:32px;">{num_books}</span></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div class="metric-box"><b>Snapshot rows ingested</b><br><span style="font-size:32px;">{snap_rows}</span></div>',
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f'<div class="metric-box"><b>Stream rows ingested</b><br><span style="font-size:32px;">{stream_rows}</span></div>',
                unsafe_allow_html=True,
            )

        if odds_all.empty:
            st.info("No odds data yet. Click **Run Snapshot + Burst** in the sidebar.")
        else:
            efd_df = compute_efd_scores(odds_all)
            if efd_df.empty:
                st.info("No EFD scores yet (need moneyline markets).")
            else:
                st.markdown("#### 🔥 Top EFD edges (all sports)")
                st.dataframe(
                    efd_df.head(40),
                    use_container_width=True,
                )

    # -------- Arbitrage --------
    with tabs[1]:
        st.markdown("### 💰 Arbitrage Opportunities")
        if odds_all.empty:
            st.info("No odds data yet. Run Snapshot + Burst first.")
        else:
            arb_df = find_arbitrage(odds_all)
            if arb_df.empty:
                st.info("No clean 2-way arbitrage found in current data.")
            else:
                st.dataframe(
                    arb_df.head(50),
                    use_container_width=True,
                )

    # -------- Line Moves --------
    with tabs[2]:
        st.markdown("### 📈 Line Movement Analysis")

        if odds_all.empty:
            st.info("No odds history yet. Run Snapshot + Burst a couple of times.")
        else:
            # Treat snapshot+stream history as "open vs latest"
            df = odds_all[odds_all["market"] == "moneyline"].copy()
            if df.empty:
                st.info("No moneyline data to compute moves.")
            else:
                agg_rows = []
                for (fixture_id, selection, sportsbook), grp in df.groupby(
                    ["fixture_id", "selection", "sportsbook"]
                ):
                    grp = grp.sort_values("timestamp")
                    first = grp.iloc[0]
                    last = grp.iloc[-1]
                    open_odds = float(first["odds_american"])
                    curr_odds = float(last["odds_american"])

                    move = curr_odds - open_odds
                    agg_rows.append(
                        {
                            "fixture_id": fixture_id,
                            "selection": selection,
                            "sportsbook": sportsbook,
                            "open_odds": open_odds,
                            "current_odds": curr_odds,
                            "line_move": move,
                            "move_abs": abs(move),
                        }
                    )
                lm_df = pd.DataFrame(agg_rows)
                if lm_df.empty:
                    st.info("No movement detected yet.")
                else:
                    lm_df = lm_df.sort_values("move_abs", ascending=False)
                    st.dataframe(
                        lm_df.head(80),
                        use_container_width=True,
                    )

    # -------- Analytics --------
    with tabs[3]:
        st.markdown("### 📊 Advanced Analytics")

        if odds_all.empty:
            st.info("No data yet. Run Snapshot + Burst.")
        else:
            total_rows = len(odds_all)
            by_sport = odds_all["sport"].value_counts().to_dict()
            by_book = odds_all["sportsbook"].value_counts().to_dict()

            st.markdown(
                f'<div class="big-card"><b>Total odds rows ingested:</b> {total_rows}<br>'
                f'<b>By sport:</b> {by_sport}<br>'
                f'<b>By sportsbook:</b> {by_book}</div>',
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()