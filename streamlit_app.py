import os
import time
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import sseclient
import streamlit as st

# ======================================
# 🔑 CONFIG & GLOBALS
# ======================================

OPTICODDS_API_KEY = os.getenv("OPTICODDS_API_KEY", "")

OPTICODDS_BASE = "https://api.opticodds.com/api/v3"

PREMADE_SETUPS = {
    "NBA – Main Moneyline Board": {
        "sport": "basketball",
        "league": "NBA",
        "markets": ["moneyline"],
        "sportsbooks": ["DraftKings", "FanDuel", "Caesars", "BetMGM"],
    },
    "NFL – Sides (Spread)": {
        "sport": "american_football",
        "league": "NFL",
        "markets": ["spread"],
        "sportsbooks": ["DraftKings", "FanDuel", "Caesars", "BetMGM"],
    },
    "MLB – Moneyline": {
        "sport": "baseball",
        "league": "MLB",
        "markets": ["moneyline"],
        "sportsbooks": ["DraftKings", "FanDuel", "Caesars", "BetMGM"],
    },
    "NHL – Moneyline": {
        "sport": "ice_hockey",
        "league": "NHL",
        "markets": ["moneyline"],
        "sportsbooks": ["DraftKings", "FanDuel", "Caesars", "BetMGM"],
    },
}

PRICE_CAP_DEFAULT = (-250, 250)
MAX_FIXTURES_PER_ODDS_CALL = 5  # OpticOdds /fixtures/odds limit (chunked)


# ======================================
# 🎨 FUTURISTIC / ELECTRIC THEME
# ======================================

def inject_theme():
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top left, #0b1220 0, #020617 40%, #000000 100%);
            color: #e2f3ff;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            color: #e5f0ff;
            text-shadow: 0 0 12px rgba(125, 211, 252, 0.8);
        }
        .efd-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 999px;
            background: linear-gradient(90deg, #0ea5e9, #22c55e);
            color: #020617;
            font-weight: 700;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .arb-badge {
            padding: 3px 9px;
            border-radius: 999px;
            background: rgba(34,197,94,0.1);
            border: 1px solid rgba(34,197,94,0.7);
            color: #bbf7d0;
            font-size: 0.7rem;
            text-transform: uppercase;
        }
        .steam-hot {
            color: #f97316;
            font-weight: 600;
        }
        .efd-score-high {
            color: #22c55e !important;
            font-weight: 700;
        }
        .efd-score-med {
            color: #eab308 !important;
            font-weight: 600;
        }
        .efd-score-low {
            color: #f97316 !important;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ======================================
# 🧮 ODDS / EV / EFD HELPERS
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
    else:
        return 1.0


def implied_prob(odds: float) -> float:
    dec = american_to_decimal(odds)
    if dec <= 1:
        return 0.0
    return 1.0 / dec


def compute_no_vig_ev(group: pd.DataFrame) -> pd.DataFrame:
    """
    Given a group of odds with same (fixture_id, market, grouping_key),
    compute fair probabilities & EV using best decimal price per selection.
    """
    if group.empty:
        return group

    best = (
        group.sort_values("price_decimal", ascending=False)
        .groupby("selection", as_index=False)
        .first()
    )

    if len(best) < 2:
        best["fair_prob"] = np.nan
        best["no_vig_ev"] = np.nan
        return best

    inv_sum = (1.0 / best["price_decimal"]).sum()
    if inv_sum <= 0:
        best["fair_prob"] = np.nan
        best["no_vig_ev"] = np.nan
        return best

    best["fair_prob"] = (1.0 / best["price_decimal"]) / inv_sum
    best["no_vig_ev"] = best["price_decimal"] * best["fair_prob"] - 1.0
    return best


def detect_two_way_arb(group: pd.DataFrame) -> Optional[Dict]:
    """
    Check if there is a 2-way arbitrage opportunity in this group.

    Use best price per selection across sportsbooks.
    """
    if group.empty:
        return None

    best = (
        group.sort_values("price_decimal", ascending=False)
        .groupby("selection", as_index=False)
        .first()
    )

    if len(best) != 2:
        return None

    d1 = best.iloc[0]["price_decimal"]
    d2 = best.iloc[1]["price_decimal"]

    inv_sum = (1.0 / d1) + (1.0 / d2)
    if inv_sum >= 1.0:
        return None

    arb_yield_pct = (1.0 - inv_sum) * 100.0
    return {
        "fixture_id": best.iloc[0]["fixture_id"],
        "market": best.iloc[0]["market"],
        "grouping_key": best.iloc[0]["grouping_key"],
        "sel1": best.iloc[0]["selection"],
        "book1": best.iloc[0]["sportsbook"],
        "price1": best.iloc[0]["price"],
        "sel2": best.iloc[1]["selection"],
        "book2": best.iloc[1]["sportsbook"],
        "price2": best.iloc[1]["price"],
        "arb_yield_pct": arb_yield_pct,
    }


def minutes_until(start_iso: Optional[str]) -> Optional[float]:
    if not start_iso:
        return None
    try:
        dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return (dt - now).total_seconds() / 60.0
    except Exception:
        return None


def compute_efd_score(ev: float, steam_abs: float, minutes_to_start: Optional[float]) -> float:
    """
    EFD-style score: EV + steam + timing.
    You can replace this with your proprietary weights.
    """
    # Math component (cap EV range roughly -10% to +12%)
    ev_cap = min(max(ev, -0.10), 0.12)
    math_component = (ev_cap + 0.10) / 0.22  # ~0–1

    # Steam based on absolute change in implied probability (cap at 10pp)
    steam_cap = min(steam_abs, 0.10)
    steam_component = steam_cap / 0.10  # 0–1

    # Timing preference
    if minutes_to_start is None:
        timing_component = 0.5
    else:
        if minutes_to_start <= 0:
            timing_component = 1.0
        elif minutes_to_start <= 360:
            timing_component = 1.0 - (minutes_to_start / 360.0) * 0.5
        else:
            timing_component = 0.5

    score_raw = 0.45 * math_component + 0.35 * steam_component + 0.20 * timing_component
    return float(round(score_raw * 100.0, 1))


# ======================================
# 🌐 OPTICODDS API WRAPPERS
# ======================================

def optic_get(path: str, params: Dict) -> Dict:
    params = dict(params)
    params["key"] = OPTICODDS_API_KEY
    url = f"{OPTICODDS_BASE}{path}"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_fixtures(sport: str, league: str, days_ahead: int = 1) -> pd.DataFrame:
    """
    Step 1: static fixtures call (sets the board).
    """
    now = datetime.now(timezone.utc)
    start_after = now.isoformat().replace("+00:00", "Z")
    start_before = (now + timedelta(days=days_ahead)).isoformat().replace("+00:00", "Z")

    data = optic_get(
        "/fixtures",
        {
            "sport": sport,
            "league": league,
            "start_date_after": start_after,
            "start_date_before": start_before,
        },
    ).get("data", [])

    rows = []
    for fx in data:
        rows.append(
            {
                "fixture_id": fx["id"],
                "start_date": fx.get("start_date"),
                "home": fx.get("home_team_display"),
                "away": fx.get("away_team_display"),
                "sport": fx.get("sport", {}).get("id"),
                "league": fx.get("league", {}).get("id"),
                "status": fx.get("status"),
            }
        )
    return pd.DataFrame(rows)


def fetch_fixture_odds(
    fixture_ids: List[str],
    sportsbooks: List[str],
    markets: List[str],
) -> pd.DataFrame:
    """
    Step 2: static odds snapshot (opening board), chunked by fixture.
    """
    if not fixture_ids:
        return pd.DataFrame()

    all_rows = []
    for i in range(0, len(fixture_ids), MAX_FIXTURES_PER_ODDS_CALL):
        chunk = fixture_ids[i : i + MAX_FIXTURES_PER_ODDS_CALL]
        data = optic_get(
            "/fixtures/odds",
            {
                "fixture_id": chunk,
                "sportsbook": sportsbooks,
                "market": markets,
                "is_main": True,
                "odds_format": "AMERICAN",
            },
        ).get("data", [])

        for fx in data:
            fid = fx["id"]
            start_date = fx.get("start_date")
            sport = fx.get("sport", {}).get("id")
            league = fx.get("league", {}).get("id")
            odds_list = fx.get("odds", [])
            for od in odds_list:
                rows = od.get("lines", [])
                for ln in rows:
                    # This assumes shape similar to /fixtures/odds docs; adapt if your payload differs
                    all_rows.append(
                        {
                            "fixture_id": fid,
                            "sport": sport,
                            "league": league,
                            "start_date": start_date,
                            "sportsbook": od.get("sportsbook"),
                            "market": od.get("market"),
                            "grouping_key": ln.get("grouping_key"),
                            "selection": ln.get("name"),
                            "price": ln.get("price_american"),
                            "price_decimal": american_to_decimal(ln.get("price_american")),
                            "is_main": od.get("is_main", True),
                        }
                    )

    return pd.DataFrame(all_rows)


def stream_live_odds_once(
    sport: str,
    league: str,
    sportsbooks: List[str],
    markets: List[str],
    max_events: int = 100,
    timeout: int = 10,
) -> List[Dict]:
    """
    Step 3: Live plugin – one SSE chunk.

    This is called on each app refresh when live mode is ON.
    """
    params = {
        "league": league,
        "sportsbook": ",".join(sportsbooks),
        "market": ",".join(markets),
        "odds_format": "AMERICAN",
        "key": OPTICODDS_API_KEY,
    }
    url = f"{OPTICODDS_BASE}/stream/odds/{sport}"

    headers = {"Accept": "text/event-stream"}

    with requests.get(url, params=params, headers=headers, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        client = sseclient.SSEClient(resp)
        events = []
        count = 0
        for event in client.events():
            if event.data:
                try:
                    data = json.loads(event.data)
                    events.append(data)
                except json.JSONDecodeError:
                    pass
            count += 1
            if count >= max_events:
                break
        return events


# ======================================
# 🧱 STATE MANAGEMENT
# ======================================

def init_session():
    if "fixtures_df" not in st.session_state:
        st.session_state["fixtures_df"] = pd.DataFrame()
    if "board_df" not in st.session_state:
        st.session_state["board_df"] = pd.DataFrame()
    if "config" not in st.session_state:
        st.session_state["config"] = {}
    if "live_mode" not in st.session_state:
        st.session_state["live_mode"] = False


def apply_stream_updates(events: List[Dict]):
    """
    Map SSE events into board updates.
    You MUST adapt the mapping to your actual streaming payload.
    """
    if st.session_state["board_df"].empty:
        return

    df = st.session_state["board_df"].copy()

    for ev in events:
        # This handler assumes an event shape roughly like:
        # { "fixture_id": "...", "odds": [ { "sportsbook": "...", "market": "...",
        #                                   "grouping_key": "...", "name": "...",
        #                                   "price_american": -120 } ] }
        fixture_id = ev.get("fixture_id") or ev.get("fixture", {}).get("id")
        odds_list = ev.get("odds", [])

        if isinstance(odds_list, dict):
            odds_list = [odds_list]

        for od in odds_list:
            sb = od.get("sportsbook")
            market = od.get("market")
            lines = od.get("lines", [])
            for ln in lines:
                grouping_key = ln.get("grouping_key")
                selection = ln.get("name")
                price = ln.get("price_american")
                dec = american_to_decimal(price)

                mask = (
                    (df["fixture_id"] == fixture_id)
                    & (df["sportsbook"] == sb)
                    & (df["market"] == market)
                    & (df["grouping_key"] == grouping_key)
                    & (df["selection"] == selection)
                )
                # If row exists, update current price + steam
                if mask.any():
                    df.loc[mask, "price"] = price
                    df.loc[mask, "price_decimal"] = dec
                # If not, you could append new rows here if you want.

    st.session_state["board_df"] = df


def recompute_board_metrics():
    """
    After static build or live updates, recompute:
    - implied probs
    - steam
    - EV
    - EFD
    - arbitrage flags
    """
    df = st.session_state["board_df"].copy()
    if df.empty:
        return

    # Baseline open implied prob (on first static build)
    if "open_implied" not in df.columns:
        df["open_implied"] = df["price"].apply(implied_prob)
    # Current implied
    df["curr_implied"] = df["price"].apply(implied_prob)
    df["steam"] = df["curr_implied"] - df["open_implied"]
    df["steam_abs"] = df["steam"].abs()

    # EV + arbitrage by group
    df["fair_prob"] = np.nan
    df["no_vig_ev"] = np.nan
    df["is_arb"] = False

    group_cols = ["fixture_id", "market", "grouping_key"]
    grouped = df.groupby(group_cols, as_index=False)
    rows_ev = []
    arb_rows = []

    for _, g in grouped:
        # EV
        ev_df = compute_no_vig_ev(g)
        rows_ev.append(ev_df)

        # Arbitrage
        arb = detect_two_way_arb(g)
        if arb is not None:
            arb_rows.append(arb)

    if rows_ev:
        ev_all = pd.concat(rows_ev, ignore_index=True)
        # Merge fair_prob + EV back to full df on keys
        join_cols = ["fixture_id", "market", "grouping_key", "selection", "sportsbook"]
        df = df.merge(
            ev_all[join_cols + ["fair_prob", "no_vig_ev"]],
            on=join_cols,
            how="left",
        )

    if arb_rows:
        arb_df = pd.DataFrame(arb_rows)
        # Flag rows that participate in arb
        for _, r in arb_df.iterrows():
            mask1 = (
                (df["fixture_id"] == r["fixture_id"])
                & (df["market"] == r["market"])
                & (df["grouping_key"] == r["grouping_key"])
                & (df["selection"] == r["sel1"])
                & (df["sportsbook"] == r["book1"])
            )
            mask2 = (
                (df["fixture_id"] == r["fixture_id"])
                & (df["market"] == r["market"])
                & (df["grouping_key"] == r["grouping_key"])
                & (df["selection"] == r["sel2"])
                & (df["sportsbook"] == r["book2"])
            )
            df.loc[mask1 | mask2, "is_arb"] = True

    # EFD score
    minutes_to_start = df["start_date"].apply(minutes_until)
    df["minutes_to_start"] = minutes_to_start
    df["efd_score"] = df.apply(
        lambda row: compute_efd_score(
            row.get("no_vig_ev") or 0.0,
            row.get("steam_abs") or 0.0,
            row.get("minutes_to_start"),
        ),
        axis=1,
    )

    st.session_state["board_df"] = df


# ======================================
# 🖥️ UI – STEP WIZARD + MAIN VIEW
# ======================================

def main():
    init_session()
    inject_theme()

    st.title("⚡ Edge Force Dominion – Live Steam Board")

    # API key
    with st.sidebar:
        st.markdown("### 🔐 API Key")
        api_key_input = st.text_input(
            "OpticOdds API Key",
            value=OPTICODDS_API_KEY,
            type="password",
        )
        if api_key_input:
            os.environ["OPTICODDS_API_KEY"] = api_key_input
            globals()["OPTICODDS_API_KEY"] = api_key_input

        st.markdown("---")

        # Premade setups
        st.markdown("### 🧠 Premade Setup")
        preset_name = st.selectbox("Choose a board preset", list(PREMADE_SETUPS.keys()))
        preset = PREMADE_SETUPS[preset_name]

        st.session_state["config"] = {
            "sport": preset["sport"],
            "league": preset["league"],
            "markets": preset["markets"],
            "sportsbooks": preset["sportsbooks"],
        }

        st.markdown(
            f"**Sport:** `{preset['sport']}`  \n"
            f"**League:** `{preset['league']}`  \n"
            f"**Markets:** `{', '.join(preset['markets'])}`  \n"
            f"**Books:** `{', '.join(preset['sportsbooks'])}`"
        )

        st.markdown("---")

        # Price cap
        st.markdown("### 🎚️ Price Filter")
        min_price, max_price = st.slider(
            "American odds range",
            min_value=-1000,
            max_value=1000,
            value=PRICE_CAP_DEFAULT,
            step=10,
        )

        st.session_state["price_cap"] = (min_price, max_price)

        st.markdown("---")

        st.markdown("### 🛰️ Live Mode")
        live_toggle = st.checkbox("Activate live stream (SSE)", value=st.session_state["live_mode"])
        st.session_state["live_mode"] = live_toggle

        if live_toggle:
            st.caption("Live updates: streaming odds chunks each refresh.")

    # --- STEP 1: Fixtures ---
    st.subheader("1️⃣ Static Setup – Fixtures")
    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("Fetch Fixtures (Static)"):
            cfg = st.session_state["config"]
            with st.spinner("Calling fixtures API..."):
                fixtures_df = fetch_fixtures(cfg["sport"], cfg["league"])
            st.session_state["fixtures_df"] = fixtures_df

    with col2:
        fx_df = st.session_state["fixtures_df"]
        if fx_df.empty:
            st.info("No fixtures loaded yet. Click **Fetch Fixtures** to set the board.")
        else:
            st.write(f"Loaded **{len(fx_df)}** fixtures.")
            st.dataframe(
                fx_df[["fixture_id", "start_date", "home", "away", "status"]].head(15),
                use_container_width=True,
            )

    st.markdown("---")

    # --- STEP 2: Opening Odds Board ---
    st.subheader("2️⃣ Static Setup – Opening Odds Snapshot")

    col3, col4 = st.columns([1, 2])

    with col3:
        if st.button("Pull Opening Odds (Static)"):
            fx_df = st.session_state["fixtures_df"]
            if fx_df.empty:
                st.warning("Run fixtures step first.")
            else:
                cfg = st.session_state["config"]
                fixture_ids = fx_df["fixture_id"].tolist()
                with st.spinner("Calling /fixtures/odds to build opening board..."):
                    odds_df = fetch_fixture_odds(
                        fixture_ids,
                        cfg["sportsbooks"],
                        cfg["markets"],
                    )

                if odds_df.empty:
                    st.warning("No odds data returned.")
                else:
                    # Connect to fixture info
                    merged = odds_df.merge(
                        fx_df[["fixture_id", "home", "away"]],
                        on="fixture_id",
                        how="left",
                    )
                    st.session_state["board_df"] = merged
                    recompute_board_metrics()

    with col4:
        board = st.session_state["board_df"]
        if board.empty:
            st.info("Opening odds board is empty. Click **Pull Opening Odds**.")
        else:
            st.write(f"Board entries: **{len(board)}** rows.")
            st.dataframe(
                board[
                    [
                        "home",
                        "away",
                        "sportsbook",
                        "market",
                        "selection",
                        "price",
                        "no_vig_ev",
                        "steam",
                        "efd_score",
                        "is_arb",
                    ]
                ].head(20),
                use_container_width=True,
            )

    st.markdown("---")

    # --- STEP 3: Live Plugins (SSE) ---
    st.subheader("3️⃣ Live Plugins – Steam & Arbitrage Updates")

    if st.session_state["live_mode"] and not st.session_state["board_df"].empty:
        cfg = st.session_state["config"]
        with st.spinner("Streaming a live odds chunk..."):
            try:
                events = stream_live_odds_once(
                    cfg["sport"],
                    cfg["league"],
                    cfg["sportsbooks"],
                    cfg["markets"],
                    max_events=50,
                    timeout=8,
                )
                if events:
                    apply_stream_updates(events)
                    recompute_board_metrics()
            except Exception as e:
                st.warning(f"Streaming error: {e}")

        # auto-refresh every few seconds while live
        st.experimental_rerun()

    # --- MAIN BOARD VIEW ---
    st.subheader("📊 Edge Force Board")

    board = st.session_state["board_df"].copy()
    if board.empty:
        st.info("Board not yet built. Complete steps 1 and 2 above.")
        return

    # Apply price filter
    cap_min, cap_max = st.session_state["price_cap"]
    board = board[(board["price"] >= cap_min) & (board["price"] <= cap_max)]

    # Highest score first
    sort_mode = st.radio(
        "Sort mode",
        ["Highest EFD score", "Highest EV", "Most Steam", "Arbitrage first"],
        horizontal=True,
    )

    if sort_mode == "Highest EV":
        board = board.sort_values("no_vig_ev", ascending=False)
    elif sort_mode == "Most Steam":
        board = board.sort_values("steam_abs", ascending=False)
    elif sort_mode == "Arbitrage first":
        board = board.sort_values(["is_arb", "no_vig_ev"], ascending=[False, False])
    else:
        board = board.sort_values("efd_score", ascending=False)

    # Display key columns nicely
    display_cols = [
        "home",
        "away",
        "sportsbook",
        "market",
        "selection",
        "price",
        "no_vig_ev",
        "steam",
        "efd_score",
        "is_arb",
    ]

    # Pretty formatting
    def fmt_row(row):
        ev_pct = f"{(row.get('no_vig_ev') or 0)*100:.1f}%"
        steam_pct = f"{(row.get('steam') or 0)*100:.1f}%"
        efd = row.get("efd_score") or 0.0
        efd_str = f"{efd:.1f}"
        return ev_pct, steam_pct, efd_str

    pretty = board[display_cols].copy()
    pretty["EV%"] = pretty.apply(lambda r: f"{(r['no_vig_ev'] or 0)*100:.1f}%", axis=1)
    pretty["Steam%"] = pretty.apply(lambda r: f"{(r['steam'] or 0)*100:.1f}%", axis=1)

    pretty = pretty.rename(
        columns={
            "home": "Home",
            "away": "Away",
            "sportsbook": "Book",
            "market": "Market",
            "selection": "Bet",
            "price": "Odds",
            "efd_score": "EFD",
            "is_arb": "Arb?",
        }
    )

    st.dataframe(
        pretty[
            [
                "Home",
                "Away",
                "Book",
                "Market",
                "Bet",
                "Odds",
                "EV%",
                "Steam%",
                "EFD",
                "Arb?",
            ]
        ].head(60),
        use_container_width=True,
    )

    st.caption("EFD = composite score (EV + steam + timing). Arb? = part of a 2-way surebet.")

if __name__ == "__main__":
    main()