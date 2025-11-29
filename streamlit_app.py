import os
import json
import time
from typing import List, Dict, Any, Tuple, Optional

import requests
import pandas as pd
import streamlit as st

# ============================================================
#  CONFIG
# ============================================================

OPTICODDS_API_BASE = "https://api.opticodds.com/api/v3"

SPORT_PATHS = {
    "NBA": ("basketball", "NBA"),
    "NCAAB": ("basketball", "NCAAB"),
    "NFL": ("football", "NFL"),
    "NCAAF": ("football", "NCAAF"),
    "NHL": ("hockey", "NHL"),
    "MLB": ("baseball", "MLB"),
}

DEFAULT_SPORTSBOOKS = [
    "FanDuel",
    "DraftKings",
    "BetMGM",
    "Caesars",
    "Pinnacle",
    "LowVig",
]

DEFAULT_MARKETS = [
    "Moneyline",
]

MAX_EVENTS_PER_BURST = 250  # a single click; keeps us polite to rate limits


# ============================================================
#  SIMPLE SSE PARSER
#  (We avoid extra dependencies like sseclient-py)
# ============================================================

def parse_sse_stream(
    resp: requests.Response,
    max_events: int = 200,
) -> List[Tuple[str, str]]:
    """
    Parse a Server-Sent Events stream from OpticOdds.

    Returns list of (event_name, data_str) pairs for up to max_events 'odds'
    or 'locked-odds' events.
    """
    events: List[Tuple[str, str]] = []
    event_name: Optional[str] = None
    data_lines: List[str] = []

    for raw_line in resp.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        line = raw_line.strip()

        if line == "":  # end of event
            if event_name and data_lines:
                data = "\n".join(data_lines)
                if event_name in ("odds", "locked-odds"):
                    events.append((event_name, data))
                    if len(events) >= max_events:
                        break
            event_name = None
            data_lines = []
            continue

        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())
        else:
            # other SSE fields (id:, retry:, etc.) – ignore
            continue

    return events


# ============================================================
#  STREAM LOGIC
# ============================================================

def stream_odds_once(
    api_key: str,
    sport_path: str,
    league: str,
    sportsbooks: List[str],
    markets: List[str],
    is_live: str,
    max_events: int = MAX_EVENTS_PER_BURST,
) -> pd.DataFrame:
    """
    Open the OpticOdds /stream/odds/{sport} SSE endpoint once,
    collect a burst of events, and aggregate into a DataFrame
    with line-move calculations.
    """

    if not api_key:
        st.error("No OpticOdds API key set.")
        return pd.DataFrame()

    url = f"{OPTICODDS_API_BASE}/stream/odds/{sport_path}"

    # EXACTLY like their docs: sportsbook[], market[], league[]
    # https://developer.opticodds.com/reference/get_stream-odds-sport
    params: Dict[str, Any] = {
        "key": api_key,
        "sportsbook": sportsbooks,
        "market": markets,
        "league": [league],
    }

    # OpticOdds supports filters like is_live
    if is_live in ("true", "false"):
        params["is_live"] = is_live

    try:
        resp = requests.get(url, params=params, stream=True, timeout=25)
        resp.raise_for_status()
    except Exception as e:
        if isinstance(e, requests.HTTPError) and e.response is not None:
            st.error(
                f"OpticOdds: HTTP {e.response.status_code}: {e.response.text}"
            )
        else:
            st.error(f"OpticOdds connection error: {e}")
        return pd.DataFrame()

    events = parse_sse_stream(resp, max_events=max_events)
    resp.close()

    if not events:
        st.info("No odds came through in this burst (for this league / filters).")
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    for ev_name, data_str in events:
        try:
            payload = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        for odd in payload.get("data", []):
            rows.append(
                {
                    "fixture_id": odd.get("fixture_id"),
                    "game_id": odd.get("game_id"),
                    "sportsbook": odd.get("sportsbook"),
                    "market": odd.get("market"),
                    "selection": odd.get("selection"),
                    "price": odd.get("price"),
                    "is_live": odd.get("is_live"),
                    "is_main": odd.get("is_main"),
                    "league": odd.get("league"),
                    "sport": odd.get("sport"),
                    "timestamp": odd.get("timestamp"),
                    "event_type": ev_name,
                }
            )

    if not rows:
        st.info("Stream returned events, but no odds payloads.")
        return pd.DataFrame()

    df_raw = pd.DataFrame(rows)
    df_raw["price"] = pd.to_numeric(df_raw["price"], errors="coerce")
    df_raw["timestamp"] = pd.to_numeric(df_raw["timestamp"], errors="coerce")
    df_raw = df_raw.dropna(subset=["price", "timestamp"])

    if df_raw.empty:
        st.info("No usable odds rows after cleaning.")
        return df_raw

    # --- Line-move calculation: first vs last for each (fixture, book, selection)
    agg_rows: List[Dict[str, Any]] = []

    for (fixture_id, sportsbook, selection, market), grp in df_raw.groupby(
        ["fixture_id", "sportsbook", "selection", "market"]
    ):
        grp_sorted = grp.sort_values("timestamp")

        first = grp_sorted.iloc[0]
        last = grp_sorted.iloc[-1]

        open_price = float(first["price"])
        curr_price = float(last["price"])
        move = curr_price - open_price

        agg_rows.append(
            {
                "fixture_id": fixture_id,
                "sportsbook": sportsbook,
                "selection": selection,
                "market": market,
                "league": last["league"],
                "sport": last["sport"],
                "is_live": last["is_live"],
                "open_odds": open_price,
                "current_odds": curr_price,
                "line_move": move,
                "move_abs": abs(move),
                "move_direction": (
                    "Steam to dog" if move > 0 else
                    ("Steam to fav" if move < 0 else "Flat")
                ),
            }
        )

    df = pd.DataFrame(agg_rows).sort_values("move_abs", ascending=False)
    return df


# ============================================================
#  ARBITRAGE SCAN (very simple 2-way arb check)
# ============================================================

def american_to_decimal(odds: float) -> float:
    try:
        o = float(odds)
    except Exception:
        return 1.0
    if o > 0:
        return 1.0 + (o / 100.0)
    else:
        return 1.0 + (100.0 / abs(o))


def find_basic_arbitrage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Very simple arb detector:
    - For each fixture + market, we look at all selections and all books.
    - If the sum of best implied probs for opposing sides < 1 → arb.
    """
    if df.empty:
        return df

    arbs: List[Dict[str, Any]] = []

    # Treat Moneyline / 2-way or 3-way; we don't try to be clever about pushes.
    for (fixture_id, market), grp in df.groupby(["fixture_id", "market"]):
        # For each selection, keep *best* price (highest decimal odds)
        best_rows = []
        for selection, s_grp in grp.groupby("selection"):
            idx = (s_grp["current_odds"].apply(american_to_decimal)).idxmax()
            best_rows.append(s_grp.loc[idx])

        if len(best_rows) < 2:
            continue

        # Compute implied probabilities and see if they sum < 1
        total_prob = 0.0
        for row in best_rows:
            dec = american_to_decimal(row["current_odds"])
            total_prob += 1.0 / dec

        if total_prob < 1.0:
            margin = 1.0 - total_prob
            arb_info = {
                "fixture_id": fixture_id,
                "market": market,
                "edge_pct": round(margin * 100.0, 3),
            }
            # add up to first 3 selections detail
            best_rows_sorted = sorted(
                best_rows,
                key=lambda r: american_to_decimal(r["current_odds"]),
                reverse=True,
            )[:3]
            for i, row in enumerate(best_rows_sorted, start=1):
                arb_info[f"sel{i}_name"] = row["selection"]
                arb_info[f"sel{i}_book"] = row["sportsbook"]
                arb_info[f"sel{i}_odds"] = row["current_odds"]
            arbs.append(arb_info)

    if not arbs:
        return pd.DataFrame()

    return pd.DataFrame(arbs).sort_values("edge_pct", ascending=False)


# ============================================================
#  UI
# ============================================================

def main() -> None:
    st.set_page_config(
        page_title="Edge Force Dominion – Live OpticOdds Stream",
        layout="wide",
    )

    # ------------- HEADER CARD -------------
    st.markdown(
        """
        <div style="padding:1.5rem;border-radius:1.5rem;
                    background:radial-gradient(circle at top left,#14213d,#000814);
                    box-shadow:0 0 40px rgba(0,0,0,0.7);">
          <h1 style="margin:0;font-size:1.9rem;">
            🏆 Edge Force Dominion — Live OpticOdds Stream
          </h1>
          <p style="margin-top:0.4rem;font-size:0.95rem;color:#e0e0e0;">
            Direct from <code>/stream/odds/{sport}</code> using live SSE.
            No fake rows, no demos – pure market feed for US sports.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------- SIDEBAR: CONFIG -------------
    st.sidebar.header("OpticOdds Connection")

    default_key = os.environ.get("OPTICODDS_API_KEY", "")
    api_key = st.sidebar.text_input(
        "API Key",
        value=default_key,
        type="password",
        help="Paste the API key Abe sent you. You can also set OPTICODDS_API_KEY env var.",
    )

    st.sidebar.caption(
        "Tip: keep sportsbooks ≤5 per connection (OpticOdds guide)."
    )

    selected_sportsbooks = st.sidebar.multiselect(
        "Sportsbooks (max ~5 recommended)",
        options=DEFAULT_SPORTSBOOKS,
        default=DEFAULT_SPORTSBOOKS[:5],
    )

    selected_markets = st.sidebar.multiselect(
        "Markets",
        options=DEFAULT_MARKETS,
        default=DEFAULT_MARKETS,
    )

    live_filter = st.sidebar.selectbox(
        "Filter by live vs pre-match",
        options=[
            "Any (no filter)",
            "Prematch only (is_live=false)",
            "Live only (is_live=true)",
        ],
    )
    is_live_param = "any"
    if "false" in live_filter:
        is_live_param = "false"
    elif "true" in live_filter:
        is_live_param = "true"

    max_events = st.sidebar.slider(
        "Max odds events per burst",
        min_value=50,
        max_value=500,
        value=MAX_EVENTS_PER_BURST,
        step=50,
    )

    st.sidebar.markdown("---")
    run_button = st.sidebar.button("🚀 Pull one live burst")

    # ------------- SPORT TABS -------------
    tabs = st.tabs(["🏀 NBA", "🏀 NCAAB", "🏈 NFL", "🏈 NCAAF", "🏒 NHL", "⚾️ MLB"])
    tab_mapping = {
        "🏀 NBA": "NBA",
        "🏀 NCAAB": "NCAAB",
        "🏈 NFL": "NFL",
        "🏈 NCAAF": "NCAAF",
        "🏒 NHL": "NHL",
        "⚾️ MLB": "MLB",
    }

    for tab, (label, league_key) in zip(tabs, tab_mapping.items()):
        with tab:
            sport_path, league_name = SPORT_PATHS[league_key]

            st.subheader(f"{label} – 🔥 Steam / Line Movement")

            if not run_button:
                st.info("Set your API key & filters on the left, then click **Pull one live burst**.")
                continue

            with st.spinner(f"Connecting to OpticOdds {league_name} stream…"):
                df = stream_odds_once(
                    api_key=api_key,
                    sport_path=sport_path,
                    league=league_name,
                    sportsbooks=selected_sportsbooks,
                    markets=selected_markets,
                    is_live=is_live_param,
                    max_events=max_events,
                )

            if df.empty:
                continue

            st.markdown("### 📈 Top line moves (this burst)")
            st.dataframe(
                df[
                    [
                        "fixture_id",
                        "sportsbook",
                        "selection",
                        "market",
                        "open_odds",
                        "current_odds",
                        "line_move",
                        "move_direction",
                    ]
                ].head(50),
                use_container_width=True,
            )

            st.markdown("### ♻️ Raw odds snapshot (aggregated)")
            st.dataframe(df.head(200), use_container_width=True)

            # --------- Arbitrage scan ----------
            st.markdown("### 💰 Basic arbitrage scan (by fixture / market)")
            arb_df = find_basic_arbitrage(df)
            if arb_df.empty:
                st.info("No clear arb between books detected in this burst.")
            else:
                st.dataframe(arb_df.head(50), use_container_width=True)


if __name__ == "__main__":
    main()