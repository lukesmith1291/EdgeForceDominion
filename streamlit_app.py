import json
import time
from typing import List, Dict, Any, Tuple

import pandas as pd
import requests
import streamlit as st

# ============================================================
#  STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="Edge Force Dominion – Live OpticOdds Stream",
    layout="wide",
)

OPTICODDS_API_BASE = "https://api.opticodds.com/api/v3"

# UI tab label -> OpticOdds league code
# (Use the exact league strings OpticOdds expects)
LEAGUE_MAP = {
    "🏀 NBA":   "NBA",
    "🏀 NCAAB": "NCAAB",
    "🏈 NFL":   "NFL",
    "🏈 NCAAF": "NCAAF",
    "🏒 NHL":   "NHL",
    "⚾ MLB":   "MLB",
}

# ============================================================
#  STYLE
# ============================================================
st.markdown(
    """
<style>
section.main {
    background: radial-gradient(circle at top, #020617 0%, #000000 60%);
    color: #f9fafb;
    font-family: -apple-system, system-ui, BlinkMacSystemFont, "SF Pro Display";
}

.efd-card {
    border-radius: 18px;
    padding: 1.25rem 1.5rem;
    border: 1px solid rgba(96, 165, 250, 0.7);
    background: rgba(15, 23, 42, 0.96);
    box-shadow: 0 0 32px rgba(37, 99, 235, 0.55);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #020617);
    border-right: 1px solid rgba(55, 65, 81, 0.9);
}

.dataframe tbody tr:hover {
    background-color: rgba(59, 130, 246, 0.18) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
#  HELPERS
# ============================================================

def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal odds."""
    try:
        o = float(odds)
    except Exception:
        return 1.0
    return (o / 100.0 + 1.0) if o > 0 else (100.0 / abs(o) + 1.0)


def parse_sse_stream(
    resp: requests.Response,
    max_events: int = 200,
    max_seconds: int = 8,
) -> List[Tuple[str, str]]:
    """
    Very small SSE parser.
    Returns list of (event_name, data_json_string).
    """
    events: List[Tuple[str, str]] = []
    event_name = None
    data_lines: List[str] = []
    start = time.time()

    for raw in resp.iter_lines(decode_unicode=True):
        if time.time() - start > max_seconds:
            break
        if raw is None:
            continue
        line = raw.strip()
        if not line:
            # dispatch current event
            if event_name and data_lines:
                events.append((event_name, "\n".join(data_lines)))
                if len(events) >= max_events:
                    break
            event_name = None
            data_lines = []
            continue

        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())

    return events


def stream_odds_once(
    api_key: str,
    league: str,
    sportsbooks: List[str],
    markets: List[str],
    is_live: str,
    max_events: int = 150,
) -> pd.DataFrame:
    """
    Hit /stream/odds once using *league only*,
    obeying rule: exactly ONE of sport, league, or fixture_id.

    We ALWAYS send league (NBA, NFL, etc) and NEVER send sport or fixture_id.

    Returns a DataFrame with:
        fixture_id, sportsbook, market, selection,
        league, sport, open_odds, current_odds,
        line_move, move_abs, move_direction
    """
    if not api_key:
        st.error("No OpticOdds API key set. Paste it in the sidebar and press 'Test & Use Key'.")
        return pd.DataFrame()

    url = f"{OPTICODDS_API_BASE}/stream/odds"

    params: Dict[str, Any] = {
        "key": api_key,
        "league": league,                     # ✅ EXACTLY ONE selector
        "sportsbook": sportsbooks or [],
        "market": markets or ["Moneyline"],
    }

    if is_live in ("true", "false"):
        params["is_live"] = is_live

    resp: requests.Response | None = None
    try:
        resp = requests.get(url, params=params, stream=True, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        msg = f"OpticOdds: {e}"
        if hasattr(e, "response") and e.response is not None:
            try:
                err = e.response.text
                msg = f"OpticOdds: HTTP {e.response.status_code}: {err}"
            except Exception:
                pass
        st.error(msg)
        if resp is not None:
            resp.close()
        return pd.DataFrame()

    events = parse_sse_stream(resp, max_events=max_events)
    resp.close()

    rows: List[Dict[str, Any]] = []
    for ev_name, data_str in events:
        if ev_name not in ("odds", "locked-odds"):
            continue
        try:
            payload = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        for odd in payload.get("data", []):
            rows.append(
                dict(
                    event_type=ev_name,
                    fixture_id=odd.get("fixture_id"),
                    game_id=odd.get("game_id"),
                    sport=odd.get("sport"),
                    league=odd.get("league"),
                    market=odd.get("market"),
                    selection=odd.get("selection") or odd.get("name"),
                    sportsbook=odd.get("sportsbook"),
                    price=odd.get("price"),
                    timestamp=odd.get("timestamp"),
                )
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["price", "timestamp"])
    df = df.sort_values("timestamp")

    key_cols = ["fixture_id", "sportsbook", "market", "selection"]
    agg_rows: List[Dict[str, Any]] = []

    for key, grp in df.groupby(key_cols):
        first = grp.iloc[0]
        last = grp.iloc[-1]
        open_price = float(first["price"])
        curr_price = float(last["price"])
        move = curr_price - open_price

        agg_rows.append(
            dict(
                fixture_id=key[0],
                sportsbook=key[1],
                market=key[2],
                selection=key[3],
                league=str(last["league"]),
                sport=str(last["sport"]),
                open_odds=open_price,
                current_odds=curr_price,
                line_move=move,
                move_abs=abs(move),
            )
        )

    if not agg_rows:
        return pd.DataFrame()

    out = pd.DataFrame(agg_rows)
    out["move_direction"] = out["line_move"].apply(
        lambda x: "Steam to dog" if x > 0 else ("Steam to fav" if x < 0 else "Flat")
    )
    return out.sort_values("move_abs", ascending=False)


def detect_arbitrage(df: pd.DataFrame, min_edge_pct: float) -> pd.DataFrame:
    """
    Simple 2-way arbitrage finder.
    For each fixture + market:
        - take best price on each selection across sportsbooks
        - if 1/d1 + 1/d2 < 1 → arbitrage exists
    """
    if df.empty:
        return df

    records: List[Dict[str, Any]] = []
    grouped = df.groupby(["fixture_id", "market"])

    for (fixture_id, market), grp in grouped:
        # best price per side
        best = (
            grp.sort_values("current_odds", ascending=False)
            .groupby("selection")
            .head(1)
        )
        if best["selection"].nunique() != 2:
            continue

        recs = list(best.to_dict("records"))
        a, b = recs[0], recs[1]

        d1 = american_to_decimal(a["current_odds"])
        d2 = american_to_decimal(b["current_odds"])
        inv_sum = 1.0 / d1 + 1.0 / d2
        if inv_sum >= 1.0:
            continue

        edge = (1.0 - inv_sum) * 100.0
        if edge < min_edge_pct:
            continue

        records.append(
            dict(
                fixture_id=fixture_id,
                market=market,
                selection1=a["selection"],
                book1=a["sportsbook"],
                odds1=a["current_odds"],
                selection2=b["selection"],
                book2=b["sportsbook"],
                odds2=b["current_odds"],
                arb_edge_pct=round(edge, 2),
            )
        )

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).sort_values("arb_edge_pct", ascending=False)


def test_key_once(api_key: str) -> Tuple[bool, str]:
    """
    Cheap smoke test: hit /sportsbooks/active with ?key=.
    If this passes, the key is valid.
    """
    if not api_key:
        return False, "No key provided."

    try:
        url = f"{OPTICODDS_API_BASE}/sportsbooks/active"
        resp = requests.get(url, params={"key": api_key}, timeout=10)
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
        return True, "Key accepted by /sportsbooks/active."
    except Exception as e:
        return False, str(e)

# ============================================================
#  SIDEBAR – CONTROLS
# ============================================================

st.sidebar.markdown("## ⚙️ Edge Force Dominion Controls")

if "optic_key" not in st.session_state:
    st.session_state["optic_key"] = ""
if "key_status" not in st.session_state:
    st.session_state["key_status"] = None
if "key_message" not in st.session_state:
    st.session_state["key_message"] = ""

key_input = st.sidebar.text_input("OpticOdds API Key", type="password")

if st.sidebar.button("🔌 Test & Use Key"):
    st.session_state["optic_key"] = key_input.strip()
    ok, msg = test_key_once(st.session_state["optic_key"])
    st.session_state["key_status"] = ok
    st.session_state["key_message"] = msg

mode = st.sidebar.radio("Mode", ["🔥 Steam / Line Move", "♟️ Arbitrage Radar"])

books_raw = st.sidebar.text_input(
    "Sportsbooks (comma-separated, max 5 per request)",
    "FanDuel,DraftKings,BetMGM,Caesars,LowVig",
)
books = [b.strip() for b in books_raw.split(",") if b.strip()]
books = books[:5]  # OpticOdds limit per request

min_move = st.sidebar.slider("Min line move (cents)", 5, 100, 20)
min_arb_edge = st.sidebar.slider("Min arb edge (%)", 0.1, 5.0, 0.5)
burst_events = st.sidebar.slider(
    "Events per refresh (per league)", 40, 400, 160, step=40
)

live_filter = st.sidebar.selectbox(
    "Odds type",
    ["Both prematch + live", "Prematch only", "Live only"],
)
if live_filter == "Prematch only":
    is_live = "false"
elif live_filter == "Live only":
    is_live = "true"
else:
    is_live = ""

if st.sidebar.button("🔄 Pull fresh stream burst"):
    st.rerun()

active_key = st.session_state["optic_key"]

# ============================================================
#  HEADER
# ============================================================

st.markdown(
    """
<div class="efd-card">
  <h1 style="margin-bottom:0.25rem;">🏆 Edge Force Dominion — Live OpticOdds Stream</h1>
  <p style="opacity:0.9;margin-bottom:0;">
    Direct from <code>/stream/odds?league=...</code> using live SSE for each US league tab.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

if st.session_state["key_status"] is True:
    st.success(f"OpticOdds: {st.session_state['key_message']}")
elif st.session_state["key_status"] is False:
    st.error(f"OpticOdds: {st.session_state['key_message']}")
else:
    st.info("Paste your key and hit '🔌 Test & Use Key' to verify access.")

st.markdown("---")

# ============================================================
#  TABS PER LEAGUE
# ============================================================

tabs = st.tabs(list(LEAGUE_MAP.keys()))

for tab_label, tab in zip(LEAGUE_MAP.keys(), tabs):
    with tab:
        league_code = LEAGUE_MAP[tab_label]

        st.markdown(f"### {tab_label} — {mode}")

        if not active_key:
            st.warning("No API key set.")
            continue

        df = stream_odds_once(
            api_key=active_key,
            league=league_code,
            sportsbooks=books,
            markets=["Moneyline"],
            is_live=is_live,
            max_events=burst_events,
        )

        if df.empty:
            st.info("No odds came through in this burst (for this league / filters).")
            continue

        if mode.startswith("🔥"):
            steam_df = df[df["move_abs"] >= min_move]
            if steam_df.empty:
                st.success("No steam above your threshold in this burst.")
            else:
                view = steam_df[
                    [
                        "league",
                        "market",
                        "selection",
                        "sportsbook",
                        "open_odds",
                        "current_odds",
                        "line_move",
                        "move_direction",
                    ]
                ]
                st.dataframe(view, use_container_width=True, hide_index=True)
        else:
            arb_df = detect_arbitrage(df, min_arb_edge)
            if arb_df.empty:
                st.success("No clear 2-way arbitrage in this burst.")
            else:
                st.dataframe(arb_df, use_container_width=True, hide_index=True)