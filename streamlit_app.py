import os
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import sseclient
import streamlit as st

# ======================================
# 🔑 CONFIG
# ======================================

OPTICODDS_API_KEY = os.getenv("OPTICODDS_API_KEY", "")
OPTICODDS_BASE = "https://api.opticodds.com/api/v3"

# Sports + leagues to auto-build at startup
SPORTS_CONFIG = {
    "nba":   {"sport": "basketball",    "league": "nba"},
    "ncaab": {"sport": "basketball",    "league": "ncaab"},
    "nfl":   {"sport": "football",      "league": "nfl"},
    "ncaaf": {"sport": "football",      "league": "ncaaf"},
    "mlb":   {"sport": "baseball",      "league": "mlb"},
    "nhl":   {"sport": "ice_hockey",    "league": "nhl"},
}

# Core markets to pull
CORE_MARKETS = ["moneyline", "spread", "total_points"]  # adjust to your account if needed
DEFAULT_SPORTSBOOKS = ["DraftKings", "FanDuel", "Caesars", "BetMGM"]

# Max fixture_ids per /fixtures/odds call (based on docs, usually 5)
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
# 🧮 BASIC MATH HELPERS
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


def minutes_until(start_iso: Optional[str]) -> Optional[float]:
    if not start_iso:
        return None
    try:
        dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return (dt - now).total_seconds() / 60.0
    except Exception:
        return None


# ======================================
# 🌐 OPTICODDS HELPERS
# ======================================

def optic_get(path: str, params: Dict) -> Dict:
    params = dict(params)
    params["key"] = OPTICODDS_API_KEY
    url = f"{OPTICODDS_BASE}{path}"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


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

    Assumes odds payload is a flat list per fixture with fields like:
      - sportsbook
      - market_id or market
      - selection / name
      - price (American)
      - grouping_key
      - points (for spreads/totals)
    You may need to tweak field names based on your actual response.
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


# ======================================
# 🧠 EFD SCORING + METRICS
# ======================================

def compute_efd_score_row(row) -> float:
    """
    EFD – Edge Force Dominion score, 0–100 style.

    Inputs assumed on the row:
      - no_vig_ev     (float, e.g. 0.06 = +6% edge)
      - open_implied  (float)
      - curr_implied  (float)
      - minutes_to_start (float or None)
      - market_id     (string: 'moneyline', 'spread', 'total_points', etc.)
    """
    ev = float(row.get("no_vig_ev") or 0.0)

    open_imp = float(row.get("open_implied") or 0.0)
    curr_imp = float(row.get("curr_implied") or 0.0)
    steam = curr_imp - open_imp
    steam_abs = abs(steam)

    mins = row.get("minutes_to_start")
    try:
        mins = float(mins) if mins is not None else None
    except Exception:
        mins = None

    market_id = (row.get("market_id") or "").lower()

    # 1) Math edge – cap between -10% and +12%
    ev_cap = min(max(ev, -0.10), 0.12)
    math_component = (ev_cap + 0.10) / 0.22  # ~0–1

    # 2) Steam – cap at 10 percentage points movement
    steam_cap = min(steam_abs, 0.10)
    steam_component = steam_cap / 0.10  # 0–1

    # 3) Timing – prefer games inside 6 hours
    if mins is None:
        timing_component = 0.5
    else:
        if mins <= 0:
            timing_component = 1.0
        elif mins <= 360:
            timing_component = 1.0 - (mins / 360.0) * 0.5
        else:
            timing_component = 0.5

    # 4) Market bias (light boost for ML)
    if "moneyline" in market_id:
        market_component = 1.0
    elif "spread" in market_id:
        market_component = 0.9
    else:
        market_component = 0.85

    base = (
        0.45 * math_component +
        0.35 * steam_component +
        0.20 * timing_component
    )

    score = base * 100.0 * market_component
    return float(round(score, 1))


def recompute_ev_and_efd(board_df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute:
      - open_implied (if not set)
      - curr_implied
      - steam
      - fair_prob / no_vig_ev (no-vig pricing)
      - minutes_to_start
      - efd_score
    """
    df = board_df.copy()
    if df.empty:
        return df

    # Open implied gets set only once (if missing or all NaN)
    if "open_implied" not in df.columns or df["open_implied"].isna().all():
        df["open_implied"] = df["price"].apply(implied_prob)

    # Current implied from latest price
    df["curr_implied"] = df["price"].apply(implied_prob)

    # Steam
    df["steam"] = df["curr_implied"] - df["open_implied"]

    # No-vig EV
    df["fair_prob"] = np.nan
    df["no_vig_ev"] = np.nan

    group_cols = ["fixture_id", "market_id", "grouping_key"]
    grouped = df.groupby(group_cols, as_index=False)

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
        upd = pd.concat(updates, ignore_index=True)
        join_cols = ["fixture_id", "market_id", "grouping_key", "selection", "sportsbook"]
        df = df.merge(
            upd[join_cols + ["fair_prob", "no_vig_ev"]],
            on=join_cols,
            how="left",
            suffixes=("", "_upd"),
        )
        for col in ["fair_prob", "no_vig_ev"]:
            df[col] = df[col + "_upd"].combine_first(df[col])
            df.drop(columns=[col + "_upd"], inplace=True)

    # Minutes to start
    df["minutes_to_start"] = df["start_date"].apply(minutes_until)

    # EFD score
    df["efd_score"] = df.apply(compute_efd_score_row, axis=1)

    return df


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
# 🚀 BOOT SEQUENCE
# ======================================

def boot_backend():
    """
    Runs once on startup:
      - loops SPORTS_CONFIG
      - fetches fixtures/active
      - fetches odds (ML/Spread/Total)
      - builds unified board
      - computes EV + EFD
    """
    board_pieces = []
    books = st.session_state["sportsbooks"]

    total_sports = len(SPORTS_CONFIG)
    progress = st.progress(0.0)
    step = 0

    for alias, info in SPORTS_CONFIG.items():
        step += 1
        progress.progress(step / total_sports)
        log_boot(f"[{alias.upper()}] Fetching fixtures/active...")

        try:
            fixtures_df = fetch_active_fixtures(info["sport"], info["league"], days_ahead=2)
        except Exception as e:
            log_boot(f"[{alias.upper()}] Fixtures error: {e}")
            continue

        if fixtures_df.empty:
            log_boot(f"[{alias.upper()}] No active fixtures with odds.")
            continue

        fixture_ids = fixtures_df["fixture_id"].tolist()
        log_boot(f"[{alias.upper()}] Found {len(fixtures_df)} active fixtures.")

        try:
            odds_df = fetch_odds_for_fixtures(fixture_ids, CORE_MARKETS, books)
        except Exception as e:
            log_boot(f"[{alias.upper()}] Odds error: {e}")
            continue

        if odds_df.empty:
            log_boot(f"[{alias.upper()}] No odds returned for ML/Spread/Total.")
            continue

        merged = odds_df.merge(
            fixtures_df[
                ["fixture_id", "home_logo", "away_logo", "status"]
            ],
            on="fixture_id",
            how="left",
        )
        merged["alias"] = alias  # sport alias label
        board_pieces.append(merged)
        log_boot(f"[{alias.upper()}] Added {len(merged)} board rows.")

    if not board_pieces:
        st.session_state["board_df"] = pd.DataFrame()
        st.session_state["boot_done"] = True
        log_boot("Boot finished, but no data was returned.")
        return

    board = pd.concat(board_pieces, ignore_index=True)
    log_boot("Computing fair probabilities, EV, and EFD scores...")
    board = recompute_ev_and_efd(board)
    st.session_state["board_df"] = board
    st.session_state["boot_done"] = True
    log_boot("Boot sequence complete. Board ready.")


# ======================================
# 🧵 LIVE STREAM (SINGLE CHUNK)
# ======================================

def stream_live_odds_chunk(
    sport_id: str,
    league_id: str,
    sportsbooks: List[str],
    markets: List[str],
    max_events: int = 80,
    timeout: int = 10,
) -> List[dict]:
    """
    Pull a single SSE chunk from /stream/odds/{sport} for the given league.
    Returns a list of event payloads (dicts).
    """
    params = {
        "league": league_id,
        "sportsbook": ",".join(sportsbooks),
        "market": ",".join(markets),
        "odds_format": "AMERICAN",
        "key": OPTICODDS_API_KEY,
    }
    url = f"{OPTICODDS_BASE}/stream/odds/{sport_id}"

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


def apply_stream_events(board_df: pd.DataFrame, events: List[dict]) -> pd.DataFrame:
    """
    Map SSE events to board price updates, then recompute EV+EFD.

    NOTE: You may need to tweak the mapping here based on your real stream payload.
    """
    df = board_df.copy()
    if df.empty or not events:
        return df

    for ev in events:
        fixture_id = ev.get("fixture_id") or ev.get("fixture", {}).get("id")
        odds_list = ev.get("odds", [])

        if isinstance(odds_list, dict):
            odds_list = [odds_list]

        for od in odds_list:
            sb = od.get("sportsbook")
            market_id = od.get("market_id") or od.get("market")
            lines = od.get("lines") or [od]  # pattern depends on your data

            for ln in lines:
                sel = ln.get("selection") or ln.get("name")
                gk = ln.get("grouping_key")
                price = ln.get("price") or ln.get("price_american")

                mask = (
                    (df["fixture_id"] == fixture_id) &
                    (df["sportsbook"] == sb) &
                    (df["market_id"] == market_id) &
                    (df["grouping_key"] == gk) &
                    (df["selection"] == sel)
                )

                if mask.any():
                    df.loc[mask, "price"] = price
                    df.loc[mask, "price_decimal"] = american_to_decimal(price)

    # After updating prices → recompute EV + EFD
    df = recompute_ev_and_efd(df)
    return df


# ======================================
# 🧠 COMMAND CONSOLE (READ-ONLY)
# ======================================

def add_message(role: str, text: str):
    st.session_state["messages"].append({"role": role, "text": text})


def handle_command(cmd: str):
    cmd = cmd.strip()
    if not cmd:
        return
    add_message("user", cmd)

    parts = cmd.lower().split()
    if not parts:
        return

    verb = parts[0]
    board = st.session_state["board_df"]

    if verb in ["help", "?"]:
        add_message(
            "system",
            "Commands:\n"
            "- help\n"
            "- show all\n"
            "- show <sport> (nba, nfl, mlb, nhl, ncaab, ncaaf)\n"
            "- show market <ml|spread|total>",
        )
        return

    if board.empty:
        add_message("system", "Board is empty (boot may have failed).")
        return

    if verb == "show" and len(parts) >= 2:
        if parts[1] == "all":
            add_message("system", "Showing all sports (use filters above).")
            return
        if parts[1] == "market" and len(parts) >= 3:
            m = parts[2]
            if m == "ml":
                add_message("system", "Showing moneyline (filter Market = ML).")
            elif m == "spread":
                add_message("system", "Showing spreads.")
            elif m == "total":
                add_message("system", "Showing totals.")
            else:
                add_message("system", f"Unknown market {m}. Use ml/spread/total.")
            return

        sport_alias = parts[1]
        if sport_alias not in SPORTS_CONFIG:
            add_message("system", f"Unknown sport {sport_alias}.")
            return
        add_message("system", f"Showing {sport_alias.upper()} (filter Sport = {sport_alias}).")
        return

    add_message("system", f"Unknown command: {cmd}. Type 'help'.")


def render_console():
    st.subheader("⌨️ Command Console")

    for msg in st.session_state["messages"]:
        cls = "cmd-user" if msg["role"] == "user" else "cmd-system"
        who = "You" if msg["role"] == "user" else "System"
        st.markdown(
            f"<div class='{cls}'><b>{who}:</b> {msg['text']}</div>",
            unsafe_allow_html=True,
        )

    cmd = st.chat_input("Type a command (e.g. 'show nba', 'show market ml', 'help')")
    if cmd is not None:
        handle_command(cmd)


# ======================================
# 🎨 BOARD RENDER
# ======================================

def render_board():
    board = st.session_state["board_df"]
    if board.empty:
        st.error("Board is empty. Check boot log for any API errors.")
        return

    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sport_filter = st.selectbox(
            "Sport",
            ["all"] + list(SPORTS_CONFIG.keys()),
            index=0,
        )
    with col2:
        market_filter = st.selectbox(
            "Market",
            ["all", "moneyline", "spread", "total"],
            index=0,
        )
    with col3:
        book_filter = st.selectbox(
            "Sportsbook",
            ["all"] + sorted(board["sportsbook"].dropna().unique().tolist()),
            index=0,
        )
    with col4:
        sort_mode = st.selectbox(
            "Sort by",
            ["EFD score", "EV", "Steam"],
            index=0,
        )

    df = board.copy()

    if sport_filter != "all":
        df = df[df["alias"] == sport_filter]

    if market_filter != "all":
        if market_filter == "moneyline":
            df = df[df["market_id"].str.contains("moneyline", case=False, na=False)]
        elif market_filter == "spread":
            df = df[df["market_id"].str.contains("spread", case=False, na=False)]
        elif market_filter == "total":
            df = df[df["market_id"].str.contains("total", case=False, na=False)]

    if book_filter != "all":
        df = df[df["sportsbook"] == book_filter]

    if df.empty:
        st.warning("No rows match the current filters.")
        return

    # Sorting
    if sort_mode == "EV":
        df = df.sort_values("no_vig_ev", ascending=False)
    elif sort_mode == "Steam":
        df = df.sort_values("steam", ascending=False)
    else:
        df = df.sort_values("efd_score", ascending=False)

    # --- Live updates button ---
    st.markdown("### 🛰️ Live Updates")

    live_sport = sport_filter if sport_filter != "all" else "nba"
    live_alias_info = SPORTS_CONFIG.get(live_sport)
    live_clicked = st.button(f"Pull live SSE updates for {live_sport.upper()}")

    if live_clicked and live_alias_info is not None:
        try:
            with st.spinner(f"Streaming odds chunk for {live_sport.upper()}..."):
                events = stream_live_odds_chunk(
                    live_alias_info["sport"],
                    live_alias_info["league"],
                    st.session_state["sportsbooks"],
                    CORE_MARKETS,
                )
                if events:
                    new_df = apply_stream_events(board, events)
                    st.session_state["board_df"] = new_df
                    df = new_df.copy()
                    # reapply filters & sorting
                    if sport_filter != "all":
                        df = df[df["alias"] == sport_filter]
                    if market_filter != "all":
                        if market_filter == "moneyline":
                            df = df[df["market_id"].str.contains("moneyline", case=False, na=False)]
                        elif market_filter == "spread":
                            df = df[df["market_id"].str.contains("spread", case=False, na=False)]
                        elif market_filter == "total":
                            df = df[df["market_id"].str.contains("total", case=False, na=False)]
                    if book_filter != "all":
                        df = df[df["sportsbook"] == book_filter]
                    if sort_mode == "EV":
                        df = df.sort_values("no_vig_ev", ascending=False)
                    elif sort_mode == "Steam":
                        df = df.sort_values("steam", ascending=False)
                    else:
                        df = df.sort_values("efd_score", ascending=False)
        except Exception as e:
            st.warning(f"Live stream error: {e}")

    # --- Cards (top fixtures) ---
    st.subheader("🎮 Matchup Cards")

    by_fixture = (
        df.groupby("fixture_id")
        .agg(
            {
                "start_date": "first",
                "home_name": "first",
                "away_name": "first",
                "home_logo": "first",
                "away_logo": "first",
            }
        )
        .reset_index()
    )

    by_fixture = by_fixture.head(8)

    for _, row in by_fixture.iterrows():
        fid = row["fixture_id"]
        subset = df[df["fixture_id"] == fid]

        colA, colB = st.columns([2, 3])

        with colA:
            st.markdown('<div class="fixture-card">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="fixture-header">{row["start_date"]}</div>',
                unsafe_allow_html=True,
            )
            logo_cols = st.columns(2)
            with logo_cols[0]:
                if isinstance(row["home_logo"], str):
                    st.image(row["home_logo"], width=48)
                st.markdown(
                    f'<span class="team-name">{row["home_name"]}</span>',
                    unsafe_allow_html=True,
                )
            with logo_cols[1]:
                if isinstance(row["away_logo"], str):
                    st.image(row["away_logo"], width=48)
                st.markdown(
                    f'<span class="team-name">{row["away_name"]}</span>',
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        with colB:
            st.write("Top prices (Moneyline / Spread / Total):")

            ml = subset[subset["market_id"].str.contains("moneyline", case=False, na=False)]
            sp = subset[subset["market_id"].str.contains("spread", case=False, na=False)]
            tot = subset[subset["market_id"].str.contains("total", case=False, na=False)]

            def summarize(market_df: pd.DataFrame, label: str):
                if market_df.empty:
                    return f"**{label}:** —"
                best = market_df.sort_values("price_decimal", ascending=False).iloc[0]
                return f"**{label}:** {best['sportsbook']} – {best['name']} ({best['price']}, EFD {best['efd_score']:.1f})"

            st.markdown(
                "<div class='odds-row'>"
                + summarize(ml, "Moneyline")
                + "<br>"
                + summarize(sp, "Spread")
                + "<br>"
                + summarize(tot, "Total")
                + "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

    # --- Full board table ---
    st.subheader("📊 Full Board (top 250 rows)")
    show_cols = [
        "alias",
        "start_date",
        "home_name",
        "away_name",
        "sportsbook",
        "market_label",
        "name",
        "price",
        "no_vig_ev",
        "steam",
        "efd_score",
    ]
    tmp = df[show_cols].copy()
    tmp["EV%"] = tmp["no_vig_ev"].apply(lambda v: f"{(v or 0)*100:.1f}%")
    tmp["Steam%"] = tmp["steam"].apply(lambda v: f"{(v or 0)*100:.1f}%")

    tmp = tmp.rename(
        columns={
            "alias": "Sport",
            "start_date": "Start",
            "home_name": "Home",
            "away_name": "Away",
            "sportsbook": "Book",
            "market_label": "Market",
            "name": "Bet",
            "price": "Odds",
            "efd_score": "EFD",
        }
    )

    st.dataframe(
        tmp[
            [
                "Sport",
                "Start",
                "Home",
                "Away",
                "Book",
                "Market",
                "Bet",
                "Odds",
                "EV%",
                "Steam%",
                "EFD",
            ]
        ].head(250),
        use_container_width=True,
    )


# ======================================
# 🧵 MAIN
# ======================================

def main():
    init_state()
    inject_theme()

    st.title("⚡ Edge Force Dominion – Global Odds Engine")

    # Sidebar: API key & sportsbooks
    with st.sidebar:
        st.markdown("### 🔐 OpticOdds API Key")
        key_input = st.text_input(
            "API key",
            value=OPTICODDS_API_KEY,
            type="password",
        )
        if key_input:
            os.environ["OPTICODDS_API_KEY"] = key_input
            globals()["OPTICODDS_API_KEY"] = key_input

        st.markdown("### 📚 Sportsbooks")
        current_books = st.session_state["sportsbooks"]
        books_text = st.text_input(
            "Comma separated sportsbooks",
            value=",".join(current_books),
        )
        st.session_state["sportsbooks"] = [
            b.strip() for b in books_text.split(",") if b.strip()
        ]

        st.markdown("---")
        st.markdown("This app builds the full board on startup, then lets you pull live SSE updates per sport.")

    # Boot sequence
    if not st.session_state["boot_done"]:
        st.subheader("🧩 Booting Edge Force Dominion Board")
        with st.spinner("Pulling fixtures + odds for all sports..."):
            boot_backend()

        st.markdown("#### Boot log")
        for line in st.session_state["boot_log"]:
            st.markdown(f"- {line}")
        st.stop()  # don't render dashboard until boot finished

    # After boot
    st.subheader("🧩 Boot complete – live board active")

    # Quick view of boot log
    with st.expander("View boot log"):
        for line in st.session_state["boot_log"]:
            st.markdown(f"- {line}")

    # Board
    render_board()

    st.markdown("---")

    # Command console (read-only)
    render_console()


if __name__ == "__main__":
    main()