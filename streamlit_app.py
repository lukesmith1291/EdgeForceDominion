# streamlit_app.py
# EDGE FORCE DOMINION – OpticOdds v3 Global Odds Engine

import os, json, threading, queue, time, requests, sseclient
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ------------------------------------------------------------------
# 🔑 CONFIG
# ------------------------------------------------------------------
OPTICODDS_BASE = "https://api.opticodds.com/api/v3"

SPORTS_CONFIG = {
    "nba":   {"sport": "basketball", "league": "nba"},
    "ncaab": {"sport": "basketball", "league": "ncaab"},
    "nfl":   {"sport": "football",   "league": "nfl"},
    "ncaaf": {"sport": "football",   "league": "ncaaf"},
    "mlb":   {"sport": "baseball",   "league": "mlb"},
    "nhl":   {"sport": "ice_hockey", "league": "nhl"},
}

CORE_MARKETS      = ["moneyline", "spread", "total_points"]
DEFAULT_BOOKS     = ["DraftKings", "FanDuel", "Caesars", "BetMGM"]
MAX_FIXTURES_CALL = 5           # max fixture_ids per /fixtures/odds call
SPORT_DELAY       = 1.0         # delay between sports on boot
CHUNK_DELAY       = 0.5         # delay between odds chunks

# ------------------------------------------------------------------
# 🔐 API-KEY HANDLER
# ------------------------------------------------------------------
def get_api_key() -> str:
    try:  # Streamlit Cloud secret
        return st.secrets["opticodds"]["api_key"]
    except Exception:
        return os.getenv("OPTICODDS_API_KEY", st.session_state.get("api_key", ""))

# ------------------------------------------------------------------
# 🎨 THEME
# ------------------------------------------------------------------
def inject_theme():
    st.markdown(
        """
        <style>
        .stApp{
          background:radial-gradient(circle at top left,#020617 0,#000 40%,#020617 100%);
          color:#e2f3ff
        }
        h1,h2,h3{
          color:#e5f0ff;
          text-shadow:0 0 12px rgba(96,165,250,.85)
        }
        .fixture-card{
          border-radius:16px;
          padding:12px 14px;
          background:linear-gradient(135deg,rgba(15,118,110,.15),rgba(59,130,246,.1));
          border:1px solid rgba(148,163,184,.5);
          box-shadow:0 0 18px rgba(56,189,248,.25)
        }
        .fixture-header{
          font-size:.9rem;
          text-transform:uppercase;
          letter-spacing:.08em;
          color:#a5b4fc;
          margin-bottom:4px
        }
        .team-name{font-weight:600}
        .odds-row{font-size:.85rem;color:#e2e8f0}
        .cmd-user{color:#93c5fd}
        .cmd-system{color:#a5b4fc}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------
# 🧮 MATH HELPERS
# ------------------------------------------------------------------
def american_to_decimal(odds: float) -> float:
    try:
        o = float(odds)
    except Exception:
        return 1.0
    if o == 0:
        return 1.0
    return 1 + o / 100 if o > 0 else 1 + 100 / abs(o)


def implied_prob(odds: float) -> float:
    dec = american_to_decimal(odds)
    return 1 / dec if dec > 1 else 0.0


def minutes_until(iso: Optional[str]) -> Optional[float]:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return (dt - datetime.now(timezone.utc)).total_seconds() / 60
    except Exception:
        return None

# ------------------------------------------------------------------
# 🔐 SESSION STATE
# ------------------------------------------------------------------
def init_state():
    defaults = dict(
        boot_done=False,
        boot_log=[],
        board_df=pd.DataFrame(),   # current odds board
        history_df=pd.DataFrame(), # SSE history (for line moves)
        messages=[],
        sportsbooks=DEFAULT_BOOKS.copy(),
        api_key=get_api_key(),
        sse_queue=queue.Queue(),
        sse_running=False,
        sse_thread=None,
        last_update=None,
        current_sport="nba",
    )
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def log_boot(msg: str):
    st.session_state["boot_log"].append(msg)

# ------------------------------------------------------------------
# 🌐 OPTICODDS HELPERS (SNAPSHOT)
# ------------------------------------------------------------------
def optic_get(path: str, params: Dict) -> Dict:
    params = dict(params)
    key = st.session_state["api_key"]
    if not key:
        raise ValueError("❌ No OpticOdds API key configured.")
    params["key"] = key
    url = f"{OPTICODDS_BASE}{path}"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_fixtures(sport_id: str, league_id: str, days: int = 2) -> pd.DataFrame:
    """
    Pull active fixtures for a sport/league in the next N days.
    Uses /fixtures/active which you already had working.
    """
    now = datetime.now(timezone.utc)
    before = now + timedelta(days=days)
    try:
        resp = optic_get(
            "/fixtures/active",
            {
                "sport": sport_id,
                "league": league_id,
                "start_date_after": now.isoformat().replace("+00:00", "Z"),
                "start_date_before": before.isoformat().replace("+00:00", "Z"),
            },
        )
    except ValueError as e:
        st.error(str(e))
        return pd.DataFrame()

    rows = []
    for fx in resp.get("data", []):
        try:
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
        except Exception as e:
            log_boot(f"Bad fixture {fx.get('id')}: {e}")
    return pd.DataFrame(rows)


def fetch_odds_chunks(
    fixture_ids: List[str],
    markets: List[str],
    books: List[str],
    status_placeholder,
) -> pd.DataFrame:
    """
    Chunked /fixtures/odds calls for a list of fixture_ids.
    """
    if not fixture_ids:
        return pd.DataFrame()

    rows = []
    total = (len(fixture_ids) + MAX_FIXTURES_CALL - 1) // MAX_FIXTURES_CALL

    for idx, i in enumerate(range(0, len(fixture_ids), MAX_FIXTURES_CALL), 1):
        chunk = fixture_ids[i : i + MAX_FIXTURES_CALL]
        status_placeholder.text(f"📡 Odds chunk {idx}/{total}")
        try:
            resp = optic_get(
                "/fixtures/odds",
                {
                    "fixture_id": chunk,
                    "sportsbook": books,
                    "market": markets,
                    "odds_format": "AMERICAN",
                },
            )

            for fx in resp.get("data", []):
                fid = fx["id"]
                sd = fx.get("start_date")
                hname = fx.get("home_team_display")
                aname = fx.get("away_team_display")
                sport = fx.get("sport", {}).get("id")
                league = fx.get("league", {}).get("id")

                for od in fx.get("odds", []):
                    rows.append(
                        {
                            "fixture_id": fid,
                            "sport": sport,
                            "league": league,
                            "start_date": sd,
                            "home_name": hname,
                            "away_name": aname,
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

            if i + MAX_FIXTURES_CALL < len(fixture_ids):
                time.sleep(CHUNK_DELAY)

        except Exception as e:
            log_boot(f"Chunk {idx} error: {e}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["price_decimal"] = df["price"].apply(american_to_decimal)
    df["implied_prob"] = df["price"].apply(implied_prob)
    df["open_price"] = df["price"]
    df["open_implied"] = df["implied_prob"]
    df["timestamp"] = datetime.utcnow().isoformat()
    return df

# ------------------------------------------------------------------
# 🧮 EV + EFD SCORING
# ------------------------------------------------------------------
def efd_row(row) -> float:
    ev = float(row.get("no_vig_ev") or 0)
    open_i = float(row.get("open_implied") or 0)
    curr_i = float(row.get("curr_implied") or 0)
    steam = abs(curr_i - open_i)
    mins = row.get("minutes_to_start")
    try:
        mins = float(mins) if mins is not None else None
    except Exception:
        mins = None
    market = (row.get("market_id") or "").lower()

    ev_c = min(max(ev, -0.1), 0.12)
    math_comp = (ev_c + 0.1) / 0.22
    steam_c = min(steam, 0.1) / 0.1
    if mins is None:
        time_c = 0.5
    elif mins <= 0:
        time_c = 1.0
    elif mins <= 360:
        time_c = 1 - (mins / 360) * 0.5
    else:
        time_c = 0.5

    market_c = 1.0 if "moneyline" in market else 0.9 if "spread" in market else 0.85
    score = (0.45 * math_comp + 0.35 * steam_c + 0.20 * time_c) * 100 * market_c
    return float(round(score, 1))


def recompute_ev_efd(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df["curr_implied"] = df["price"].apply(implied_prob)
    df["steam"] = df["curr_implied"] - df["open_implied"]
    df["fair_prob"] = df["no_vig_ev"] = np.nan

    grp_cols = ["fixture_id", "market_id", "grouping_key"]

    updates = []
    for _, g in df.groupby(grp_cols, as_index=False):
        best = (
            g.sort_values("price_decimal", ascending=False)
            .groupby("selection", as_index=False)
            .first()
        )
        if len(best) < 2:
            continue

        inv = (1 / best["price_decimal"]).sum()
        if inv <= 0:
            continue

        best["fair_prob"] = (1 / best["price_decimal"]) / inv
        best["no_vig_ev"] = best["price_decimal"] * best["fair_prob"] - 1
        updates.append(best)

    if updates:
        upd = pd.concat(updates, ignore_index=True)
        df = df.merge(
            upd[grp_cols + ["selection", "sportsbook", "fair_prob", "no_vig_ev"]],
            on=grp_cols + ["selection", "sportsbook"],
            how="left",
            suffixes=("", "_u"),
        )
        df["fair_prob"] = df["fair_prob_u"].combine_first(df["fair_prob"])
        df["no_vig_ev"] = df["no_vig_ev_u"].combine_first(df["no_vig_ev"])
        df = df.drop(columns=[c for c in df.columns if c.endswith("_u")])

    df["minutes_to_start"] = df["start_date"].apply(minutes_until)
    df["efd_score"] = df.apply(efd_row, axis=1)
    return df

# ------------------------------------------------------------------
# 🧵 BOOT SNAPSHOT
# ------------------------------------------------------------------
def boot_backend():
    log_boot("Booting Edge Force Dominion…")
    all_dfs = []
    total = len(SPORTS_CONFIG)
    status = st.empty()

    for idx, (alias, cfg) in enumerate(SPORTS_CONFIG.items(), 1):
        sport, league = cfg["sport"], cfg["league"]
        log_boot(f"[{idx}/{total}] Fixtures → {alias.upper()}")
        fixtures = fetch_fixtures(sport, league, days=2)
        if fixtures.empty:
            log_boot(f"No fixtures for {alias}")
            continue

        fids = fixtures["fixture_id"].tolist()
        log_boot(f"Found {len(fids)} fixtures for {alias}")
        odds = fetch_odds_chunks(
            fids, CORE_MARKETS, st.session_state["sportsbooks"], status
        )
        if odds.empty:
            log_boot(f"No odds for {alias}")
            continue

        odds = odds.merge(
            fixtures[
                [
                    "fixture_id",
                    "home_logo",
                    "away_logo",
                    "status",
                ]
            ],
            on="fixture_id",
            how="left",
        )

        odds["alias"] = alias
        all_dfs.append(odds)

        if idx < total:
            time.sleep(SPORT_DELAY)

    if not all_dfs:
        log_boot("Zero data returned – check API key")
        st.session_state["board_df"] = pd.DataFrame()
        return

    full = pd.concat(all_dfs, ignore_index=True)
    log_boot("Computing EV + EFD…")
    full = recompute_ev_efd(full)

    st.session_state["board_df"] = full
    st.session_state["history_df"] = full.copy()
    st.session_state["boot_done"] = True
    st.session_state["last_update"] = datetime.utcnow()
    log_boot("Boot complete ✅")

# ------------------------------------------------------------------
# 🌐 SSE – LIVE ODDS STREAM
# ------------------------------------------------------------------
def _sse_worker(sport_alias: str):
    """
    Background thread: listen to /stream/odds/{sport} and push updates into a queue.
    Adjust the URL/fields here if OpticOdds' docs differ.
    """
    cfg = SPORTS_CONFIG[sport_alias]
    api_key = st.session_state["api_key"]
    q = st.session_state["sse_queue"]

    params = {
        "key": api_key,
        "league": cfg["league"],
        "sportsbook": st.session_state["sportsbooks"],
        "market": CORE_MARKETS,
        "odds_format": "AMERICAN",
    }

    # If docs say it's /stream/odds?sport=, change this line only.
    url = f"{OPTICODDS_BASE}/stream/odds/{cfg['sport']}"

    try:
        with requests.get(url, params=params, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            client = sseclient.SSEClient(resp)

            for event in client.events():
                if not st.session_state.get("sse_running"):
                    break
                if not event.data:
                    continue
                try:
                    payload = json.loads(event.data)
                    q.put(payload)
                except Exception:
                    continue
    except Exception as e:
        q.put({"_error": f"SSE error for {sport_alias}: {e}"})
    finally:
        # thread will naturally die when flag flipped
        st.session_state["sse_running"] = False


def start_sse_streaming(sport_alias: str):
    if st.session_state["sse_running"]:
        return
    st.session_state["sse_running"] = True
    st.session_state["current_sport"] = sport_alias
    th = threading.Thread(target=_sse_worker, args=(sport_alias,), daemon=True)
    st.session_state["sse_thread"] = th
    th.start()


def stop_sse_streaming():
    st.session_state["sse_running"] = False
    th = st.session_state.get("sse_thread")
    if th and th.is_alive():
        # worker will exit on next event loop when it sees sse_running=False
        pass


def process_pending_updates():
    """
    Read SSE queue, update board_df prices, recompute EV/EFD.
    """
    board = st.session_state["board_df"]
    if board.empty:
        return

    q = st.session_state["sse_queue"]
    changed = False

    while not q.empty():
        msg = q.get()
        if "_error" in msg:
            log_boot(msg["_error"])
            continue

        data = msg.get("data") or msg

        # Try to be robust to shape; adjust here once you see real SSE payloads.
        fx = data.get("fixture") or {}
        fid = data.get("fixture_id") or fx.get("id")
        sportsbook = data.get("sportsbook")
        selection = data.get("selection")
        market_id = data.get("market_id") or data.get("market")
        grouping_key = data.get("grouping_key")
        price = data.get("price")

        if not (fid and sportsbook and selection and market_id and price is not None):
            continue

        mask = (
            (board["fixture_id"] == fid)
            & (board["sportsbook"] == sportsbook)
            & (board["selection"] == selection)
            & (board["market_id"] == market_id)
        )
        if grouping_key is not None and "grouping_key" in board.columns:
            mask &= board["grouping_key"].eq(grouping_key)

        if not mask.any():
            # If it's a brand new leg we don't know, ignore for now.
            continue

        idx = board.index[mask][0]
        board.at[idx, "price"] = price
        board.at[idx, "price_decimal"] = american_to_decimal(price)
        changed = True

        # Append to history for line-move analytics
        hist = st.session_state["history_df"]
        row_copy = board.loc[[idx]].copy()
        row_copy["timestamp"] = datetime.utcnow().isoformat()
        st.session_state["history_df"] = pd.concat(
            [hist, row_copy], ignore_index=True
        )

    if changed:
        board = recompute_ev_efd(board)
        st.session_state["board_df"] = board
        st.session_state["last_update"] = datetime.utcnow()

# ------------------------------------------------------------------
# 💰 ARBITRAGE ENGINE
# ------------------------------------------------------------------
def compute_arbitrage(df: pd.DataFrame, min_edge_pct: float = 0.25) -> pd.DataFrame:
    """
    Scan current board for cross-book arbitrage.
    min_edge_pct is net profit % threshold to show.
    """
    if df.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["fixture_id", "market_id", "grouping_key"]

    for (_, market_id, gk), g in df.groupby(group_cols):
        if g.empty:
            continue

        alias = g["alias"].iloc[0]
        start = g["start_date"].iloc[0]
        home = g["home_name"].iloc[0]
        away = g["away_name"].iloc[0]
        market_label = g["market_label"].iloc[0]

        # Best price per selection across books
        best = (
            g.sort_values("price_decimal", ascending=False)
            .groupby("selection", as_index=False)
            .first()
        )
        if len(best) < 2:
            continue

        implied_sum = (1 / best["price_decimal"]).sum()
        if implied_sum >= 1:
            continue

        edge = (1 / implied_sum - 1) * 100
        if edge < min_edge_pct:
            continue

        legs = []
        for _, r in best.iterrows():
            legs.append(
                {
                    "selection": r["selection"],
                    "book": r["sportsbook"],
                    "price": r["price"],
                    "dec": r["price_decimal"],
                }
            )

        rows.append(
            {
                "fixture_id": g["fixture_id"].iloc[0],
                "alias": alias,
                "start_date": start,
                "home_name": home,
                "away_name": away,
                "market_label": market_label,
                "edge_pct": round(edge, 2),
                "legs": legs,
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("edge_pct", ascending=False)

# ------------------------------------------------------------------
# ⌨️ CONSOLE
# ------------------------------------------------------------------
def add_msg(role: str, text: str):
    st.session_state["messages"].append({"role": role, "text": text})


def handle_cmd(cmd: str):
    cmd = cmd.strip()
    if not cmd:
        return
    add_msg("user", cmd)
    parts = cmd.lower().split()
    verb = parts[0]
    board = st.session_state["board_df"]

    if verb in {"help", "?"}:
        add_msg(
            "system",
            "Commands:\n- help\n- show all\n- show <sport>\n- show market <ml|spread|total>",
        )
        return
    if board.empty:
        add_msg("system", "Board empty – boot may have failed")
        return
    if verb == "show" and len(parts) >= 2:
        if parts[1] == "all":
            add_msg("system", "Showing all (use filters)")
            return
        if parts[1] == "market" and len(parts) >= 3:
            m = parts[2]
            add_msg("system", f"Filter Market = {m}")
            return
        alias = parts[1]
        if alias not in SPORTS_CONFIG:
            add_msg("system", f"Unknown sport {alias}")
            return
        add_msg("system", f"Filter Sport = {alias}")
        return
    add_msg("system", "Unknown cmd – try 'help'")


def render_console():
    st.subheader("⌨️ Command Console")
    for m in st.session_state["messages"]:
        cls = "cmd-user" if m["role"] == "user" else "cmd-system"
        who = "You" if m["role"] == "user" else "System"
        st.markdown(
            f"<div class='{cls}'><b>{who}:</b> {m['text']}</div>",
            unsafe_allow_html=True,
        )
    cmd = st.chat_input("Type command…")
    if cmd:
        handle_cmd(cmd)

# ------------------------------------------------------------------
# 🎮 DASHBOARD – MAIN BOARD
# ------------------------------------------------------------------
def render_board():
    board = st.session_state["board_df"]
    if board.empty:
        st.error("Board empty – check boot log")
        return

    # status / controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.session_state["sse_running"]:
            st.success("🟢 LIVE")
            if st.session_state["last_update"]:
                st.caption(
                    f"Last update: {st.session_state['last_update'].strftime('%H:%M:%S UTC')}"
                )
        else:
            st.error("🔴 Offline")
    with col2:
        if st.session_state["sse_running"]:
            if st.button("⏸️ Pause live"):
                stop_sse_streaming()
                st.rerun()
        else:
            if st.button("▶️ Start live"):
                start_sse_streaming(st.session_state["current_sport"])
                st.rerun()
    with col3:
        sport_sel = st.selectbox(
            "Live Sport",
            list(SPORTS_CONFIG.keys()),
            index=list(SPORTS_CONFIG.keys()).index(
                st.session_state["current_sport"]
            ),
            key="live_sport",
        )
        if sport_sel != st.session_state["current_sport"]:
            stop_sse_streaming()
            start_sse_streaming(sport_sel)
            st.rerun()

    st.markdown("---")

    # filters
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sport_f = st.selectbox("Sport", ["all"] + list(SPORTS_CONFIG.keys()))
    with c2:
        market_f = st.selectbox("Market", ["all", "moneyline", "spread", "total"])
    with c3:
        book_f = st.selectbox(
            "Book", ["all"] + sorted(board["sportsbook"].dropna().unique())
        )
    with c4:
        sort = st.selectbox("Sort", ["EFD score", "EV", "Steam"])

    # apply SSE updates
    if st.session_state["sse_running"]:
        process_pending_updates()

    df = board.copy()
    if sport_f != "all":
        df = df[df["alias"] == sport_f]
    if market_f != "all":
        df = df[df["market_id"].str.contains(market_f, case=False, na=False)]
    if book_f != "all":
        df = df[df["sportsbook"] == book_f]
    if df.empty:
        st.warning("No rows match filters")
        return

    if sort == "EV":
        df = df.sort_values("no_vig_ev", ascending=False)
    elif sort == "Steam":
        df = df.sort_values("steam", ascending=False)
    else:
        df = df.sort_values("efd_score", ascending=False)

    # matchup cards (top 8 fixtures)
    st.subheader("🎮 Matchup Cards")
    card_fixtures = (
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
        .head(8)
    )

    for _, row in card_fixtures.iterrows():
        a, b = st.columns([2, 3])
        with a:
            st.markdown('<div class="fixture-card">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="fixture-header">{row["start_date"]}</div>',
                unsafe_allow_html=True,
            )
            lc1, lc2 = st.columns(2)
            with lc1:
                if row["home_logo"]:
                    st.image(row["home_logo"], width=48)
                st.markdown(
                    f'<span class="team-name">{row["home_name"]}</span>',
                    unsafe_allow_html=True,
                )
            with lc2:
                if row["away_logo"]:
                    st.image(row["away_logo"], width=48)
                st.markdown(
                    f'<span class="team-name">{row["away_name"]}</span>',
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        with b:
            st.write("Top prices (ML / Spread / Total):")
            sub = df[df["fixture_id"] == row["fixture_id"]]
            ml = sub[
                sub["market_id"].str.contains("moneyline", case=False, na=False)
            ]
            sp = sub[sub["market_id"].str.contains("spread", case=False, na=False)]
            tot = sub[sub["market_id"].str.contains("total", case=False, na=False)]

            def summarise(mdf, label):
                if mdf.empty:
                    return f"**{label}:** —"
                top = mdf.sort_values("price_decimal", ascending=False).iloc[0]
                return (
                    f"**{label}:** {top['sportsbook']} – {top['name']} "
                    f"({top['price']}, EFD {top['efd_score']:.1f})"
                )

            st.markdown(
                "<div class='odds-row'>"
                + summarise(ml, "Moneyline")
                + "<br>"
                + summarise(sp, "Spread")
                + "<br>"
                + summarise(tot, "Total")
                + "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

    # table
    st.subheader("📊 Full Board (top 250)")
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
    tmp = df[show_cols].rename(
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
    tmp["EV%"] = tmp["no_vig_ev"].apply(lambda v: f"{(v or 0) * 100:.1f}%")
    tmp["Steam%"] = tmp["steam"].apply(lambda v: f"{(v or 0) * 100:.1f}%")
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

# ------------------------------------------------------------------
# 💰 ARBITRAGE TAB
# ------------------------------------------------------------------
def render_arbitrage():
    df = st.session_state["board_df"]
    if df.empty:
        st.warning("No board data yet.")
        return

    arb_df = compute_arbitrage(df, min_edge_pct=0.25)
    st.subheader("💰 Live Arbitrage Opportunities")

    if arb_df.empty:
        st.info("No true arbitrage found at the moment (edge < 0.25%).")
        return

    for _, row in arb_df.head(50).iterrows():
        st.markdown("----")
        st.markdown(
            f"**{row['alias'].upper()} – {row['home_name']} vs {row['away_name']}**  "
        )
        st.caption(f"{row['start_date']}  |  {row['market_label']}")
        st.markdown(f"**Edge:** {row['edge_pct']:.2f}%")
        for leg in row["legs"]:
            st.markdown(
                f"- {leg['selection']} – {leg['book']} @ {leg['price']} "
                f"(dec {leg['dec']:.3f})"
            )

# ------------------------------------------------------------------
# 📈 LINE MOVES TAB
# ------------------------------------------------------------------
def render_line_moves():
    df = st.session_state["board_df"]
    if df.empty:
        st.warning("No board data yet.")
        return

    st.subheader("📈 Steam & Reverse Line Movement")

    # label direction: positive steam = to dog, negative = to fav
    df = df.copy()
    df["steam_bp"] = df["steam"] * 10000
    df["direction"] = np.where(
        df["steam"] > 0, "Steam to underdog", np.where(df["steam"] < 0, "Steam to favorite", "Flat")
    )

    sport_f = st.selectbox("Sport", ["all"] + list(SPORTS_CONFIG.keys()), key="lm_sport")
    market_f = st.selectbox(
        "Market", ["all", "moneyline", "spread", "total"], key="lm_market"
    )

    if sport_f != "all":
        df = df[df["alias"] == sport_f]
    if market_f != "all":
        df = df[df["market_id"].str.contains(market_f, case=False, na=False)]

    if df.empty:
        st.warning("No rows match filters.")
        return

    # show strongest moves
    df = df.sort_values("steam_bp", ascending=False)

    cols = [
        "alias",
        "start_date",
        "home_name",
        "away_name",
        "sportsbook",
        "market_label",
        "name",
        "price",
        "open_price",
        "steam",
        "direction",
    ]
    out = df[cols].rename(
        columns={
            "alias": "Sport",
            "start_date": "Start",
            "home_name": "Home",
            "away_name": "Away",
            "sportsbook": "Book",
            "market_label": "Market",
            "name": "Bet",
            "price": "Current",
            "open_price": "Open",
            "steam": "SteamΔ",
            "direction": "Direction",
        }
    )
    out["SteamΔ (bps)"] = (out["SteamΔ"] * 10000).round(1)
    st.dataframe(
        out[
            [
                "Sport",
                "Start",
                "Home",
                "Away",
                "Book",
                "Market",
                "Bet",
                "Open",
                "Current",
                "SteamΔ (bps)",
                "Direction",
            ]
        ].head(250),
        use_container_width=True,
    )

# ------------------------------------------------------------------
# 📊 ANALYTICS TAB
# ------------------------------------------------------------------
def render_analytics():
    df = st.session_state["board_df"]
    if df.empty:
        st.warning("No board data yet.")
        return

    st.subheader("📊 Advanced Analytics")

    total = len(df)
    pos_ev = (df["no_vig_ev"] > 0).sum()
    avg_efd = df["efd_score"].mean()
    avg_ev = df["no_vig_ev"].mean() * 100

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total rows", total)
    with c2:
        pct = (pos_ev / total * 100) if total else 0
        st.metric("EV > 0 selections", f"{pos_ev} ({pct:.1f}%)")
    with c3:
        st.metric("Avg EFD score", f"{avg_efd:.1f}")

    st.caption(f"Average no-vig EV across board: {avg_ev:.2f}%")

# ------------------------------------------------------------------
# 🧵 MAIN
# ------------------------------------------------------------------
def main():
    init_state()
    inject_theme()
    st.title("⚡ Edge Force Dominion – Global Odds Engine")

    # light auto-refresh while live
    if st.session_state["boot_done"]:
        st_autorefresh(interval=3000, key="oddsrefresher")

    # API key config
    if not st.session_state["api_key"]:
        st.warning("⚠️ No OpticOdds API key!")
        with st.expander("🔑 How to add OpticOdds API key", expanded=True):
            st.markdown(
                """
                **Local dev**: create `.env` with  
                `OPTICODDS_API_KEY=your_key`  

                **Streamlit Cloud**: Settings → Secrets →  

                ```toml
                [opticodds]
                api_key = "your_key"
                ```
                """
            )
        return

    with st.sidebar:
        st.markdown("### 🔐 OpticOdds API Key")
        key_inp = st.text_input(
            "API key", value=st.session_state["api_key"], type="password"
        )
        if key_inp:
            st.session_state["api_key"] = key_inp

        st.markdown("### 📚 Sportsbooks")
        books_inp = st.text_area(
            "Comma-separated", ", ".join(st.session_state["sportsbooks"])
        )
        st.session_state["sportsbooks"] = [
            b.strip() for b in books_inp.split(",") if b.strip()
        ]

    # initial boot
    if not st.session_state["boot_done"]:
        st.subheader("🧩 Booting Edge Force Dominion Board")
        with st.spinner("Pulling fixtures + odds for all sports…"):
            boot_backend()
        with st.expander("Boot log"):
            for line in st.session_state["boot_log"]:
                st.markdown(f"- {line}")
        st.stop()

    # auto-start SSE once booted
    if st.session_state["boot_done"] and not st.session_state["sse_running"]:
        start_sse_streaming(st.session_state["current_sport"])

    with st.expander("Boot log", expanded=False):
        for line in st.session_state["boot_log"]:
            st.markdown(f"- {line}")

    # Tabs: Dashboard / Arbitrage / Line Moves / Analytics
    tab_dash, tab_arb, tab_line, tab_ana = st.tabs(
        ["🏠 Dashboard", "💰 Arbitrage", "📈 Line Moves", "📊 Analytics"]
    )

    with tab_dash:
        render_board()
    with tab_arb:
        render_arbitrage()
    with tab_line:
        render_line_moves()
    with tab_ana:
        render_analytics()

    st.markdown("---")
    render_console()


if __name__ == "__main__":
    main()
