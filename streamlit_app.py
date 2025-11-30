# streamlit_app.py
# Edge-Force-Dominion – OpticOdds v3 Global Odds Engine
# ------------------------------------------------------------------
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
MAX_FIXTURES_CALL = 5
SPORT_DELAY       = 1.0
CHUNK_DELAY       = 0.5

# ------------------------------------------------------------------
# 🔐 API-KEY HANDLER
# ------------------------------------------------------------------
def get_api_key() -> str:
    try:                        # Streamlit-Cloud secret
        return st.secrets["opticodds"]["api_key"]
    except Exception:
        return os.getenv("OPTICODDS_API_KEY", st.session_state.get("api_key", ""))

# ------------------------------------------------------------------
# 🎨 THEME
# ------------------------------------------------------------------
def inject_theme():
    st.markdown("""
    <style>
    .stApp{background:radial-gradient(circle at top left,#020617 0,#000 40%,#020617 100%);color:#e2f3ff}
    h1,h2,h3{color:#e5f0ff;text-shadow:0 0 12px rgba(96,165,250,.85)}
    .fixture-card{border-radius:16px;padding:12px 14px;background:linear-gradient(135deg,rgba(15,118,110,.15),rgba(59,130,246,.1));border:1px solid rgba(148,163,184,.5);box-shadow:0 0 18px rgba(56,189,248,.25)}
    .fixture-header{font-size:.9rem;text-transform:uppercase;letter-spacing:.08em;color:#a5b4fc;margin-bottom:4px}
    .team-name{font-weight:600}
    .odds-row{font-size:.85rem;color:#e2e8f0}
    .cmd-user{color:#93c5fd}.cmd-system{color:#a5b4fc}
    </style>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# 🧮 MATH HELPERS
# ------------------------------------------------------------------
def american_to_decimal(odds: float) -> float:
    try:
        odds = float(odds)
        return 1 + odds/100 if odds > 0 else 1 + 100/abs(odds)
    except Exception:
        return 1.0

def implied_prob(odds: float) -> float:
    dec = american_to_decimal(odds)
    return 1/dec if dec > 1 else 0.0

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
        boot_done     = False,
        boot_log      = [],
        board_df      = pd.DataFrame(),
        messages      = [],
        sportsbooks   = DEFAULT_BOOKS.copy(),
        api_key       = get_api_key(),
        sse_queue     = queue.Queue(),
        sse_running   = False,
        sse_thread    = None,
        last_update   = None,
        current_sport = "nba",
    )
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

def log_boot(msg: str):
    st.session_state["boot_log"].append(msg)

# ------------------------------------------------------------------
# 🌐 OPTICODDS HELPERS
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
    now, before = datetime.now(timezone.utc), datetime.now(timezone.utc) + timedelta(days=days)
    try:
        resp = optic_get("/fixtures/active", {
            "sport": sport_id,
            "league": league_id,
            "start_date_after":  now.isoformat().replace("+00:00", "Z"),
            "start_date_before": before.isoformat().replace("+00:00", "Z"),
        })
    except ValueError as e:
        st.error(str(e))
        return pd.DataFrame()

    rows = []
    for fx in resp.get("data", []):
        try:
            rows.append({
                "fixture_id": fx["id"],
                "sport":      fx["sport"]["id"],
                "league":     fx["league"]["id"],
                "start_date": fx.get("start_date"),
                "home_name":  fx.get("home_team_display"),
                "away_name":  fx.get("away_team_display"),
                "home_logo":  (fx.get("home_competitors") or [{}])[0].get("logo"),
                "away_logo":  (fx.get("away_competitors") or [{}])[0].get("logo"),
                "status":     fx.get("status"),
            })
        except Exception as e:
            log_boot(f"Bad fixture {fx.get('id')}: {e}")
    return pd.DataFrame(rows)

def fetch_odds_chunks(fixture_ids: List[str], markets: List[str], books: List[str], status) -> pd.DataFrame:
    if not fixture_ids:
        return pd.DataFrame()
    rows, total = [], (len(fixture_ids) + MAX_FIXTURES_CALL - 1) // MAX_FIXTURES_CALL
    for idx, i in enumerate(range(0, len(fixture_ids), MAX_FIXTURES_CALL), 1):
        chunk = fixture_ids[i:i+MAX_FIXTURES_CALL]
        status.text(f"📡 Odds chunk {idx}/{total}")
        try:
            resp = optic_get("/fixtures/odds", {
                "fixture_id": chunk,
                "sportsbook": books,
                "market":     markets,
                "odds_format":"AMERICAN",
            })
            for fx in resp.get("data", []):
                fid   = fx["id"]
                sd    = fx.get("start_date")
                hname = fx.get("home_team_display")
                aname = fx.get("away_team_display")
                sport = fx.get("sport", {}).get("id")
                league= fx.get("league", {}).get("id")
                for od in fx.get("odds", []):
                    rows.append({
                        "fixture_id":  fid,
                        "sport":       sport,
                        "league":      league,
                        "start_date":  sd,
                        "home_name":   hname,
                        "away_name":   aname,
                        "sportsbook":  od.get("sportsbook"),
                        "market_id":   od.get("market_id") or od.get("market"),
                        "market_label":od.get("market"),
                        "selection":   od.get("selection"),
                        "name":        od.get("name"),
                        "price":       od.get("price"),
                        "grouping_key":od.get("grouping_key"),
                        "points":      od.get("points"),
                    })
            if i+MAX_FIXTURES_CALL < len(fixture_ids):
                time.sleep(CHUNK_DELAY)
        except Exception as e:
            log_boot(f"Chunk {idx} error: {e}")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["price_decimal"] = df["price"].apply(american_to_decimal)
    df["implied_prob"]  = df["price"].apply(implied_prob)
    df["open_price"]    = df["price"]
    df["open_implied"]  = df["implied_prob"]
    return df

# ------------------------------------------------------------------
# 🧮 EV + EFD
# ------------------------------------------------------------------
def efd_row(row) -> float:
    ev     = float(row.get("no_vig_ev") or 0)
    open_i = float(row.get("open_implied") or 0)
    curr_i = float(row.get("curr_implied") or 0)
    steam  = abs(curr_i - open_i)
    mins   = row.get("minutes_to_start")
    try:
        mins = float(mins) if mins is not None else None
    except Exception:
        mins = None
    market = (row.get("market_id") or "").lower()

    ev_c      = min(max(ev, -.1), .12)
    math_comp = (ev_c + .1) / .22
    steam_c   = min(steam, .1) / .1
    if mins is None:
        time_c = .5
    elif mins <= 0:
        time_c = 1
    elif mins <= 360:
        time_c = 1 - (mins/360)*.5
    else:
        time_c = .5
    market_c = 1 if "moneyline" in market else .9 if "spread" in market else .85
    score    = ( .45*math_comp + .35*steam_c + .20*time_c ) * 100 * market_c
    return float(round(score, 1))

def recompute_ev_efd(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df["curr_implied"] = df["price"].apply(implied_prob)
    df["steam"]        = df["curr_implied"] - df["open_implied"]
    df["fair_prob"] = df["no_vig_ev"] = np.nan

    grp = ["fixture_id", "market_id", "grouping_key"]
    updates = []
    for _, g in df.groupby(grp, as_index=False):
        best = (g.sort_values("price_decimal", ascending=False)
                  .groupby("selection", as_index=False).first())
        if len(best) < 2:
            continue
        inv = (1/best["price_decimal"]).sum()
        if inv <= 0:
            continue
        best["fair_prob"] = (1/best["price_decimal"])/inv
        best["no_vig_ev"] = best["price_decimal"]*best["fair_prob"] - 1
        updates.append(best)
    if updates:
        upd = pd.concat(updates, ignore_index=True)
        df  = df.merge(upd[grp+["selection","sportsbook","fair_prob","no_vig_ev"]],
                       on=grp+["selection","sportsbook"], how="left", suffixes=("","_u"))
        df["fair_prob"] = df["fair_prob_u"].combine_first(df["fair_prob"])
        df["no_vig_ev"] = df["no_vig_ev_u"].combine_first(df["no_vig_ev"])
        df = df.drop(columns=[c for c in df.columns if c.endswith("_u")])

    df["minutes_to_start"] = df["start_date"].apply(minutes_until)
    df["efd_score"]        = df.apply(efd_row, axis=1)
    return df

# ------------------------------------------------------------------
# 🧵 BOOT BACKEND  (MISSING PIECE)
# ------------------------------------------------------------------
def boot_backend():
    log_boot("Booting Edge Force Dominion…")
    all_dfs = []
    total   = len(SPORTS_CONFIG)
    status  = st.empty()
    for idx, (alias, cfg) in enumerate(SPORTS_CONFIG.items(), 1):
        sport, league = cfg["sport"], cfg["league"]
        log_boot(f"[{idx}/{total}] Fixtures → {alias.upper()}")
        fixtures = fetch_fixtures(sport, league, days=2)
        if fixtures.empty:
            log_boot(f"No fixtures for {alias}")
            continue
        fids = fixtures["fixture_id"].tolist()
        log_boot(f"Found {len(fids)} fixtures for {alias}")
        odds = fetch_odds_chunks(fids, CORE_MARKETS, st.session_state["sportsbooks"], status)
        if odds.empty:
            log_boot(f"No odds for {alias}")
            continue
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
    st.session_state["boot_done"] = True
    log_boot("Boot complete ✅")

# ------------------------------------------------------------------
# 🌐 SSE (STUB – OPTICODDS SSE SHAPE TBD)
# ------------------------------------------------------------------
def start_sse_streaming(sport_alias: str):
    # Placeholder – wire real SSE when OpticOdds docs finalised
    st.session_state["sse_running"] = True
    st.session_state["current_sport"] = sport_alias
    st.session_state["last_update"]   = datetime.utcnow()

def stop_sse_streaming():
    st.session_state["sse_running"] = False

def process_pending_updates():
    # Merge queued SSE deltas into board_df
    pass

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
        add_msg("system", "Commands:\n- help\n- show all\n- show <sport>\n- show market <ml|spread|total>")
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
    add_msg("system", f"Unknown cmd – try 'help'")

def render_console():
    st.subheader("⌨️ Command Console")
    for m in st.session_state["messages"]:
        cls = "cmd-user" if m["role"] == "user" else "cmd-system"
        who = "You" if m["role"] == "user" else "System"
        st.markdown(f"<div class='{cls}'><b>{who}:</b> {m['text']}</div>", unsafe_allow_html=True)
    cmd = st.chat_input("Type command…")
    if cmd:
        handle_cmd(cmd)

# ------------------------------------------------------------------
# 🎨 BOARD RENDER
# ------------------------------------------------------------------
def render_board():
    board = st.session_state["board_df"]
    if board.empty:
        st.error("Board empty – check boot log")
        return

    # --- status bar ---
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.session_state["sse_running"]:
            st.success("🟢 LIVE")
            if st.session_state["last_update"]:
                st.caption(f"Last: {st.session_state['last_update'].strftime('%H:%M:%S UTC')}")
        else:
            st.error("🔴 Offline")
    with col2:
        if st.session_state["sse_running"]:
            if st.button("⏸️ Pause"):
                stop_sse_streaming()
                st.rerun()
        else:
            if st.button("▶️ Start Live"):
                start_sse_streaming(st.session_state["current_sport"])
                st.rerun()
    with col3:
        sport_sel = st.selectbox("Live Sport", list(SPORTS_CONFIG.keys()),
                                 index=list(SPORTS_CONFIG.keys()).index(st.session_state["current_sport"]),
                                 key="live_sport")
        if sport_sel != st.session_state["current_sport"]:
            stop_sse_streaming()
            start_sse_streaming(sport_sel)
            st.rerun()

    st.markdown("---")

    # --- filters ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sport_f = st.selectbox("Sport", ["all"] + list(SPORTS_CONFIG.keys()))
    with c2:
        market_f = st.selectbox("Market", ["all", "moneyline", "spread", "total"])
    with c3:
        book_f = st.selectbox("Book", ["all"] + sorted(board["sportsbook"].dropna().unique()))
    with c4:
        sort = st.selectbox("Sort", ["EFD score", "EV", "Steam"])

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

    # --- cards ---
    st.subheader("🎮 Matchup Cards")
    for _, row in (df.groupby("fixture_id").agg({
        "start_date": "first", "home_name": "first", "away_name": "first",
        "home_logo": "first", "away_logo": "first"
    }).reset_index().head(8)).iterrows():
        a, b = st.columns([2, 3])
        with a:
            st.markdown('<div class="fixture-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="fixture-header">{row["start_date"]}</div>', unsafe_allow_html=True)
            lc1, lc2 = st.columns(2)
            with lc1:
                if row["home_logo"]: st.image(row["home_logo"], width=48)
                st.markdown(f'<span class="team-name">{row["home_name"]}</span>', unsafe_allow_html=True)
            with lc2:
                if row["away_logo"]: st.image(row["away_logo"], width=48)
                st.markdown(f'<span class="team-name">{row["away_name"]}</span>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with b:
            st.write("Top prices (ML / Spread / Total):")
            sub = df[df["fixture_id"] == row["fixture_id"]]
            ml  = sub[sub["market_id"].str.contains("moneyline", case=False, na=False)]
            sp  = sub[sub["market_id"].str.contains("spread",   case=False, na=False)]
            tot = sub[sub["market_id"].str.contains("total",    case=False, na=False)]

            def summarise(mdf, label):
                if mdf.empty: return f"**{label}:** —"
                top = mdf.sort_values("price_decimal", ascending=False).iloc[0]
                return f"**{label}:** {top['sportsbook']} – {top['name']} ({top['price']}, EFD {top['efd_score']:.1f})"
            st.markdown("<div class='odds-row'>" +
                        summarise(ml, "Moneyline") + "<br>" +
                        summarise(sp, "Spread") + "<br>" +
                        summarise(tot, "Total") + "</div>", unsafe_allow_html=True)
        st.markdown("---")

    # --- table ---
    st.subheader("📊 Full Board (top 250)")
    show_cols = ["alias", "start_date", "home_name", "away_name", "sportsbook",
                 "market_label", "name", "price", "no_vig_ev", "steam", "efd_score"]
    tmp = df[show_cols].rename(columns={
        "alias": "Sport", "start_date": "Start", "home_name": "Home", "away_name": "Away",
        "sportsbook": "Book", "market_label": "Market", "name": "Bet", "price": "Odds", "efd_score": "EFD"
    })
    tmp["EV%"]    = tmp["no_vig_ev"].apply(lambda v: f"{(v or 0)*100:.1f}%")
    tmp["Steam%"] = tmp["steam"].apply(lambda v: f"{(v or 0)*100:.1f}%")
    st.dataframe(tmp[["Sport", "Start", "Home", "Away", "Book", "Market", "Bet", "Odds", "EV%", "Steam%", "EFD"]].head(250),
                 use_container_width=True)

# ------------------------------------------------------------------
# 🧵 MAIN
# ------------------------------------------------------------------
def main():
    init_state()
    inject_theme()
    st.title("⚡ Edge Force Dominion – Global Odds Engine")

    if st.session_state["boot_done"]:
        st_autorefresh(interval=3000, key="oddsrefresher")

    if not st.session_state["api_key"]:
        st.warning("⚠️ No API key!")
        with st.expander("🔑 How to add OpticOdds API key", expanded=True):
            st.markdown("""
            **Local dev**: create `.env` with  
            `OPTICODDS_API_KEY=your_key`  
            **Streamlit-Cloud**: Settings → Secrets →  
            `[opticodds]\napi_key = your_key`
            """)
        return

    with st.sidebar:
        st.markdown("### 🔐 OpticOdds API Key")
        key_inp = st.text_input("API key", value=st.session_state["api_key"], type="password")
        if key_inp:
            st.session_state["api_key"] = key_inp
        st.markdown("### 📚 Sportsbooks")
        books_inp = st.text_area("Comma-separated", ", ".join(st.session_state["sportsbooks"]))
        st.session_state["sportsbooks"] = [b.strip() for b in books_inp.split(",") if b.strip()]

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
        st.rerun()

    with st.expander("Boot log", expanded=False):
        for line in st.session_state["boot_log"]:
            st.markdown(f"- {line}")

    render_board()
    st.markdown("---")
    render_console()

if __name__ == "__main__":
    main()
