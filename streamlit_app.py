# streamlit_app.py
# EDGE FORCE DOMINION — OpticOdds v3 Dashboard
# Snapshot + refresh based live board with EV / EFD / arbitrage.

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# -----------------------------------------------------------
# GLOBAL CONFIG
# -----------------------------------------------------------

OPTICODDS_BASE = "https://api.opticodds.com/api/v3"

# app alias -> (sport id, league id) per OpticOdds docs
SPORTS_CONFIG: Dict[str, Dict[str, str]] = {
    "NBA":   {"sport": "basketball", "league": "nba"},
    "NCAAB": {"sport": "basketball", "league": "ncaab"},
    "NFL":   {"sport": "football",   "league": "nfl"},
    "NCAAF": {"sport": "football",   "league": "ncaaf"},
    "MLB":   {"sport": "baseball",   "league": "mlb"},
    "NHL":   {"sport": "ice_hockey", "league": "nhl"},
}

CORE_MARKETS = ["moneyline", "spread", "total_points"]
DEFAULT_BOOKS = ["DraftKings", "FanDuel", "Caesars", "BetMGM", "Pinnacle", "LowVig"]

MAX_FIXTURES_PER_CHUNK = 10          # to keep URLs sane
SNAPSHOT_LOOKAHEAD_DAYS = 2          # how far ahead to pull fixtures

# -----------------------------------------------------------
# THEME
# -----------------------------------------------------------

def inject_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top left,#020617 0,#000 45%,#020617 100%);
            color: #e2f3ff;
        }
        h1, h2, h3 {
            color: #e5f0ff;
            text-shadow: 0 0 12px rgba(96,165,250,.85);
        }
        .fixture-card {
            border-radius: 16px;
            padding: 10px 14px;
            margin-bottom: 10px;
            background: linear-gradient(135deg,
                        rgba(15,118,110,.16),
                        rgba(59,130,246,.12));
            border: 1px solid rgba(148,163,184,.6);
            box-shadow: 0 0 20px rgba(56,189,248,.25);
        }
        .fixture-header {
            font-size: .8rem;
            text-transform: uppercase;
            letter-spacing: .08em;
            color: #a5b4fc;
            margin-bottom: 4px;
        }
        .team-name {
            font-weight: 600;
            font-size: .92rem;
        }
        .odds-row {
            font-size: .84rem;
            color: #e2e8f0;
        }
        .efd-pill {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 999px;
            font-size: .75rem;
            font-weight: 600;
            background: radial-gradient(circle, #22c55e 0, #16a34a 60%, #166534 100%);
            color: #ecfdf5;
            box-shadow: 0 0 10px rgba(34,197,94,.7);
            margin-left: 6px;
        }
        .metric-label {
            font-size: .75rem;
            text-transform: uppercase;
            letter-spacing: .08em;
            color: #94a3b8;
        }
        .metric-value {
            font-size: 1.4rem;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------
# API KEY
# -----------------------------------------------------------

def get_api_key() -> str:
    # Streamlit secrets → env → session_state
    try:
        return st.secrets["opticodds"]["api_key"]
    except Exception:
        return os.getenv("OPTICODDS_API_KEY", st.session_state.get("api_key", ""))

# -----------------------------------------------------------
# MATH HELPERS
# -----------------------------------------------------------

def american_to_decimal(odds: float) -> float:
    try:
        o = float(odds)
    except Exception:
        return 1.0
    if o == 0:
        return 1.0
    if o > 0:
        return 1.0 + (o / 100.0)
    return 1.0 + (100.0 / abs(o))


def implied_prob(odds: float) -> float:
    dec = american_to_decimal(odds)
    if dec <= 1:
        return 0.0
    return 1.0 / dec


def minutes_until(iso: Optional[str]) -> Optional[float]:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return (dt - datetime.now(timezone.utc)).total_seconds() / 60.0
    except Exception:
        return None

# -----------------------------------------------------------
# SESSION STATE INIT
# -----------------------------------------------------------

def init_state() -> None:
    defaults = dict(
        boot_done=False,
        boot_log=[],
        open_board_df=pd.DataFrame(),   # opening snapshot for steam
        board_df=pd.DataFrame(),        # current snapshot
        sportsbooks=DEFAULT_BOOKS.copy(),
        api_key=get_api_key(),
        last_snapshot_ts=None,
        snapshot_run_id=0,              # bumps to force cache miss
    )
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def log_boot(msg: str) -> None:
    st.session_state["boot_log"].append(msg)

# -----------------------------------------------------------
# OPTIC ODDS HTTP
# -----------------------------------------------------------

def optic_get(path: str, params: Dict) -> Dict:
    params = dict(params)
    key = st.session_state["api_key"]
    if not key:
        raise ValueError("No OpticOdds API key configured.")
    params["key"] = key
    url = f"{OPTICODDS_BASE}{path}"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_fixtures_for_sport(sport_id: str, league_id: str) -> pd.DataFrame:
    """
    Hit /fixtures/active for a single sport/league, in the next N days.
    """
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=SNAPSHOT_LOOKAHEAD_DAYS)
    try:
        data = optic_get(
            "/fixtures/active",
            {
                "sport": sport_id,
                "league": league_id,
                "start_date_after": now.isoformat().replace("+00:00", "Z"),
                "start_date_before": end.isoformat().replace("+00:00", "Z"),
            },
        )
    except Exception as e:
        log_boot(f"{sport_id}/{league_id} fixture error: {e}")
        return pd.DataFrame()

    rows: List[Dict] = []
    for fx in data.get("data", []):
        try:
            rows.append(
                {
                    "fixture_id": fx["id"],
                    "sport": fx.get("sport", {}).get("id"),
                    "league": fx.get("league", {}).get("id"),
                    "start_date": fx.get("start_date"),
                    "status": fx.get("status"),
                    "home_name": fx.get("home_team_display"),
                    "away_name": fx.get("away_team_display"),
                    "home_logo": (fx.get("home_competitors") or [{}])[0].get("logo"),
                    "away_logo": (fx.get("away_competitors") or [{}])[0].get("logo"),
                }
            )
        except Exception as e:
            log_boot(f"Bad fixture row: {e}")
    return pd.DataFrame(rows)


def fetch_odds_for_fixtures(
    fixture_ids: List[str],
    markets: List[str],
    books: List[str],
) -> pd.DataFrame:
    """
    Hit /fixtures/odds in chunks for a list of fixture_ids.
    """
    if not fixture_ids:
        return pd.DataFrame()

    rows: List[Dict] = []
    total_chunks = (len(fixture_ids) + MAX_FIXTURES_PER_CHUNK - 1) // MAX_FIXTURES_PER_CHUNK

    progress = st.empty()

    for chunk_index, pos in enumerate(range(0, len(fixture_ids), MAX_FIXTURES_PER_CHUNK), start=1):
        chunk = fixture_ids[pos : pos + MAX_FIXTURES_PER_CHUNK]
        progress.text(f"Pulling odds chunk {chunk_index}/{total_chunks}…")
        try:
            data = optic_get(
                "/fixtures/odds",
                {
                    "fixture_id": chunk,
                    "sportsbook": books,
                    "market": markets,
                    "odds_format": "AMERICAN",
                },
            )
        except Exception as e:
            log_boot(f"Odds chunk error: {e}")
            continue

        for fx in data.get("data", []):
            fid = fx["id"]
            sport = fx.get("sport", {}).get("id")
            league = fx.get("league", {}).get("id")
            start_date = fx.get("start_date")
            hname = fx.get("home_team_display")
            aname = fx.get("away_team_display")

            for od in fx.get("odds", []):
                try:
                    rows.append(
                        {
                            "fixture_id": fid,
                            "sport": sport,
                            "league": league,
                            "start_date": start_date,
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
                except Exception as e:
                    log_boot(f"Bad odds row: {e}")

        time.sleep(0.3)

    progress.empty()
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["price_decimal"] = df["price"].apply(american_to_decimal)
    df["implied_prob"] = df["price"].apply(implied_prob)
    return df

# -----------------------------------------------------------
# EV + EFD
# -----------------------------------------------------------

def compute_no_vig_ev(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each fixture/market/grouping, find best odds per selection and compute
    fair probabilities + no-vig EV.
    """
    if df.empty:
        return df

    df = df.copy()
    df["fair_prob"] = np.nan
    df["no_vig_ev"] = np.nan

    group_cols = ["fixture_id", "market_id", "grouping_key"]

    updates: List[pd.DataFrame] = []

    for _, g in df.groupby(group_cols, dropna=False):
        if g.empty:
            continue

        # best odds per selection
        best = g.sort_values("price_decimal", ascending=False).groupby("selection").head(1)
        if len(best) < 2:
            continue

        inv_sum = (1.0 / best["price_decimal"]).sum()
        if inv_sum <= 0:
            continue

        best = best.copy()
        best["fair_prob"] = (1.0 / best["price_decimal"]) / inv_sum
        best["no_vig_ev"] = best["price_decimal"] * best["fair_prob"] - 1.0
        updates.append(best[["fixture_id", "market_id", "grouping_key", "selection", "sportsbook", "fair_prob", "no_vig_ev"]])

    if updates:
        upd = pd.concat(updates, ignore_index=True)
        df = df.merge(
            upd,
            on=["fixture_id", "market_id", "grouping_key", "selection", "sportsbook"],
            how="left",
            suffixes=("", "_upd"),
        )
        for col in ["fair_prob", "no_vig_ev"]:
            df[col] = df[f"{col}_upd"].combine_first(df[col])
            df.drop(columns=[f"{col}_upd"], inplace=True)

    return df


def efd_score_row(row: pd.Series) -> float:
    """
    Edge Force Dominion scoring:
    - no_vig_ev              → math_comp
    - |steam| (implied diff) → steam_comp
    - minutes to start       → time_comp
    - moneyline > spread > total weighting
    Returns 0–100-ish score.
    """
    ev = float(row.get("no_vig_ev") or 0.0)
    open_i = float(row.get("open_implied") or 0.0)
    curr_i = float(row.get("implied_prob") or 0.0)
    steam = abs(curr_i - open_i)

    mins = row.get("minutes_to_start")
    try:
        mins = float(mins) if mins is not None else None
    except Exception:
        mins = None

    market_id = str(row.get("market_id") or "").lower()

    # compress EV into [0,1] window around [-0.10, +0.12]
    ev_clamped = min(max(ev, -0.10), 0.12)
    math_comp = (ev_clamped + 0.10) / 0.22  # 0–1

    steam_clamped = min(steam, 0.10)
    steam_comp = steam_clamped / 0.10  # 0–1

    if mins is None:
        time_comp = 0.5
    elif mins <= 0:
        time_comp = 1.0
    elif mins <= 360:
        time_comp = 1.0 - (mins / 360.0) * 0.5
    else:
        time_comp = 0.5

    if "moneyline" in market_id:
        market_comp = 1.0
    elif "spread" in market_id:
        market_comp = 0.9
    else:
        market_comp = 0.85

    score = (0.45 * math_comp + 0.35 * steam_comp + 0.20 * time_comp) * 100.0 * market_comp
    return round(float(score), 1)


def enrich_board_with_ev_efd(board_df: pd.DataFrame, open_df: pd.DataFrame) -> pd.DataFrame:
    if board_df.empty:
        return board_df

    df = board_df.copy()

    # link to opening snapshot for steam vs open
    merge_cols = ["fixture_id", "sportsbook", "market_id", "selection"]
    open_cols = merge_cols + ["implied_prob"]
    open_df_small = open_df[open_cols].rename(columns={"implied_prob": "open_implied"}) if not open_df.empty else pd.DataFrame()

    df = df.merge(open_df_small, on=merge_cols, how="left")

    # if this is the first snapshot, open_implied == current
    df["open_implied"] = df["open_implied"].fillna(df["implied_prob"])

    # minutes to start
    df["minutes_to_start"] = df["start_date"].apply(minutes_until)

    # compute fair probs + EV
    df = compute_no_vig_ev(df)

    # steam vs open
    df["steam"] = df["implied_prob"] - df["open_implied"]

    # EFD score
    df["efd_score"] = df.apply(efd_score_row, axis=1)

    return df

# -----------------------------------------------------------
# SNAPSHOT BUILDER (CACHED)
# -----------------------------------------------------------

@st.cache_data(ttl=30)
def build_snapshot(
    api_key: str,
    sports: Tuple[str, ...],
    books: Tuple[str, ...],
    markets: Tuple[str, ...],
    run_id: int,
) -> pd.DataFrame:
    """
    Pull fixtures + odds for selected sports.
    run_id is only there to let us force-refresh cache.
    """
    _ = api_key  # just to tie cache to key
    all_rows: List[pd.DataFrame] = []

    for alias in sports:
        cfg = SPORTS_CONFIG.get(alias)
        if not cfg:
            continue

        sport_id = cfg["sport"]
        league_id = cfg["league"]

        log_boot(f"{alias}: fixtures…")
        fixtures_df = fetch_fixtures_for_sport(sport_id, league_id)
        if fixtures_df.empty:
            log_boot(f"{alias}: no fixtures.")
            continue

        fixture_ids = fixtures_df["fixture_id"].tolist()
        log_boot(f"{alias}: {len(fixture_ids)} fixtures; odds…")

        odds_df = fetch_odds_for_fixtures(fixture_ids, list(markets), list(books))
        if odds_df.empty:
            log_boot(f"{alias}: no odds.")
            continue

        # add alias + logos by joining fixtures
        odds_df = odds_df.merge(
            fixtures_df[["fixture_id", "home_logo", "away_logo"]],
            on="fixture_id",
            how="left",
        )
        odds_df["alias"] = alias
        all_rows.append(odds_df)

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)

# -----------------------------------------------------------
# ARBITRAGE DETECTION
# -----------------------------------------------------------

def detect_arbitrage(board_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple 2-way market arbitrage: use best price per selection across books.
    """
    if board_df.empty:
        return pd.DataFrame()

    df = board_df.copy()
    df = df.dropna(subset=["price_decimal"])

    records: List[Dict] = []

    group_cols = ["fixture_id", "market_id", "grouping_key"]

    for (fixture_id, market_id, gk), g in df.groupby(group_cols, dropna=False):
        # best odds per selection (across all books)
        best_rows = g.sort_values("price_decimal", ascending=False).groupby("selection").head(1)
        if len(best_rows) < 2:
            continue  # need at least 2 selections for an arb

        inv_sum = (1.0 / best_rows["price_decimal"]).sum()
        if inv_sum >= 1.0:
            continue  # no arbitrage

        profit_pct = (1.0 / inv_sum - 1.0) * 100.0

        # Build leg description strings
        legs = []
        for _, r in best_rows.iterrows():
            legs.append(f"{r['selection']} @ {r['sportsbook']} ({r['price']})")

        first_row = best_rows.iloc[0]
        records.append(
            {
                "Sport": first_row.get("alias"),
                "Start": first_row.get("start_date"),
                "Home": first_row.get("home_name"),
                "Away": first_row.get("away_name"),
                "Market": first_row.get("market_label"),
                "Arb Profit %": round(profit_pct, 2),
                "Legs": " | ".join(legs),
            }
        )

    if not records:
        return pd.DataFrame()

    out = pd.DataFrame(records).sort_values("Arb Profit %", ascending=False)
    return out

# -----------------------------------------------------------
# BOARD RENDERING
# -----------------------------------------------------------

def render_summary(board_df: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-label">Active Sports</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{board_df["alias"].nunique() if not board_df.empty else 0}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-label">Tracked Sportsbooks</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value">{board_df["sportsbook"].nunique() if not board_df.empty else 0}</div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown('<div class="metric-label">Fixtures</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value">{board_df["fixture_id"].nunique() if not board_df.empty else 0}</div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown('<div class="metric-label">Rows Ingested</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{len(board_df)}</div>', unsafe_allow_html=True)


def render_matchup_cards(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No odds after filters.")
        return

    # group by fixture
    fixture_meta = (
        df.groupby("fixture_id")
        .agg(
            start_date=("start_date", "first"),
            home_name=("home_name", "first"),
            away_name=("away_name", "first"),
            home_logo=("home_logo", "first"),
            away_logo=("away_logo", "first"),
            alias=("alias", "first"),
        )
        .reset_index()
    )

    # we sort fixtures by top EFD in that fixture
    fixture_scores = (
        df.groupby("fixture_id")["efd_score"]
        .max()
        .reset_index()
        .rename(columns={"efd_score": "top_efd"})
    )
    fixture_meta = fixture_meta.merge(fixture_scores, on="fixture_id", how="left")
    fixture_meta = fixture_meta.sort_values("top_efd", ascending=False).head(15)

    for _, meta in fixture_meta.iterrows():
        sub = df[df["fixture_id"] == meta["fixture_id"]]
        ml = sub[sub["market_id"].str.contains("moneyline", case=False, na=False)]
        sp = sub[sub["market_id"].str.contains("spread", case=False, na=False)]
        tot = sub[sub["market_id"].str.contains("total", case=False, na=False)]

        def best_row(mdf: pd.DataFrame) -> Optional[pd.Series]:
            if mdf.empty:
                return None
            return mdf.sort_values("efd_score", ascending=False).iloc[0]

        best_ml = best_row(ml)
        best_sp = best_row(sp)
        best_tot = best_row(tot)

        col_left, col_right = st.columns([2, 3])

        with col_left:
            st.markdown('<div class="fixture-card">', unsafe_allow_html=True)
            start = meta["start_date"] or ""
            alias = meta["alias"] or ""
            header = f"{alias} • {start}"
            st.markdown(f'<div class="fixture-header">{header}</div>', unsafe_allow_html=True)

            lc1, lc2 = st.columns(2)
            with lc1:
                if meta["home_logo"]:
                    st.image(meta["home_logo"], width=48)
                st.markdown(f'<span class="team-name">{meta["home_name"]}</span>', unsafe_allow_html=True)
            with lc2:
                if meta["away_logo"]:
                    st.image(meta["away_logo"], width=48)
                st.markdown(f'<span class="team-name">{meta["away_name"]}</span>', unsafe_allow_html=True)

            if not np.isnan(meta.get("top_efd", np.nan)):
                st.markdown(
                    f"<div style='margin-top:6px'>Top EFD<span class=\"efd-pill\">{meta['top_efd']:.1f}</span></div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        with col_right:
            st.write("Top bets by EFD:")
            lines = []
            if best_ml is not None:
                lines.append(
                    f"**Moneyline:** {best_ml['sportsbook']} – {best_ml['name']} "
                    f"({best_ml['price']}) • EFD {best_ml['efd_score']:.1f}"
                )
            if best_sp is not None:
                pt = best_sp['points']
                pt_str = f"{pt:+}" if pt is not None else ""
                lines.append(
                    f"**Spread:** {best_sp['sportsbook']} – {best_sp['name']} {pt_str} "
                    f"({best_sp['price']}) • EFD {best_sp['efd_score']:.1f}"
                )
            if best_tot is not None:
                pt = best_tot['points']
                lines.append(
                    f"**Total:** {best_tot['sportsbook']} – {best_tot['name']} {pt} "
                    f"({best_tot['price']}) • EFD {best_tot['efd_score']:.1f}"
                )
            if lines:
                st.markdown("<div class='odds-row'>" + "<br>".join(lines) + "</div>", unsafe_allow_html=True)
            else:
                st.caption("No core markets found for this fixture.")

        st.markdown("---")


def render_board() -> None:
    board = st.session_state["board_df"]
    if board.empty:
        st.error("No odds data loaded. Check boot log & API key.")
        return

    # filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sport_opt = ["All"] + list(SPORTS_CONFIG.keys())
        sport_f = st.selectbox("Sport", sport_opt, index=0)
    with col2:
        market_opt = ["All", "moneyline", "spread", "total"]
        market_f = st.selectbox("Market", market_opt, index=0)
    with col3:
        book_opt = ["All"] + sorted(board["sportsbook"].dropna().unique().tolist())
        book_f = st.selectbox("Sportsbook", book_opt, index=0)
    with col4:
        sort_opt = ["EFD score", "EV", "Steam"]
        sort_by = st.selectbox("Sort by", sort_opt, index=0)

    df = board.copy()
    if sport_f != "All":
        df = df[df["alias"] == sport_f]
    if market_f != "All":
        df = df[df["market_id"].str.contains(market_f, case=False, na=False)]
    if book_f != "All":
        df = df[df["sportsbook"] == book_f]

    if df.empty:
        st.warning("No rows match current filters.")
        return

    if sort_by == "EV":
        df = df.sort_values("no_vig_ev", ascending=False)
    elif sort_by == "Steam":
        df = df.sort_values("steam", ascending=False)
    else:
        df = df.sort_values("efd_score", ascending=False)

    st.subheader("Matchups ordered by EFD")
    render_matchup_cards(df)

    st.subheader("Full Board (Top 250 rows)")
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
    tbl = df[show_cols].copy()
    tbl.rename(
        columns={
            "alias": "Sport",
            "start_date": "Start",
            "home_name": "Home",
            "away_name": "Away",
            "sportsbook": "Book",
            "market_label": "Market",
            "name": "Bet",
            "price": "Odds",
            "no_vig_ev": "EV",
            "efd_score": "EFD",
            "steam": "Steam",
        },
        inplace=True,
    )
    tbl["EV %"] = tbl["EV"].apply(lambda v: f"{(v or 0)*100:.1f}%")
    tbl["Steam %"] = tbl["Steam"].apply(lambda v: f"{(v or 0)*100:.1f}%")
    st.dataframe(
        tbl[["Sport", "Start", "Home", "Away", "Book", "Market", "Bet", "Odds", "EV %", "Steam %", "EFD"]].head(250),
        use_container_width=True,
    )

# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

def main() -> None:
    init_state()
    inject_theme()

    st.title("🏆 Edge Force Dominion — OpticOdds Live Dashboard")
    st.caption("Snapshot + refresh live odds with EV, EFD scoring, line moves, and arbitrage radar.")

    # auto-refresh every few seconds once booted
    if st.session_state["boot_done"]:
        st_autorefresh(interval=10000, key="efd_auto_refresh")

    # sidebar controls
    with st.sidebar:
        st.markdown("### OpticOdds Configuration")
        key_input = st.text_input(
            "API Key",
            value=st.session_state["api_key"],
            type="password",
            help="Stored in session only (or from Streamlit secrets / env).",
        )
        if key_input:
            st.session_state["api_key"] = key_input

        st.markdown("### Sportsbooks")
        books_str = ", ".join(st.session_state["sportsbooks"])
        books_input = st.text_area(
            "Comma-separated list",
            value=books_str,
            height=80,
        )
        chosen_books = [b.strip() for b in books_input.split(",") if b.strip()]
        if chosen_books:
            st.session_state["sportsbooks"] = chosen_books

        st.markdown("### Sports")
        sports_multi = st.multiselect(
            "Tracked sports",
            options=list(SPORTS_CONFIG.keys()),
            default=["NBA", "NFL", "NCAAF", "NCAAB"],
        )

        st.markdown("### Markets")
        markets_multi = st.multiselect(
            "Markets",
            options=CORE_MARKETS,
            default=CORE_MARKETS,
        )

        st.markdown("---")
        if st.button("Run Snapshot / Refresh Now"):
            st.session_state["snapshot_run_id"] += 1
            st.session_state["boot_done"] = False  # force rebuild

    if not st.session_state["api_key"]:
        st.warning("Add your OpticOdds API key in the sidebar to begin.")
        return

    # first-time boot or forced refresh
    if not st.session_state["boot_done"]:
        st.subheader("Booting Edge Force Dominion Board")
        with st.spinner("Pulling fixtures & odds from OpticOdds…"):
            board_snapshot = build_snapshot(
                api_key=st.session_state["api_key"],
                sports=tuple(sports_multi),
                books=tuple(st.session_state["sportsbooks"]),
                markets=tuple(markets_multi),
                run_id=st.session_state["snapshot_run_id"],
            )

        if board_snapshot.empty:
            st.error("No odds came back from OpticOdds. Check key, sport/league coverage, and account access.")
            with st.expander("Boot log", expanded=True):
                for line in st.session_state["boot_log"]:
                    st.markdown(f"- {line}")
            return

        # If we don't have an "open" board yet, set it to this snapshot
        if st.session_state["open_board_df"].empty:
            st.session_state["open_board_df"] = board_snapshot.copy()

        # enrich with EV / EFD using existing open snapshot
        enriched = enrich_board_with_ev_efd(board_snapshot, st.session_state["open_board_df"])

        st.session_state["board_df"] = enriched
        st.session_state["boot_done"] = True
        st.session_state["last_snapshot_ts"] = datetime.utcnow()

    # dashboard tabs
    board_df = st.session_state["board_df"]

    tabs = st.tabs(["Dashboard", "Arbitrage", "Line Moves", "Analytics", "Boot Log"])

    with tabs[0]:
        st.subheader("Live Dashboard")
        render_summary(board_df)
        if st.session_state["last_snapshot_ts"]:
            ts = st.session_state["last_snapshot_ts"].strftime("%H:%M:%S UTC")
            st.caption(f"Last snapshot: {ts}")
        st.markdown("---")
        render_board()

    with tabs[1]:
        st.subheader("Arbitrage Opportunities")
        arb_df = detect_arbitrage(board_df)
        if arb_df.empty:
            st.info("No clean 2-way arbitrage spots detected in the current snapshot.")
        else:
            st.dataframe(arb_df, use_container_width=True)

    with tabs[2]:
        st.subheader("Line Movement (Steam vs Session Open)")
        if board_df.empty or st.session_state["open_board_df"].empty:
            st.info("Need at least one snapshot to track line movement.")
        else:
            lm = board_df.copy()
            lm["Steam %"] = lm["steam"].apply(lambda v: f"{(v or 0)*100:.2f}%")
            lm_small = lm[
                [
                    "alias",
                    "start_date",
                    "home_name",
                    "away_name",
                    "sportsbook",
                    "market_label",
                    "name",
                    "price",
                    "Steam %",
                    "efd_score",
                ]
            ].rename(
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
            lm_small = lm_small.sort_values("EFD", ascending=False).head(200)
            st.dataframe(lm_small, use_container_width=True)

    with tabs[3]:
        st.subheader("Analytics")
        if board_df.empty:
            st.info("No data yet.")
        else:
            by_sport = (
                board_df.groupby("alias")
                .agg(
                    fixtures=("fixture_id", "nunique"),
                    books=("sportsbook", "nunique"),
                    avg_efd=("efd_score", "mean"),
                    max_efd=("efd_score", "max"),
                )
                .reset_index()
                .rename(columns={"alias": "Sport"})
            )
            st.markdown("#### By Sport")
            st.dataframe(by_sport, use_container_width=True)

            st.markdown("#### EFD Score Distribution (sample)")
            st.bar_chart(board_df["efd_score"].fillna(0).clip(0, 120))

    with tabs[4]:
        st.subheader("Boot Log")
        if not st.session_state["boot_log"]:
            st.caption("No log entries yet.")
        else:
            for line in st.session_state["boot_log"]:
                st.markdown(f"- {line}")


if __name__ == "__main__":
    main()
