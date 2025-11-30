# streamlit_app.py
"""
Edge Force Dominion – OpticOdds Live Engine (Snapshot + Line-Move v2)

- Pulls fixtures + odds from OpticOdds for selected sports.
- Computes no-vig EV and Edge Force Dominion (EFD) scores.
- Tracks line movement vs a session-long baseline (reverse line moves).
- Detects 2-way arbitrage spots across books.
"""

import os
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:  # fallback if package not installed
    def st_autorefresh(*args, **kwargs):
        return None


# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------

OPTICODDS_BASE = "https://api.opticodds.com/api/v3"

SPORTS_CONFIG: Dict[str, Dict[str, str]] = {
    "NBA":   {"sport": "basketball", "league": "nba"},
    "NCAAB": {"sport": "basketball", "league": "ncaab"},
    "NFL":   {"sport": "football",   "league": "nfl"},
    "NCAAF": {"sport": "football",   "league": "ncaaf"},
    "MLB":   {"sport": "baseball",   "league": "mlb"},
    "NHL":   {"sport": "ice_hockey", "league": "nhl"},
}

CORE_MARKETS = ["moneyline", "spread", "total_points"]
DEFAULT_BOOKS = ["DraftKings", "FanDuel", "Caesars", "BetMGM"]

FIXTURE_LOOKAHEAD_DAYS = 2


# -----------------------------------------------------------
# BASIC HELPERS
# -----------------------------------------------------------

def get_api_key() -> str:
    # secrets -> env -> session
    try:
        return st.secrets["opticodds"]["api_key"]
    except Exception:
        pass
    env_key = os.getenv("OPTICODDS_API_KEY", "")
    if env_key:
        return env_key
    return st.session_state.get("api_key", "")


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


def minutes_until(iso_ts: Optional[str]) -> Optional[float]:
    if not iso_ts:
        return None
    try:
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        return (dt - datetime.now(timezone.utc)).total_seconds() / 60.0
    except Exception:
        return None


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
# SESSION STATE
# -----------------------------------------------------------

def init_state() -> None:
    defaults = dict(
        api_key=get_api_key(),
        baseline_df=pd.DataFrame(),
        board_df=pd.DataFrame(),
        arb_df=pd.DataFrame(),
        auto_refresh=False,
    )
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


# -----------------------------------------------------------
# OPTIC ODDS HTTP
# -----------------------------------------------------------

def optic_get(path: str, params: Dict) -> Dict:
    key = st.session_state.get("api_key") or get_api_key()
    if not key:
        raise ValueError("No OpticOdds API key configured.")
    p = dict(params)
    p["key"] = key
    url = f"{OPTICODDS_BASE}{path}"
    r = requests.get(url, params=p, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_fixtures_for_sport(
    sport_id: str, league_id: str, days: int = FIXTURE_LOOKAHEAD_DAYS
) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=days)
    params = {
        "sport": sport_id,
        "league": league_id,
        "start_date_after": now.isoformat().replace("+00:00", "Z"),
        "start_date_before": end.isoformat().replace("+00:00", "Z"),
    }
    try:
        data = optic_get("/fixtures/active", params)
    except Exception as e:
        st.warning(f"Fixture fetch failed for {sport_id}/{league_id}: {e}")
        return pd.DataFrame()

    rows: List[Dict] = []
    for fx in data.get("data", []):
        try:
            rows.append(
                {
                    "fixture_id": fx["id"],
                    "sport": fx["sport"]["id"],
                    "league": fx["league"]["id"],
                    "start_date": fx.get("start_date"),
                    "home_name": fx.get("home_team_display"),
                    "away_name": fx.get("away_team_display"),
                }
            )
        except Exception:
            continue
    return pd.DataFrame(rows)


def fetch_odds_for_fixtures(
    fixture_ids: List[str],
    markets: List[str],
    sportsbooks: List[str],
) -> pd.DataFrame:
    """
    Safer /fixtures/odds usage:
      - ONE fixture_id per request
      - sportsbook and market as comma-separated strings
    """
    if not fixture_ids:
        return pd.DataFrame()

    rows: List[Dict] = []

    books_param = ",".join(sportsbooks) if sportsbooks else None
    markets_param = ",".join(markets) if markets else None

    import time as _time  # local alias to avoid shadowing

    for idx, fid in enumerate(fixture_ids, start=1):
        params = {
            "fixture_id": fid,
            "odds_format": "AMERICAN",
        }
        if books_param:
            params["sportsbook"] = books_param
        if markets_param:
            params["market"] = markets_param

        try:
            data = optic_get("/fixtures/odds", params)
        except Exception as e:
            st.warning(f"Odds request failed for fixture {fid}: {e}")
            continue

        for fx in data.get("data", []):
            fixture_id = fx["id"]
            sd = fx.get("start_date")
            hname = fx.get("home_team_display")
            aname = fx.get("away_team_display")
            sport = fx.get("sport", {}).get("id")
            league = fx.get("league", {}).get("id")

            for o in fx.get("odds", []):
                try:
                    rows.append(
                        {
                            "fixture_id": fixture_id,
                            "sport": sport,
                            "league": league,
                            "start_date": sd,
                            "home_name": hname,
                            "away_name": aname,
                            "sportsbook": o.get("sportsbook"),
                            "market_id": o.get("market_id") or o.get("market"),
                            "market_label": o.get("market"),
                            "selection": o.get("selection"),
                            "name": o.get("name"),
                            "price": o.get("price"),
                            "grouping_key": o.get("grouping_key"),
                            "points": o.get("points"),
                        }
                    )
                except Exception:
                    continue

        if idx % 5 == 0:
            _time.sleep(0.25)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["price_decimal"] = df["price"].apply(american_to_decimal)
    df["implied_prob"] = df["price"].apply(implied_prob)
    df["open_price"] = df["price"]
    df["open_implied"] = df["implied_prob"]
    return df


def build_snapshot(selected_sports: List[str], books: List[str]) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for alias in selected_sports:
        cfg = SPORTS_CONFIG.get(alias)
        if not cfg:
            continue
        st.write(f"Pulling {alias} fixtures…")
        fx_df = fetch_fixtures_for_sport(cfg["sport"], cfg["league"])
        if fx_df.empty:
            continue
        st.write(f"{alias}: {len(fx_df)} fixtures")
        odds_df = fetch_odds_for_fixtures(fx_df["fixture_id"].tolist(), CORE_MARKETS, books)
        if odds_df.empty:
            continue
        odds_df["alias"] = alias
        parts.append(odds_df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


# -----------------------------------------------------------
# BASELINE FOR LINE MOVES
# -----------------------------------------------------------

def merge_with_baseline(new_board: pd.DataFrame) -> pd.DataFrame:
    if new_board.empty:
        return new_board

    key_cols = ["fixture_id", "sportsbook", "market_id", "selection"]
    baseline = st.session_state.get("baseline_df")
    if baseline is None or baseline.empty:
        baseline = new_board[key_cols + ["open_price", "open_implied"]].copy()
        st.session_state["baseline_df"] = baseline
        return new_board

    merged = new_board.merge(
        baseline,
        on=key_cols,
        how="left",
        suffixes=("", "_base"),
    )

    new_mask = merged["open_price_base"].isna()
    merged["open_price"] = merged["open_price_base"].where(
        ~new_mask, merged["open_price"]
    )
    merged["open_implied"] = merged["open_implied_base"].where(
        ~new_mask, merged["open_implied"]
    )

    if new_mask.any():
        extra = merged.loc[new_mask, key_cols + ["open_price", "open_implied"]]
        baseline = pd.concat([baseline, extra], ignore_index=True)
        baseline = baseline.drop_duplicates(subset=key_cols, keep="first")
        st.session_state["baseline_df"] = baseline

    drop_cols = [c for c in merged.columns if c.endswith("_base")]
    merged = merged.drop(columns=drop_cols)
    return merged


# -----------------------------------------------------------
# EV + EFD
# -----------------------------------------------------------

def efd_row(row: pd.Series) -> float:
    ev = float(row.get("no_vig_ev") or 0.0)
    open_i = float(row.get("open_implied") or 0.0)
    curr_i = float(row.get("curr_implied") or 0.0)
    steam = abs(curr_i - open_i)

    mins = row.get("minutes_to_start")
    try:
        mins = float(mins) if mins is not None else None
    except Exception:
        mins = None

    market = (row.get("market_id") or "").lower()

    ev_clamped = min(max(ev, -0.10), 0.12)
    math_comp = (ev_clamped + 0.10) / 0.22

    steam_clamped = min(steam, 0.10)
    steam_comp = steam_clamped / 0.10

    if mins is None:
        time_comp = 0.5
    elif mins <= 0:
        time_comp = 1.0
    elif mins <= 360:
        time_comp = 1.0 - (mins / 360.0) * 0.5
    else:
        time_comp = 0.5

    if "moneyline" in market:
        market_comp = 1.0
    elif "spread" in market:
        market_comp = 0.9
    else:
        market_comp = 0.85

    score = (0.45 * math_comp + 0.35 * steam_comp + 0.20 * time_comp) * 100.0 * market_comp
    return float(round(score, 1))


def recompute_ev_and_efd(board_df: pd.DataFrame) -> pd.DataFrame:
    if board_df.empty:
        return board_df

    df = board_df.copy()
    df["curr_implied"] = df["price"].apply(implied_prob)
    df["steam"] = df["curr_implied"] - df["open_implied"]

    df["fair_prob"] = np.nan
    df["no_vig_ev"] = np.nan

    group_cols = ["fixture_id", "market_id", "grouping_key"]
    updates: List[pd.DataFrame] = []

    for _, g in df.groupby(group_cols, as_index=False):
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
        df = df.merge(
            upd[group_cols + ["selection", "sportsbook", "fair_prob", "no_vig_ev"]],
            on=group_cols + ["selection", "sportsbook"],
            how="left",
            suffixes=("", "_u"),
        )
        for col in ["fair_prob", "no_vig_ev"]:
            df[col] = df[f"{col}_u"].combine_first(df[col])
        drop_cols = [c for c in df.columns if c.endswith("_u")]
        df = df.drop(columns=drop_cols)

    df["minutes_to_start"] = df["start_date"].apply(minutes_until)
    df["efd_score"] = df.apply(efd_row, axis=1)
    return df


# -----------------------------------------------------------
# ARBITRAGE
# -----------------------------------------------------------

def find_arbitrage(board_df: pd.DataFrame) -> pd.DataFrame:
    if board_df.empty:
        return pd.DataFrame()

    rows: List[Dict] = []
    group_cols = ["fixture_id", "market_id", "grouping_key"]

    for (fid, mid, gk), g in board_df.groupby(group_cols):
        best = (
            g.sort_values("price_decimal", ascending=False)
            .groupby("selection", as_index=False)
            .first()
        )
        if len(best) < 2:
            continue
        best2 = best.nlargest(2, "price_decimal")
        inv_sum = (1.0 / best2["price_decimal"]).sum()
        edge = 1.0 - inv_sum
        if edge <= 0:
            continue
        meta = g.iloc(0) if callable(getattr(g, "iloc", None)) else g.iloc[0]
        meta = g.iloc[0]  # safe
        rows.append(
            {
                "fixture_id": fid,
                "alias": meta.get("alias"),
                "start_date": meta.get("start_date"),
                "home_name": meta.get("home_name"),
                "away_name": meta.get("away_name"),
                "market_label": meta.get("market_label"),
                "edge_pct": edge * 100.0,
                "legs": json.dumps(
                    [
                        {
                            "selection": r["selection"],
                            "name": r["name"],
                            "sportsbook": r["sportsbook"],
                            "price": r["price"],
                        }
                        for _, r in best2.iterrows()
                    ]
                ),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("edge_pct", ascending=False)


# -----------------------------------------------------------
# RENDERING
# -----------------------------------------------------------

def render_header_and_sidebar() -> Tuple[List[str], List[str], bool]:
    inject_theme()
    st.title("⚡ Edge Force Dominion – OpticOdds Live Engine")

    with st.sidebar:
        st.markdown("### OpticOdds API Key")
        key = st.text_input("API Key", value=get_api_key(), type="password")
        st.session_state["api_key"] = key

        st.markdown("### Sports")
        all_sports = list(SPORTS_CONFIG.keys())
        default_sports = all_sports
        selected_sports = st.multiselect(
            "Tracked sports", options=all_sports, default=default_sports
        )

        st.markdown("### Sportsbooks")
        books_text = st.text_area(
            "Comma-separated books",
            ", ".join(DEFAULT_BOOKS),
            height=80,
        )
        books = [b.strip() for b in books_text.split(",") if b.strip()]

        st.markdown("### Auto-refresh")
        auto = st.checkbox("Auto-refresh every 30s", value=False)
        st.session_state["auto_refresh"] = auto

        st.markdown("---")
        run_btn = st.button("🔄 Run fresh snapshot")

    if st.session_state.get("auto_refresh"):
        st_autorefresh(interval=30000, key="efd_auto_refresh")

    return selected_sports, books, run_btn


def render_dashboard(board: pd.DataFrame, arb_df: pd.DataFrame) -> None:
    tabs = st.tabs(["Dashboard", "Arbitrage", "Line Moves"])

    # ----- DASHBOARD TAB -----
    with tabs[0]:
        st.subheader("Market Snapshot")

        if board.empty:
            st.info("No odds data loaded yet. Hit 'Run fresh snapshot' in the sidebar.")
            return

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown('<div class="metric-label">Sports</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value">{board["alias"].nunique()}</div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown('<div class="metric-label">Sportsbooks</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value">{board["sportsbook"].nunique()}</div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown('<div class="metric-label">Fixtures</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value">{board["fixture_id"].nunique()}</div>',
                unsafe_allow_html=True,
            )
        with c4:
            best_efd = float(board["efd_score"].max())
            st.markdown('<div class="metric-label">Top EFD</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value">{best_efd:.1f}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        f1, f2, f3, f4 = st.columns(4)
        with f1:
            sport_f = st.selectbox(
                "Sport",
                ["ALL"] + sorted(board["alias"].unique().tolist()),
            )
        with f2:
            market_f = st.selectbox("Market", ["ALL", "moneyline", "spread", "total"])
        with f3:
            book_f = st.selectbox(
                "Book", ["ALL"] + sorted(board["sportsbook"].unique().tolist())
            )
        with f4:
            sort_f = st.selectbox("Sort by", ["EFD score", "EV", "Steam"])

        df = board.copy()
        if sport_f != "ALL":
            df = df[df["alias"] == sport_f]
        if market_f != "ALL":
            df = df[df["market_id"].str.contains(market_f, case=False, na=False)]
        if book_f != "ALL":
            df = df[df["sportsbook"] == book_f]

        if df.empty:
            st.warning("No rows match current filters.")
            return

        if sort_f == "EV":
            df = df.sort_values("no_vig_ev", ascending=False)
        elif sort_f == "Steam":
            df = df.sort_values("steam", ascending=False)
        else:
            df = df.sort_values("efd_score", ascending=False)

        st.subheader("Top Matchups by EFD")

        meta = (
            df.groupby("fixture_id")
            .agg(
                start_date=("start_date", "first"),
                home_name=("home_name", "first"),
                away_name=("away_name", "first"),
                alias=("alias", "first"),
            )
            .reset_index()
        )
        scores = df.groupby("fixture_id")["efd_score"].max().reset_index(name="top_efd")
        meta = meta.merge(scores, on="fixture_id", how="left")
        meta = meta.sort_values("top_efd", ascending=False).head(10)

        for _, row in meta.iterrows():
            left, right = st.columns([2, 3])
            with left:
                st.markdown('<div class="fixture-card">', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="fixture-header">{row["alias"]} • {row["start_date"]}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="team-name">{row["home_name"]} vs {row["away_name"]}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="efd-pill">EFD {row["top_efd"]:.1f}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with right:
                sub = df[df["fixture_id"] == row["fixture_id"]]

                ml = sub[sub["market_id"].str.contains("moneyline", case=False, na=False)]
                sp = sub[sub["market_id"].str.contains("spread", case=False, na=False)]
                tot = sub[sub["market_id"].str.contains("total", case=False, na=False)]

                def best_line(mdf: pd.DataFrame, label: str) -> str:
                    if mdf.empty:
                        return f"**{label}:** —"
                    top = mdf.sort_values("efd_score", ascending=False).iloc[0]
                    return (
                        f"**{label}:** {top['sportsbook']} – {top['name']} "
                        f"({top['price']}), EFD {top['efd_score']:.1f}"
                    )

                st.markdown(
                    "<div class='odds-row'>"
                    + best_line(ml, "Moneyline")
                    + "<br>"
                    + best_line(sp, "Spread")
                    + "<br>"
                    + best_line(tot, "Total")
                    + "</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("---")

        st.subheader("Full Board (top 250 rows)")
        cols = [
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
        tbl = df[cols].copy()
        tbl = tbl.rename(
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
        tbl["EV%"] = tbl["no_vig_ev"].fillna(0).apply(lambda v: f"{v*100:.1f}%")
        tbl["Steam%"] = tbl["steam"].fillna(0).apply(lambda v: f"{v*100:.1f}%")
        st.dataframe(
            tbl[
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

    # ----- ARBITRAGE TAB -----
    with tabs[1]:
        st.subheader("Arbitrage Opportunities")
        if arb_df.empty:
            st.info("No positive 2-way arbitrage found in this snapshot.")
        else:
            show = arb_df.copy().head(100)
            show = show.rename(
                columns={
                    "alias": "Sport",
                    "start_date": "Start",
                    "home_name": "Home",
                    "away_name": "Away",
                    "market_label": "Market",
                    "edge_pct": "Edge %",
                }
            )
            show["Edge %"] = show["Edge %"].apply(lambda v: f"{v:.2f}%")
            st.dataframe(
                show[["Sport", "Start", "Home", "Away", "Market", "Edge %", "legs"]],
                use_container_width=True,
            )

    # ----- LINE MOVES TAB -----
    with tabs[2]:
        st.subheader("Line Movement vs Session Open")
        if board.empty:
            st.info("No odds loaded.")
            return
        lm = board.copy()
        lm["abs_steam"] = lm["steam"].abs()
        top_moves = (
            lm.sort_values("abs_steam", ascending=False)
            .head(100)[
                [
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
                ]
            ]
            .rename(
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
                    "steam": "Steam (Δ prob)",
                }
            )
        )
        st.dataframe(top_moves, use_container_width=True)


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Edge Force Dominion", layout="wide")
    init_state()
    selected_sports, books, run_btn = render_header_and_sidebar()

    if not st.session_state.get("api_key"):
        st.warning("Add your OpticOdds API key in the sidebar to begin.")
        return

    need_refresh = run_btn or st.session_state["board_df"].empty
    if not need_refresh and st.session_state.get("auto_refresh"):
        need_refresh = True

    if need_refresh:
        with st.spinner("Building multi-sport snapshot from OpticOdds…"):
            board_raw = build_snapshot(selected_sports, books)
            if board_raw.empty:
                st.session_state["board_df"] = pd.DataFrame()
                st.session_state["arb_df"] = pd.DataFrame()
            else:
                board_with_baseline = merge_with_baseline(board_raw)
                board_scored = recompute_ev_and_efd(board_with_baseline)
                st.session_state["board_df"] = board_scored
                st.session_state["arb_df"] = find_arbitrage(board_scored)

    board_df = st.session_state.get("board_df", pd.DataFrame())
    arb_df = st.session_state.get("arb_df", pd.DataFrame())
    render_dashboard(board_df, arb_df)


if __name__ == "__main__":
    main()
