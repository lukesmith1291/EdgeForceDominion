# streamlit_app.py
"""
Edge Force Dominion – OpticOdds Global Odds Engine (v1.2)

- Snapshot odds loader across core US sports
- No-vig EV + custom EFD scoring
- Arbitrage radar
- Matchup cards sorted by EFD
- Line move tracking using a session-long baseline (reverse line movement visible as steam)

NOTE: This is a pull-based dashboard (snapshot + optional auto-refresh).
SSE streaming can be added later, but this version should RUN CLEAN.
"""
import os
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Optional autorefresh – safe even if package is not installed
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:  # define no-op if module missing
    def st_autorefresh(*args, **kwargs):
        return None


# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
OPTICODDS_BASE = "https://api.opticodds.com/api/v3"

SPORTS_CONFIG = {
    "NBA":   {"sport": "basketball", "league": "nba"},
    "NCAAB": {"sport": "basketball", "league": "ncaab"},
    "NFL":   {"sport": "football",   "league": "nfl"},
    "NCAAF": {"sport": "football",   "league": "ncaaf"},
    "MLB":   {"sport": "baseball",   "league": "mlb"},
    "NHL":   {"sport": "ice_hockey", "league": "nhl"},
}

CORE_MARKETS = ["moneyline", "spread", "total_points"]
DEFAULT_BOOKS = ["DraftKings", "FanDuel", "Caesars", "BetMGM"]

MAX_FIXTURES_PER_CALL = 20           # keep URLs safe
FIXTURE_LOOKAHEAD_DAYS = 2           # how far ahead to fetch fixtures


# ------------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------------
def get_api_key() -> str:
    """Read API key from Streamlit secrets, env var, or session."""
    # Streamlit Cloud secret
    try:
        return st.secrets["opticodds"]["api_key"]
    except Exception:
        pass

    # Local dev: env var
    key = os.getenv("OPTICODDS_API_KEY", "")
    if key:
        return key

    # Last resort: whatever is already in session
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


def implied_prob_from_american(odds: float) -> float:
    dec = american_to_decimal(odds)
    return 1.0 / dec if dec > 1.0 else 0.0


def minutes_until(iso_timestamp: Optional[str]) -> Optional[float]:
    if not iso_timestamp:
        return None
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
        return (dt - datetime.now(timezone.utc)).total_seconds() / 60.0
    except Exception:
        return None


# ------------------------------------------------------------------
# THEME
# ------------------------------------------------------------------
def inject_theme():
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top left,#020617 0,#000 40%,#020617 100%);
            color: #e2f3ff;
        }
        h1, h2, h3 {
            color: #e5f0ff;
            text-shadow: 0 0 12px rgba(96,165,250,.85);
        }
        .fixture-card {
            border-radius: 16px;
            padding: 12px 14px;
            background: linear-gradient(135deg,rgba(15,118,110,.15),rgba(59,130,246,.10));
            border: 1px solid rgba(148,163,184,.7);
            box-shadow: 0 0 18px rgba(56,189,248,.25);
        }
        .fixture-header {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: .08em;
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
        .efd-chip {
            display:inline-block;
            padding:2px 8px;
            border-radius:999px;
            background:rgba(56,189,248,.16);
            border:1px solid rgba(125,211,252,.7);
            font-size:0.75rem;
            margin-left:6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# API WRAPPERS
# ------------------------------------------------------------------
def optic_get(path: str, params: Dict) -> Dict:
    key = st.session_state.get("api_key") or get_api_key()
    if not key:
        raise ValueError("No OpticOdds API key configured.")
    params = dict(params)
    params["key"] = key
    url = f"{OPTICODDS_BASE}{path}"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_fixtures_for_sport(
    sport_id: str, league_id: str, days: int = FIXTURE_LOOKAHEAD_DAYS
) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    before = now + timedelta(days=days)
    payload = {
        "sport": sport_id,
        "league": league_id,
        "start_date_after": now.isoformat().replace("+00:00", "Z"),
        "start_date_before": before.isoformat().replace("+00:00", "Z"),
    }
    data = optic_get("/fixtures/active", payload)
    rows = []
    for fx in data.get("data", []):
        rows.append(
            {
                "fixture_id": fx["id"],
                "sport": fx["sport"]["id"],
                "league": fx["league"]["id"],
                "start_date": fx.get("start_date"),
                "home_name": fx.get("home_team_display"),
                "away_name": fx.get("away_team_display"),
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
    Pull odds for chunks of fixtures using /fixtures/odds.
    Pass lists; requests turns them into repeated query keys.
    """
    if not fixture_ids:
        return pd.DataFrame()

    rows: List[Dict] = []

    for i in range(0, len(fixture_ids), MAX_FIXTURES_PER_CALL):
        chunk = fixture_ids[i : i + MAX_FIXTURES_PER_CALL]
        try:
            payload = {
                "fixture_id": chunk,         # repeated keys
                "sportsbook": sportsbooks,   # repeated keys
                "market": markets,           # repeated keys
                "odds_format": "AMERICAN",
            }
            data = optic_get("/fixtures/odds", payload)
        except Exception as e:
            st.warning(f"Odds chunk failed ({chunk[:3]}…): {e}")
            continue

        for fx in data.get("data", []):
            fid = fx["id"]
            sd = fx.get("start_date")
            hname = fx.get("home_team_display")
            aname = fx.get("away_team_display")
            sport = fx.get("sport", {}).get("id")
            league = fx.get("league", {}).get("id")

            for o in fx.get("odds", []):
                rows.append(
                    {
                        "fixture_id": fid,
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

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["price_decimal"] = df["price"].apply(american_to_decimal)
    df["implied_prob"] = df["price"].apply(implied_prob_from_american)
    # baseline copies; may be replaced by session baseline later
    df["open_price"] = df["price"]
    df["open_implied"] = df["implied_prob"]
    return df


def build_snapshot(selected_sports: List[str], books: List[str]) -> pd.DataFrame:
    """Pull fixtures + odds for selected sports and return combined raw board df."""
    all_parts = []
    for sport_name in selected_sports:
        cfg = SPORTS_CONFIG[sport_name]
        st.write(f"Pulling {sport_name} fixtures…")
        fx_df = fetch_fixtures_for_sport(cfg["sport"], cfg["league"])
        if fx_df.empty:
            continue
        fixture_ids = fx_df["fixture_id"].tolist()
        st.write(f"{sport_name}: {len(fixture_ids)} fixtures")
        odds_df = fetch_odds_for_fixtures(fixture_ids, CORE_MARKETS, books)
        if odds_df.empty:
            continue
        odds_df["alias"] = sport_name
        all_parts.append(odds_df)

    if not all_parts:
        return pd.DataFrame()

    board = pd.concat(all_parts, ignore_index=True)
    return board


# ------------------------------------------------------------------
# BASELINE FOR LINE MOVES
# ------------------------------------------------------------------
def merge_with_baseline(new_board: pd.DataFrame) -> pd.DataFrame:
    """
    Maintain a session-long baseline for line movement:
    - On first snapshot, baseline = current prices.
    - On later snapshots, keep the earliest seen price as open_* for each key.
    """
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

    # rows not yet in baseline → open_*_base is NaN
    new_mask = merged["open_price_base"].isna()

    # where baseline exists, use its open_*; otherwise keep current
    merged["open_price"] = merged["open_price_base"].where(
        ~new_mask, merged["open_price"]
    )
    merged["open_implied"] = merged["open_implied_base"].where(
        ~new_mask, merged["open_implied"]
    )

    # extend baseline with any brand-new keys
    if new_mask.any():
        new_baseline_rows = merged.loc[
            new_mask, key_cols + ["open_price", "open_implied"]
        ]
        baseline = pd.concat([baseline, new_baseline_rows], ignore_index=True)
        baseline = baseline.drop_duplicates(subset=key_cols, keep="first")
        st.session_state["baseline_df"] = baseline

    drop_cols = [c for c in merged.columns if c.endswith("_base")]
    merged = merged.drop(columns=drop_cols)

    return merged


# ------------------------------------------------------------------
# EV + EFD SCORING
# ------------------------------------------------------------------
def recompute_ev_and_efd(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # current implied
    df["curr_implied"] = df["price"].apply(implied_prob_from_american)
    df["steam"] = df["curr_implied"] - df["open_implied"]

    # fair prob / no-vig EV within each fixture/market/group
    df["fair_prob"] = np.nan
    df["no_vig_ev"] = np.nan

    group_cols = ["fixture_id", "market_id", "grouping_key"]
    updates = []
    for _, g in df.groupby(group_cols, as_index=False):
        # best price per selection
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

    # time to start
    df["minutes_to_start"] = df["start_date"].apply(minutes_until)
    df["efd_score"] = df.apply(efd_row, axis=1)
    return df


def efd_row(row: pd.Series) -> float:
    """
    Edge Force Dominion scoring:
    - math component: no-vig EV
    - steam component: change in implied prob vs open
    - time component: closer to start -> hotter
    - market weighting: ML > spread > total
    """
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

    # clamp EV to [-10%, +12%]
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


# ------------------------------------------------------------------
# ARBITRAGE SCAN
# ------------------------------------------------------------------
def find_arbitrage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple 2-way arbitrage:
    - within each fixture/market/grouping, look at all selections across books
    - if 1 / best_price_home + 1 / best_price_away < 1 → positive arb
    """
    if df.empty:
        return pd.DataFrame()

    arb_rows: List[Dict] = []

    group_cols = ["fixture_id", "market_id", "grouping_key"]
    for (fid, mid, gk), g in df.groupby(group_cols):
        # group by selection, pick best price per selection
        best = (
            g.sort_values("price_decimal", ascending=False)
            .groupby("selection", as_index=False)
            .first()
        )
        if len(best) < 2:
            continue
        # assume binary market: take two best prices
        best2 = best.nlargest(2, "price_decimal")
        inv_sum = (1.0 / best2["price_decimal"]).sum()
        edge = 1.0 - inv_sum
        if edge <= 0:
            continue

        meta = g.iloc[0]
        arb_rows.append(
            {
                "fixture_id": fid,
                "sport": meta.get("sport"),
                "alias": meta.get("alias"),
                "start_date": meta.get("start_date"),
                "home_name": meta.get("home_name"),
                "away_name": meta.get("away_name"),
                "market_id": mid,
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

    if not arb_rows:
        return pd.DataFrame()

    arb_df = pd.DataFrame(arb_rows).sort_values("edge_pct", ascending=False)
    return arb_df


# ------------------------------------------------------------------
# RENDERING
# ------------------------------------------------------------------
def render_header_and_sidebar():
    inject_theme()
    st.title("⚡ Edge Force Dominion – OpticOdds Live Engine")

    with st.sidebar:
        st.markdown("### 🔐 OpticOdds API Key")
        key = st.text_input("API Key", value=get_api_key(), type="password")
        st.session_state["api_key"] = key

        st.markdown("### 🏟 Sports")
        all_sports = list(SPORTS_CONFIG.keys())
        default_sel = all_sports
        selected_sports = st.multiselect(
            "Tracked sports", options=all_sports, default=default_sel
        )

        st.markdown("### 📚 Sportsbooks")
        books_text = st.text_area("Comma separated", ", ".join(DEFAULT_BOOKS))
        books = [b.strip() for b in books_text.split(",") if b.strip()]

        st.markdown("### ⏱ Data")
        auto_refresh = st.checkbox("Auto-refresh snapshot every 30s", value=False)
        st.session_state["auto_refresh"] = auto_refresh

        st.markdown("---")
        run_btn = st.button("🔄 Run fresh snapshot")

    if st.session_state.get("auto_refresh"):
        st_autorefresh(interval=30000, key="efd_auto_refresh")

    return selected_sports, books, run_btn


def render_dashboard(board: pd.DataFrame, arb_df: pd.DataFrame):
    tabs = st.tabs(["🏠 Dashboard", "💰 Arbitrage", "📈 Line Moves"])

    # ---------------- Dashboard tab ----------------
    with tabs[0]:
        st.subheader("Market Snapshot")

        if board.empty:
            st.info("No odds data loaded yet. Hit **Run fresh snapshot** in the sidebar.")
            return

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sports tracked", len(board["alias"].unique()))
        with col2:
            st.metric("Sportsbooks", len(board["sportsbook"].unique()))
        with col3:
            st.metric("Rows (markets)", len(board))
        with col4:
            best_efd = float(board["efd_score"].max())
            st.metric("Top EFD score", f"{best_efd:.1f}")

        st.markdown("---")

        # filters
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sport_f = st.selectbox(
                "Sport", ["ALL"] + sorted(board["alias"].unique().tolist())
            )
        with c2:
            market_f = st.selectbox("Market", ["ALL", "moneyline", "spread", "total"])
        with c3:
            book_f = st.selectbox(
                "Book", ["ALL"] + sorted(board["sportsbook"].unique().tolist())
            )
        with c4:
            sort_f = st.selectbox("Sort by", ["EFD score", "EV", "Steam"])

        df = board.copy()
        if sport_f != "ALL":
            df = df[df["alias"] == sport_f]
        if market_f != "ALL":
            df = df[df["market_id"].str.contains(market_f, case=False, na=False)]
        if book_f != "ALL":
            df = df[df["sportsbook"] == book_f]

        if df.empty:
            st.warning("No rows match filters.")
            return

        if sort_f == "EV":
            df = df.sort_values("no_vig_ev", ascending=False)
        elif sort_f == "Steam":
            df = df.sort_values("steam", ascending=False)
        else:
            df = df.sort_values("efd_score", ascending=False)

        # ---- Matchup cards
        st.markdown("### 🔥 Top Matchups by EFD")

        agg_cols = {
            "start_date": "first",
            "home_name": "first",
            "away_name": "first",
            "alias": "first",
        }
        fixture_meta = df.groupby("fixture_id").agg(agg_cols).reset_index()
        efd_max = df.groupby("fixture_id")["efd_score"].max().reset_index(
            name="max_efd"
        )
        fixture_meta = fixture_meta.merge(efd_max, on="fixture_id", how="left")
        fixture_meta = fixture_meta.sort_values("max_efd", ascending=False).head(8)

        for _, row in fixture_meta.iterrows():
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
                    f'<div class="efd-chip">EFD {row["max_efd"]:.1f}</div>',
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

        # ---- Full board table
        st.markdown("### 📋 Full Board (top 250 rows)")
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

    # ---------------- Arbitrage tab ----------------
    with tabs[1]:
        st.subheader("Arbitrage Radar (2-way)")

        if arb_df is None or arb_df.empty:
            st.info("No positive 2-way arbitrage found in this snapshot.")
        else:
            st.markdown(
                f"Found **{len(arb_df)}** arb spots where combined implied probability < 100%."
            )
            show = arb_df.head(100).copy()
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
                show[
                    [
                        "Sport",
                        "Start",
                        "Home",
                        "Away",
                        "Market",
                        "Edge %",
                        "legs",
                    ]
                ],
                use_container_width=True,
            )

    # ---------------- Line moves tab ----------------
    with tabs[2]:
        st.subheader("Line Movement (Steam) – Session baseline vs latest snapshot")
        if board.empty:
            st.info("No odds loaded.")
            return

        line_df = board.copy()
        line_df["abs_steam"] = line_df["steam"].abs()
        top_moves = (
            line_df.sort_values("abs_steam", ascending=False)
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
                    "steam": "Steam Δ (prob)",
                }
            )
        )
        st.dataframe(top_moves, use_container_width=True)


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Edge Force Dominion", layout="wide")
    selected_sports, books, run_btn = render_header_and_sidebar()

    if not st.session_state.get("api_key"):
        st.warning("Add your OpticOdds API key in the sidebar to begin.")
        return

    need_refresh = run_btn or "board_df" not in st.session_state
    if not need_refresh and st.session_state.get("auto_refresh"):
        # auto-refresh → new snapshot each rerun
        need_refresh = True

    if need_refresh:
        with st.spinner("Building fresh multi-sport snapshot from OpticOdds…"):
            try:
                raw_board = build_snapshot(selected_sports, books)
            except Exception as e:
                st.error(f"Snapshot failed: {e}")
                return

            if raw_board.empty:
                st.session_state["board_df"] = pd.DataFrame()
                st.session_state["arb_df"] = pd.DataFrame()
            else:
                board_with_baseline = merge_with_baseline(raw_board)
                board = recompute_ev_and_efd(board_with_baseline)
                st.session_state["board_df"] = board
                st.session_state["arb_df"] = find_arbitrage(board)

    board_df = st.session_state.get("board_df", pd.DataFrame())
    arb_df = st.session_state.get("arb_df", pd.DataFrame())
    render_dashboard(board_df, arb_df)


if __name__ == "__main__":
    main()
