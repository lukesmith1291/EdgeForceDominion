import os
from typing import List, Dict, Any

import requests
import pandas as pd
import streamlit as st

# ============================================================
#  CONFIG & MAPPINGS
# ============================================================

OPTICODDS_BASE = "https://api.opticodds.com/api/v3"

# Map your front-end sport labels to sport + league used by OpticOdds
SPORT_LEAGUE_MAP = {
    "NBA":   {"sport": "basketball", "league": "NBA"},
    "NCAAB": {"sport": "basketball", "league": "NCAAB"},
    "NFL":   {"sport": "football",   "league": "NFL"},
    "NCAAF": {"sport": "football",   "league": "NCAAF"},
    "MLB":   {"sport": "baseball",   "league": "MLB"},
    "NHL":   {"sport": "hockey",     "league": "NHL"},
}

DEFAULT_SPORTSBOOKS = [
    "FanDuel",
    "DraftKings",
    "BetMGM",
    "Caesars",
    "Pinnacle",
    "LowVig",
]

# Core markets (OpticOdds names/ids)
DEFAULT_MARKETS = [
    "moneyline",
    "point_spread",
    "total_points",
]


# ============================================================
#  API HELPERS
# ============================================================

def get_active_fixtures(api_key: str, sport: str, league: str) -> pd.DataFrame:
    """
    Pull active fixtures for a given sport/league using /fixtures/active.
    """
    params = {
        "key": api_key,
        "sport": sport,
        "league": league,
    }
    url = f"{OPTICODDS_BASE}/fixtures/active"
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json().get("data", [])

    if not data:
        return pd.DataFrame()

    rows = []
    for fx in data:
        rows.append(
            {
                "fixture_id": fx.get("id"),
                "sport": fx.get("sport", {}).get("name"),
                "league": fx.get("league", {}).get("name"),
                "start_date": fx.get("start_date"),
                "status": fx.get("status"),
                "home_team": fx.get("home_team_display") or "",
                "away_team": fx.get("away_team_display") or "",
            }
        )
    return pd.DataFrame(rows)


def get_odds_for_fixtures(
    api_key: str,
    fixture_ids: List[str],
    sportsbooks: List[str],
    markets: List[str],
) -> pd.DataFrame:
    """
    Pull odds for up to N fixtures at a time via /fixtures/odds.
    OpticOdds allows up to 5 fixture_ids + 5 sportsbooks per request.
    """
    if not fixture_ids:
        return pd.DataFrame()

    all_rows: List[Dict[str, Any]] = []
    chunk_size = 5
    url = f"{OPTICODDS_BASE}/fixtures/odds"

    for i in range(0, len(fixture_ids), chunk_size):
        chunk = fixture_ids[i : i + chunk_size]
        params: Dict[str, Any] = {
            "key": api_key,
            "fixture_id": chunk,         # list → repeated query param
            "sportsbook": sportsbooks,   # list → repeated query param
            "market": markets,           # list → repeated query param
            "is_main": True,             # main lines only
        }

        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json().get("data", [])

        for fx in data:
            fixture_id = fx.get("fixture_id")
            sport = fx.get("sport")
            league = fx.get("league")
            start_date = fx.get("start_date")
            home = fx.get("home_team_display")
            away = fx.get("away_team_display")

            for odd in fx.get("odds", []):
                all_rows.append(
                    {
                        "fixture_id": fixture_id,
                        "sport": sport,
                        "league": league,
                        "start_date": start_date,
                        "home_team": home,
                        "away_team": away,
                        "sportsbook": odd.get("sportsbook"),
                        "market": odd.get("market"),
                        "selection": odd.get("selection") or odd.get("name"),
                        "price": odd.get("price"),
                        "points": odd.get("points"),
                        "is_live": odd.get("is_live"),
                    }
                )

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df.dropna(subset=["price"])


# ============================================================
#  MATH HELPERS (EFD + ARB)
# ============================================================

def american_to_decimal(odds: float) -> float:
    """
    Convert American odds to decimal.
    """
    o = float(odds)
    if o > 0:
        return 1.0 + (o / 100.0)
    else:
        return 1.0 + (100.0 / abs(o))


def compute_efd_scores(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute an EFD-style score per fixture using:
      - number of books
      - volatility of prices
      - spread between best and worst price

    Placeholder heuristic until you plug in the real EFD formula.
    """
    if odds_df.empty:
        return pd.DataFrame()

    scored_rows: List[Dict[str, Any]] = []

    for fixture_id, grp in odds_df.groupby("fixture_id"):
        sport = grp["sport"].iloc[0]
        league = grp["league"].iloc[0]
        home = grp["home_team"].iloc[0]
        away = grp["away_team"].iloc[0]
        start_date = grp["start_date"].iloc[0]
        num_books = grp["sportsbook"].nunique()

        # Focus on moneyline for scoring base; if none, use all
        ml = grp[grp["market"].str.lower().str.contains("moneyline")]
        if ml.empty:
            ml = grp

        price_std = ml["price"].std() if len(ml) > 1 else 0.0

        best_edges = []
        for sel, s_grp in ml.groupby("selection"):
            if len(s_grp) < 2:
                continue
            best = s_grp["price"].max()
            worst = s_grp["price"].min()
            best_edges.append(abs(best - worst))
        spread_edge = max(best_edges) if best_edges else 0.0

        # Compress into 0–100
        books_factor = min(num_books / 5.0, 1.0)
        vol_factor = min(abs(price_std) / 50.0, 1.0)
        edge_factor = min(abs(spread_edge) / 100.0, 1.0)

        efd = 40 + 30 * books_factor + 20 * vol_factor + 10 * edge_factor
        efd = max(0.0, min(100.0, efd))

        scored_rows.append(
            {
                "fixture_id": fixture_id,
                "sport": sport,
                "league": league,
                "home_team": home,
                "away_team": away,
                "start_date": start_date,
                "num_books": num_books,
                "price_std": price_std,
                "spread_edge": spread_edge,
                "EFD_score": round(efd, 1),
            }
        )

    if not scored_rows:
        return pd.DataFrame()

    df_scored = pd.DataFrame(scored_rows)
    if "EFD_score" not in df_scored.columns:
        return df_scored

    return df_scored.sort_values("EFD_score", ascending=False)


def find_arbitrage(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Very simple 2-way arb detector:
      - For each fixture + market:
        - Take best decimal price per selection across books
        - If sum(1/decimal) < 1 → arb edge exists
    """
    if odds_df.empty:
        return pd.DataFrame()

    arb_rows: List[Dict[str, Any]] = []

    for (fixture_id, market), grp in odds_df.groupby(["fixture_id", "market"]):
        best_rows = []
        for sel, s_grp in grp.groupby("selection"):
            s_grp = s_grp.copy()
            s_grp["decimal"] = s_grp["price"].apply(american_to_decimal)
            best_row = s_grp.loc[s_grp["decimal"].idxmax()]
            best_rows.append(best_row)

        if len(best_rows) < 2:
            continue

        best_df = pd.DataFrame(best_rows)
        inv_sum = (1.0 / best_df["decimal"]).sum()
        if inv_sum >= 1.0:
            continue

        edge = (1.0 - inv_sum) * 100.0
        row = {
            "fixture_id": fixture_id,
            "sport": best_df["sport"].iloc[0],
            "league": best_df["league"].iloc[0],
            "home_team": best_df["home_team"].iloc[0],
            "away_team": best_df["away_team"].iloc[0],
            "market": market,
            "arb_edge_pct": round(edge, 2),
        }

        best_df = best_df.sort_values("decimal", ascending=False).reset_index(drop=True)
        for i in range(min(3, len(best_df))):
            row[f"sel{i+1}_name"] = best_df.loc[i, "selection"]
            row[f"sel{i+1}_book"] = best_df.loc[i, "sportsbook"]
            row[f"sel{i+1}_odds"] = best_df.loc[i, "price"]

        arb_rows.append(row)

    if not arb_rows:
        return pd.DataFrame()

    return pd.DataFrame(arb_rows).sort_values("arb_edge_pct", ascending=False)


# ============================================================
#  NEON THEME
# ============================================================

def apply_neon_theme():
    st.markdown(
        """
        <style>
        body, .stApp {
            background: radial-gradient(circle at top, #020617 0%, #000000 55%);
            color: #e5e7eb;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
        }
        .efd-card {
            border-radius: 18px;
            padding: 1.25rem 1.5rem;
            border: 1px solid rgba(56, 189, 248, 0.7);
            background: linear-gradient(135deg, rgba(15,23,42,0.98), rgba(5,22,45,0.98));
            box-shadow: 0 0 32px rgba(56, 189, 248, 0.5);
        }
        .efd-pill {
            border-radius: 999px;
            padding: 0.35rem 0.8rem;
            border: 1px solid rgba(94, 234, 212, 0.8);
            background: radial-gradient(circle at top, rgba(8,47,73,0.9), rgba(15,23,42,0.95));
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #e0f2fe;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #020617 0%, #020617 60%, #000000 100%);
            border-right: 1px solid rgba(31, 41, 55, 0.9);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            padding-top: 0.35rem;
            padding-bottom: 0.35rem;
            border-radius: 999px !important;
            background: rgba(15,23,42,0.85);
            box-shadow: 0 0 12px rgba(37,99,235,0.3);
        }
        .stTabs [aria-selected="true"] {
            background: radial-gradient(circle at top, #0ea5e9, #1d4ed8);
        }
        table {
            border-radius: 12px;
            overflow: hidden;
        }
        .dataframe tbody tr:hover {
            background-color: rgba(59,130,246,0.18) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
#  MAIN APP
# ============================================================

def main():
    st.set_page_config(
        page_title="Edge Force Dominion – Live Odds Board",
        layout="wide",
    )
    apply_neon_theme()

    # ---------- SIDEBAR ----------
    st.sidebar.markdown("## ⚙️ EFD Control Panel")

    default_key = os.environ.get("OPTICODDS_API_KEY", "")
    api_key = st.sidebar.text_input(
        "OpticOdds API key",
        type="password",
        value=default_key,
        help="You can also set OPTICODDS_API_KEY as an env var.",
    )

    # Sports with ALL option
    sport_options = ["ALL"] + list(SPORT_LEAGUE_MAP.keys())
    selected_sports_raw = st.sidebar.multiselect(
        "Sports",
        sport_options,
        default=["ALL"],
    )

    # Sportsbooks with ALL option
    book_options = ["ALL"] + DEFAULT_SPORTSBOOKS
    selected_books_raw = st.sidebar.multiselect(
        "Sportsbooks (max 5 per request)",
        book_options,
        default=["ALL"],
    )

    # Markets with ALL option
    market_options_friendly = ["ALL", "Moneyline", "Point Spread", "Total Points"]
    selected_markets_friendly_raw = st.sidebar.multiselect(
        "Markets",
        market_options_friendly,
        default=["ALL"],
    )

    max_fixtures_per_sport = st.sidebar.slider(
        "Max fixtures per sport",
        5,
        40,
        20,
        step=5,
    )

    run_scan = st.sidebar.button("🚀 Pull live snapshot")

    # ---------- Resolve ALL selections ----------
    # Sports
    if "ALL" in selected_sports_raw or not selected_sports_raw:
        selected_sports = list(SPORT_LEAGUE_MAP.keys())
    else:
        selected_sports = [s for s in selected_sports_raw if s in SPORT_LEAGUE_MAP]

    # Sportsbooks
    if "ALL" in selected_books_raw or not selected_books_raw:
        selected_books = list(DEFAULT_SPORTSBOOKS)
    else:
        selected_books = [b for b in selected_books_raw if b in DEFAULT_SPORTSBOOKS]

    # Markets → actual OpticOdds IDs
    if "ALL" in selected_markets_friendly_raw or not selected_markets_friendly_raw:
        market_ids = ["moneyline", "point_spread", "total_points"]
    else:
        market_ids: List[str] = []
        for m in selected_markets_friendly_raw:
            if m == "Moneyline":
                market_ids.append("moneyline")
            elif m == "Point Spread":
                market_ids.append("point_spread")
            elif m == "Total Points":
                market_ids.append("total_points")

    # ---------- HEADER ----------
    st.markdown(
        """
        <div class="efd-card">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:1rem;">
            <div>
              <div class="efd-pill">Edge Force Dominion • Live Board</div>
              <h1 style="margin:0.3rem 0 0.4rem 0;">Live Odds & Arbitrage Console</h1>
              <p style="margin:0;color:#bfdbfe;font-size:0.9rem;">
                Ranked matchups by EFD score, odds snapshots per game, and a cross-sport arbitrage radar.
              </p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- GUARDS ----------
    if not api_key:
        st.info("Paste your OpticOdds API key in the sidebar and hit **Pull live snapshot**.")
        return

    if not run_scan:
        st.info("Set your filters on the left, then hit **Pull live snapshot**.")
        return

    if not selected_sports:
        st.warning("No sports selected after resolving ALL. Check your sports filter.")
        return

    if not selected_books:
        st.warning("No sportsbooks selected after resolving ALL. Check your sportsbook filter.")
        return

    if not market_ids:
        st.warning("No markets selected after resolving ALL. Check your market filter.")
        return

    # ---------- PULL DATA ----------
    all_odds_frames = []

    for sport_key in selected_sports:
        meta = SPORT_LEAGUE_MAP[sport_key]
        base_sport = meta["sport"]
        league = meta["league"]

        with st.spinner(f"Pulling active fixtures for {sport_key}..."):
            try:
                fx_df = get_active_fixtures(api_key, base_sport, league)
            except Exception as e:
                st.error(f"{sport_key}: failed to pull fixtures: {e}")
                continue

        if fx_df.empty:
            st.info(f"{sport_key}: no active fixtures with odds right now.")
            continue

        fx_df = fx_df.sort_values("start_date").head(max_fixtures_per_sport)
        fixture_ids = fx_df["fixture_id"].tolist()

        with st.spinner(f"Pulling odds for {sport_key} fixtures..."):
            try:
                odds_df = get_odds_for_fixtures(api_key, fixture_ids, selected_books, market_ids)
            except Exception as e:
                st.error(f"{sport_key}: failed to pull odds: {e}")
                continue

        if odds_df.empty:
            st.info(f"{sport_key}: no odds returned for selected books/markets.")
            continue

        all_odds_frames.append(odds_df)

    if not all_odds_frames:
        st.warning("No odds data came back. Check that your key, sports, books, and markets are valid.")
        return

    odds_all = pd.concat(all_odds_frames, ignore_index=True)

    # ---------- EFD & ARB ----------
    efd_df = compute_efd_scores(odds_all)
    arb_df = find_arbitrage(odds_all)

    # Tabs: one per sport + global arbitrage tab
    sport_tabs = list(selected_sports) + ["ARB"]
    tabs = st.tabs([f"🏟 {s}" if s != "ARB" else "💰 Arbitrage Radar" for s in sport_tabs])

    for s_key, tab in zip(sport_tabs, tabs):
        with tab:
            if s_key == "ARB":
                st.subheader("Cross-Sport Arbitrage Radar")
                if arb_df.empty:
                    st.info("No clear 2-way arbitrage found in this snapshot.")
                else:
                    st.dataframe(
                        arb_df[
                            [
                                "sport",
                                "league",
                                "home_team",
                                "away_team",
                                "market",
                                "arb_edge_pct",
                                "sel1_name",
                                "sel1_book",
                                "sel1_odds",
                                "sel2_name",
                                "sel2_book",
                                "sel2_odds",
                            ]
                        ],
                        use_container_width=True,
                    )
                continue

            # --- Per-sport EFD board ---
            st.subheader(f"{s_key} – Edge Force Dominion Board")

            league_name = SPORT_LEAGUE_MAP[s_key]["league"]
            sport_efd = efd_df[efd_df["league"] == league_name] if not efd_df.empty else pd.DataFrame()

            if sport_efd.empty:
                st.info("No EFD-ranked matchups for this sport in the current snapshot.")
                continue

            ranked_view = sport_efd.copy()
            ranked_view["Matchup"] = ranked_view["away_team"] + " @ " + ranked_view["home_team"]
            ranked_view = ranked_view[
                [
                    "EFD_score",
                    "Matchup",
                    "start_date",
                    "num_books",
                    "price_std",
                    "spread_edge",
                ]
            ].sort_values("EFD_score", ascending=False)

            st.markdown("#### 🧬 Ranked matchups (by EFD score)")
            st.dataframe(ranked_view, use_container_width=True)

            # --- Drilldown for a single matchup ---
            st.markdown("#### 🔍 Drilldown: matchup details")

            fixture_choices = [
                f"{row.away_team} @ {row.home_team} — {row.start_date} — EFD {row.EFD_score}"
                for _, row in sport_efd.iterrows()
            ]
            selected = st.selectbox(
                "Choose a matchup to inspect",
                options=fixture_choices,
            )

            if selected:
                idx = fixture_choices.index(selected)
                fx_row = sport_efd.iloc[idx]
                fx_id = fx_row["fixture_id"]

                st.markdown(
                    f"**{fx_row['away_team']} @ {fx_row['home_team']}**  \n"
                    f"Start: `{fx_row['start_date']}`  \n"
                    f"EFD Score: `{fx_row['EFD_score']}`"
                )

                fx_odds = odds_all[odds_all["fixture_id"] == fx_id].copy()
                if fx_odds.empty:
                    st.info("No odds rows for this fixture (for current filters).")
                else:
                    fx_odds["decimal"] = fx_odds["price"].apply(american_to_decimal)
                    st.markdown("**Live odds snapshot (by market & book)**")
                    st.dataframe(
                        fx_odds[
                            [
                                "sportsbook",
                                "market",
                                "selection",
                                "price",
                                "points",
                                "is_live",
                                "decimal",
                            ]
                        ].sort_values(["market", "selection", "sportsbook"]),
                        use_container_width=True,
                    )

                    fx_arb = find_arbitrage(fx_odds)
                    if not fx_arb.empty:
                        st.markdown("**Arbitrage inside this matchup**")
                        st.dataframe(fx_arb, use_container_width=True)
                    else:
                        st.caption("No self-contained arb for this fixture in this snapshot.")


if __name__ == "__main__":
    main()