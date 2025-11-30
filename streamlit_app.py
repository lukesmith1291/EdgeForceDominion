import json
from datetime import datetime, timezone
from typing import Dict, Any, List

import requests
import pandas as pd
import streamlit as st


BASE_URL = "https://api.opticodds.com/api/v3"

# Map simple labels to OpticOdds sport codes.
# You may need to tweak these to match your plan’s docs.
SPORT_CODES = {
    "NFL": "football_nfl",
    "NCAAF": "football_ncaaf",
    "NBA": "basketball_nba",
    "NCAAB": "basketball_ncaab",
    "NHL": "icehockey_nhl",
    "MLB": "baseball_mlb",
}


# ---------------------------
# Helpers
# ---------------------------

def get_api_key() -> str:
    """Read OpticOdds API key from Streamlit secrets."""
    try:
        return st.secrets["optic_odds"]["api_key"]
    except Exception:
        return ""


@st.cache_data(ttl=600, show_spinner=False)
def fetch_fixtures_raw(api_key: str, sport_code: str) -> Dict[str, Any]:
    """
    Raw call to fixtures endpoint.
    We are deliberately NOT doing any fancy flattening here yet.
    """
    url = f"{BASE_URL}/fixtures"
    params = {
        "key": api_key,
        "sport": sport_code,
        "event_status": "upcoming,live",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=600, show_spinner=False)
def fetch_fixtures_table(api_key: str, sport_code: str) -> pd.DataFrame:
    """
    Use fetch_fixtures_raw, then try to flatten into a simple table:
    fixture_id, home_team, away_team, league, start_time.
    """
    raw = fetch_fixtures_raw(api_key, sport_code)

    # Try to find the list of fixtures in a robust way.
    if isinstance(raw, list):
        fixtures = raw
    elif isinstance(raw, dict):
        if isinstance(raw.get("fixtures"), list):
            fixtures = raw["fixtures"]
        elif isinstance(raw.get("data"), list):
            fixtures = raw["data"]
        else:
            # If all else fails, treat it as a single-item list if it looks like a dict
            fixtures = [raw] if "fixture_id" in raw or "id" in raw else []
    else:
        fixtures = []

    rows: List[Dict[str, Any]] = []

    for fx in fixtures:
        if not isinstance(fx, dict):
            continue
        fixture_id = fx.get("fixture_id") or fx.get("id")
        if not fixture_id:
            continue

        home = fx.get("home_team") or fx.get("home")
        away = fx.get("away_team") or fx.get("away")
        league = fx.get("league") or fx.get("competition")
        ts_raw = fx.get("start_time") or fx.get("start_timestamp")

        # Attempt to parse date/time, but keep original as text too.
        start_time_parsed = None
        if isinstance(ts_raw, (int, float)):
            start_time_parsed = datetime.fromtimestamp(ts_raw, tz=timezone.utc)
        elif isinstance(ts_raw, str):
            try:
                start_time_parsed = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except Exception:
                start_time_parsed = None

        rows.append(
            {
                "fixture_id": fixture_id,
                "home_team": home,
                "away_team": away,
                "league": league,
                "start_time_raw": ts_raw,
                "start_time_parsed": start_time_parsed,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "fixture_id",
                "home_team",
                "away_team",
                "league",
                "start_time_raw",
                "start_time_parsed",
            ]
        )

    return pd.DataFrame(rows)


# ---------------------------
# Streamlit UI
# ---------------------------

def main():
    st.set_page_config(
        page_title="EDGE|FORCE – OpticOdds Smoke Test",
        layout="wide",
    )

    st.title("🏆 EDGE|FORCE Dominion – OpticOdds Smoke Test")

    api_key = get_api_key()
    if not api_key:
        st.error(
            "No OpticOdds API key found in secrets.\n\n"
            "In Streamlit Cloud → Settings → Secrets, paste:\n\n"
            "[optic_odds]\napi_key = \"YOUR_KEY_HERE\""
        )
        return

    st.success("✅ OpticOdds API key loaded from secrets.")

    # Sport selector
    sport_label = st.selectbox(
        "Choose a sport to test fixtures for:",
        options=list(SPORT_CODES.keys()),
        index=0,
    )
    sport_code = SPORT_CODES[sport_label]

    if st.button("🔍 Fetch fixtures for this sport"):
        with st.spinner(f"Calling OpticOdds /fixtures for {sport_label}…"):
            try:
                # Show raw JSON for debugging
                raw = fetch_fixtures_raw(api_key, sport_code)
                st.subheader("Raw JSON (first 1000 chars)")
                st.code(json.dumps(raw, indent=2)[:1000] + " ...", language="json")

                # Show flattened table
                st.subheader("Flattened fixtures table")
                fixtures_df = fetch_fixtures_table(api_key, sport_code)
                if fixtures_df.empty:
                    st.warning("No fixtures parsed. Check the raw JSON above.")
                else:
                    st.dataframe(fixtures_df, use_container_width=True)
            except requests.HTTPError as http_err:
                st.error(f"HTTP error from OpticOdds: {http_err}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()