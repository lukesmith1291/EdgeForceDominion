import streamlit as st

# ==============================
# CONFIG
# ==============================
st.set_page_config(
    page_title="Edge Force Dominion – EFD Scoring Engine",
    layout="wide"
)

EFD_WEIGHTS = {
    "ev": 0.40,
    "market_ineff": 0.25,
    "sharp_steam": 0.15,
    "line_velocity": 0.10,
    "public_bias": 0.10,
}


def compute_efd(ev_pct,
                market_ineff,
                sharp_steam,
                line_velocity,
                public_bias,
                committee_override):
    """
    EFD Score formula:

    EFD =
      (EV_Score * 0.40) +
      (Market_Inefficiency * 0.25) +
      (Sharp_Steam * 0.15) +
      (Line_Move_Velocity * 0.10) +
      (Public_Sentiment_Bias * 0.10) +
      (Committee_Override)

    Where:
      - ev_pct is EV in percent (e.g. 6.4 for +6.4% EV)
      - market_ineff, sharp_steam, line_velocity, public_bias are 0–100 scaled scores
      - committee_override is 0–10
    """

    # Clamp EV to [0, 100] (negative EV treated as 0 edge)
    ev_score = max(min(ev_pct, 100.0), 0.0)

    mi_score = max(min(market_ineff, 100.0), 0.0)
    ss_score = max(min(sharp_steam, 100.0), 0.0)
    lv_score = max(min(line_velocity, 100.0), 0.0)
    pb_score = max(min(public_bias, 100.0), 0.0)
    co_score = max(min(committee_override, 10.0), 0.0)

    efd = (
        ev_score * EFD_WEIGHTS["ev"] +
        mi_score * EFD_WEIGHTS["market_ineff"] +
        ss_score * EFD_WEIGHTS["sharp_steam"] +
        lv_score * EFD_WEIGHTS["line_velocity"] +
        pb_score * EFD_WEIGHTS["public_bias"] +
        co_score
    )

    # Clamp final score to [0, 100] for consistency
    efd = max(min(efd, 100.0), 0.0)
    return round(efd, 1)


def tier_from_efd(efd_score: float) -> str:
    if efd_score >= 70:
        return "Tier A – Glitch / Must-Bet"
    elif efd_score >= 55:
        return "Tier B – High-Value"
    elif efd_score >= 40:
        return "Tier C – Situational"
    else:
        return "Do Not Bet"


# ==============================
# UI LAYOUT
# ==============================
st.title("🏆 Edge Force Dominion")
st.subheader("EFD Scoring Engine (Prototype Streamlit UI)")

st.markdown(
    """
This tool lets you plug in the components of an Edge Force Dominion edge and see the resulting **EFD score (0–100)** and Tier.

Use this as a:
- **manual grading UI** now
- and later swap the inputs with live data from **BigQuery / Odds APIs / Google Trends**.
"""
)

# --- LEFT: Inputs ---
left_col, right_col = st.columns([1, 1])

with left_col:
    st.header("📥 Market Inputs")

    st.markdown("### Core Value")
    ev_pct = st.number_input(
        "Expected Value (EV %) – e.g. 6.4 for +6.4% EV",
        min_value=-50.0,
        max_value=50.0,
        value=5.0,
        step=0.1,
        help="If EV is negative, the score will clip it at 0."
    )

    market_ineff = st.number_input(
        "Market Inefficiency Score (0–100)",
        min_value=0.0,
        max_value=100.0,
        value=15.0,
        step=0.5,
        help="How mispriced is this line vs consensus? 0 = none, 100 = huge."
    )

    st.markdown("### Movement & Sharps")
    sharp_steam = st.slider(
        "Sharp Steam Score (0–100)",
        min_value=0,
        max_value=100,
        value=30,
        help="Captures sharp-driven line movement (across key numbers, etc.)."
    )

    line_velocity = st.slider(
        "Line Move Velocity (0–100)",
        min_value=0,
        max_value=100,
        value=20,
        help="How quickly the line has moved in a short window."
    )

    st.markdown("### Public & Narrative")
    public_bias = st.slider(
        "Public Sentiment Bias (0–100)",
        min_value=0,
        max_value=100,
        value=10,
        help="Public hype / Google Trends imbalance / book splits."
    )

    committee_override = st.slider(
        "Committee Override Bonus (+0 to +10)",
        min_value=0,
        max_value=10,
        value=0,
        help="Manual boost for motivation, injuries, revenge, etc."
    )

with right_col:
    st.header("📊 EFD Score")

    efd_score = compute_efd(
        ev_pct=ev_pct,
        market_ineff=market_ineff,
        sharp_steam=sharp_steam,
        line_velocity=line_velocity,
        public_bias=public_bias,
        committee_override=committee_override,
    )

    tier_label = tier_from_efd(efd_score)

    # Big score display
    st.markdown(
        f"""
        <div style="padding: 1.5rem; border-radius: 0.75rem; border: 1px solid #444;">
            <h2 style="margin-bottom: 0.5rem;">EFD Score</h2>
            <h1 style="font-size: 3.5rem; margin: 0;">{efd_score}</h1>
            <p style="font-size: 1.25rem; margin-top: 0.5rem;"><b>{tier_label}</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Component Breakdown")
    st.write(
        {
            "EV Weight (40%)": ev_pct,
            "Market Inefficiency (25%)": market_ineff,
            "Sharp Steam (15%)": sharp_steam,
            "Line Velocity (10%)": line_velocity,
            "Public Bias (10%)": public_bias,
            "Committee Override (+0–10)": committee_override,
        }
    )

    st.markdown("### Tier Rules")
    st.markdown(
        """
- **Tier A**: EFD ≥ 70 — *Glitch / Must-Bet*  
- **Tier B**: 55–69 — *High-Value*  
- **Tier C**: 40–54 — *Situational*  
- **Do Not Bet**: EFD < 40
"""
    )

st.markdown("---")
st.caption(
    "Prototype only. Next step: wire these inputs to live data from Odds APIs, "
    "BigQuery (historical & consensus lines), and Google Trends for public bias."
)