Understood.
Here is the entire README in one single copy-paste block with ZERO code windows, ZERO backticks, ZERO fenced blocks.
Everything is plain text so you can paste directly into Google AI Studio, GitHub, Notepad, anywhere.

⸻

BEGIN README (NO CODE BLOCKS)

Hot Picks HQ — Edge Force Dominion

Live Odds Engine • Market Inefficiency Scanner • EV Calculator • Steam Tracker
Powered by Optic Odds • Streamlit • Google AI Studio

Edge Force Dominion (EFD) is a real-time sports betting intelligence system that pulls live odds from Optic Odds, removes vig, calculates true probabilities, identifies mispriced lines, evaluates expected value (EV), tracks line movement (steam) across 200+ sportsbooks, generates a proprietary EFD Score from 0–100, and displays everything through a modern Streamlit dashboard. The system integrates directly with Google AI Studio to generate daily newsletters, pick cards, parlay recommendations, and market insights.

SYSTEM ARCHITECTURE
User → Streamlit UI → Optic Odds API → Data Pipeline → No-Vig Engine → EV Engine → EFD Score → Steam Tracker → Output Dashboard → AI-Generated Newsletter

REQUIREMENTS
APIs required: Optic Odds
APIs optional: SerpAPI, SportsRadar, ESPN
Google AI Studio (Gemini) for narrative output
Python libraries: streamlit, pandas, numpy, requests, plotly, pydantic

OPTIC ODDS SETUP
Create a config file with your Optic Odds API key.
Use Optic Odds endpoints to fetch real-time prices across all major American sportsbooks.

DATA FLOW
	1.	Pull odds for every market from 200+ sportsbooks
	2.	Convert American odds to decimal
	3.	Compute implied probabilities
	4.	Remove vig to get fair probabilities
	5.	Compute fair odds
	6.	Compare each sportsbook price to fair odds
	7.	Calculate expected value (EV)
	8.	Detect market mispricing and arbitrage
	9.	Track opening line vs live line movement (steam)
	10.	Compute full EFD Score
	11.	Display all intelligence inside the UI
	12.	Send structured pick data to Google AI Studio for narrative creation

CORE MATHEMATICS

American to Decimal Conversion:
Positive odds → decimal = (odds / 100) + 1
Negative odds → decimal = (100 / abs(odds)) + 1

No-Vig Probability Calculation:
	1.	Convert both sides to decimal
	2.	Compute implied probability for each side
	3.	Add implied probabilities to get total vig
	4.	Divide each implied probability by total vig to get fair probability

Fair Odds:
1 / fair_probability

Expected Value (EV):
(fair_probability × decimal_odds) − 1

Market Deviation Index:
(best_price − consensus_price) / consensus_price

Bookmaker Disagreement Index:
Standard deviation of all available prices across books

Sharp Money Delta:
Absolute line movement × strength of sharp-side betting splits

Steam Intensity:
Absolute change from opening line divided by minutes since opening

PROPRIETARY EFD SCORE FORMULA
This system uses the following weighted scoring model to create a 0–100 rating:

EFD =
25% Market Deviation
20% Expected Value
15% Line Movement Strength
10% Bookmaker Disagreement
10% Sharp Money Delta
10% Betting Splits Pressure
10% AI Projection Gap
Multiply total by 100 to scale to a 0–100 score.

EFD SCORE TIERS
Tier A: 80–100 (Glitch, major misprice, strongest plays)
Tier B: 65–79 (Strong value)
Tier C: 50–64 (Mild value)
Below 50: No significant edge

STEAM TRACKER RULES
Steam Event Trigger:
Absolute line change of 1.0 point (or equivalent price movement) from open.

Reverse Line Movement:
Public bets increase on a team while the line moves against them.

Frozen Line Indicator:
Line remains unchanged for 30 minutes or longer.

STREAMLIT UI FEATURES
• Live odds board
• EV heatmaps
• Steam visualization over time
• Price comparison across all books
• Arbitrage/misprice detection
• Betting splits (if available)
• Custom filters: sport, sportsbook list, odds range, EV threshold
• Tier A / Tier B / Tier C pick lists
• Parlay builder
• Export system for Google AI Studio to generate the daily Hot Picks HQ newsletter

PROJECT STRUCTURE
edge-force-dominion
	•	app.py (Streamlit app)
	•	engine/odds.py (Optic Odds API requests)
	•	engine/math.py (probabilities, EV, fair odds)
	•	engine/scoring.py (EFD score engine)
	•	engine/tracker.py (steam logic)
	•	engine/utils.py (helpers)
	•	config.py (API keys)
	•	README.md
	•	requirements.txt

RUNNING THE APPLICATION LOCALLY
Install packages
Run Streamlit to start the dashboard

GOOGLE AI STUDIO INTEGRATION
Streamlit sends structured data including picks, EVs, splits, steam, and notes to Google AI Studio.
AI Studio returns a completed narrative newsletter including:
• Top plays
• Tier listings
• Market analysis
• EV reasoning
• Steam summaries
• Parlay of the day

MONETIZATION FEATURES
• Premium picks feed
• Tier A alert system
• Live steam tracker
• Parlay generator
• Arbitrage notifications
• Discord access
• Premium EV board
• Real-time line movement monitor
• Daily newsletter subscription

SUMMARY
This README provides the full Edge Force Dominion specification: live odds ingestion, no-vig calculation, EV detection, market inefficiency scanning, EFD scoring system, steam logic, UI design, architecture, deployment, and newsletter generation flow using Google AI Studio. This system is ready to power Hot Picks HQ as a full professional betting intelligence platform.

⸻

END README (NO CODE BLOCKS)

If you want the full working code in one single copy-paste window (also with zero code blocks), tell me:

“Give me the full code with no code windows.”