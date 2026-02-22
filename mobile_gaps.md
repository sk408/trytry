Web/Mobile vs Desktop Gap Analysis
Priority: HIGH (Core User-Facing Features)
1. Gamecast — Missing Box Score Display
Desktop: Full box score with two team tabs, 11 columns (photo, player, MIN, PTS, REB, AST, STL, BLK, FG, 3PT, +/-), player headshot photos, auto-refreshing. Web: The SSE stream at /api/gamecast/stream/{game_id} already sends boxscore data, but gamecast.html never renders it — there's no box score section in the template.

API: /api/gamecast/stream/{game_id} and /api/gamecast/boxscore/{game_id} already exist
Template: gamecast.html — add a tabbed box score section below the play-by-play
Priority: HIGH
2. Gamecast — Missing Live Prediction / Win Probability
Desktop: Runs live_predict() on every poll cycle, shows predicted spread/total, a win probability bar with team colors, and adjustments over time. Web: Only shows ESPN odds. No model prediction, no win probability bar.

API: Need a new endpoint (e.g. /api/gamecast/prediction/{game_id}) that calls live_predict() with parsed game state, OR embed it in the SSE stream
Template: gamecast.html — add prediction card and win probability bar
Priority: HIGH
3. Gamecast — Missing Court Visualization
Desktop: Has CourtWidget — an animated half-court showing shot locations from play-by-play data. Web: No court visualization at all.

API: Play data already available in the SSE stream; needs client-side rendering
Template: gamecast.html — add a canvas/SVG court element with JS to plot shots
Priority: HIGH
4. Live Games — Missing Spread, O/U, and Betting Recommendations
Desktop: Table has 8 columns including Spread, O/U, and Recommendation (calls get_live_recommendations() for each live game). Web: live.html only shows Status, Away, Score, Home, Clock, Period — no spread, O/U, or recommendations.

API: Need a new endpoint /api/live/recommendations or embed recommendations in the live page's server-side data (the fetch_live_scores() return likely already includes spread/O/U)
Template: live.html — add Spread, O/U, Recommendation columns
Priority: HIGH
5. Live Games — Missing Link to Gamecast
Desktop: Selecting a live game row can navigate you into the Gamecast view for details. Web: No clickable rows or links from live games to gamecast.

API: None needed — just add links
Template: live.html — make rows clickable, link to /gamecast?game_id=X
Priority: HIGH
6. Matchup — Missing Player Rosters with Injury Status
Desktop: Two full roster tables (home & away) showing every player with: name, position, PPG, MPG, injury status, injury reason, and impact estimate (e.g. "-24.5 pts"). Web: matchups.html shows a simple injury section with just player name + status — no roster tables, no PPG/MPG, no impact estimates.

API: The predict_matchup() result dict already contains injury info, but the web route at app.py doesn't fetch roster data. Need to add roster+stats query.
Template: matchups.html — add two roster tables below prediction
Priority: HIGH
7. Matchup — Missing Game Picker from Schedule
Desktop: Has a "pick a game" combo that loads the next 14 days of scheduled games so users can quickly predict a specific matchup without manually selecting teams. Web: Only has manual team dropdowns.

API: Schedule data already available from the existing route
Template: matchups.html — add a <select> populated with upcoming games that auto-submits home/away
Priority: HIGH
8. Matchup — Missing Confidence Score
Desktop: Shows 6 prediction cards: Spread, Total, Home Score, Away Score, Win Probability, Confidence. Web: Shows spread, total, scores, win prob — but no confidence metric.

API: The predict_matchup() result already contains confidence; it's just not passed to the template
Template: matchups.html — add confidence display
Priority: HIGH
Priority: MEDIUM (Important Quality-of-Life)
9. Players — Missing Search/Filter
Desktop: Has a search text box with live filtering of the player list. Web: players.html renders all 300 players with no search functionality.

API: Can be done client-side with JS table filtering
Template: players.html — add search input with JS filter
Priority: MEDIUM
10. Players — Missing Player Photos
Desktop: Both injured and all-players tables show circular player headshot photos. Web: No photos anywhere.

API: Need to serve player photos via a static/API route (e.g. /static/player_photos/ or /api/player/photo/{id})
Template: players.html — add <img> tags
Priority: MEDIUM
11. Players — Missing Play Probability / Availability Estimates
Desktop: Injured table shows rich availability labels like "QUESTIONABLE 50%", "OUT ~3d", "OUT FOR SEASON" with color-coded probability using _estimate_play_probability(). Web: Only shows raw status (Out, Questionable, etc.) with no probability or estimated return.

API: Need to run _estimate_play_probability() server-side in the players route or add expected_return to the query
Template: players.html — add Availability and Return columns
Priority: MEDIUM
12. Players — Missing Impact Classification (KEY / ROTATION / BENCH)
Desktop: Injured players table classifies each by MPG impact: KEY (25+ mpg), ROTATION (15-25), BENCH (<15) with color coding. Web: No impact column or classification.

API: MPG data would need to be queried and classified in the route
Template: players.html — add Impact column
Priority: MEDIUM
13. Schedule — Missing Team Logos
Desktop: Schedule table shows team logos next to team names. Web: schedule.html has plain text only.

API: Need a route to serve team logos (e.g. /api/team/logo/{id})
Template: schedule.html — add <img> tags
Priority: MEDIUM
14. Schedule — Missing Date Range Picker
Desktop: Has a "From" date picker and configurable day range. Web: Hard-coded to today + 14 days with no user control.

API: The route already accepts no parameters — add query params for date range
Template: schedule.html — add date input and days selector
Priority: MEDIUM
15. Schedule — Cross-Navigation to Matchup Doesn't Pre-populate
Desktop: "Predict" button auto-selects teams in the Matchup tab and triggers prediction immediately. Web: schedule.html links to /matchups?home_team=X&away_team=Y which does work, but the flow could be smoother (no visual feedback that prediction is running).

Template: Minor — add loading indicator in matchups.html
Priority: MEDIUM
16. Dashboard — Missing Force Sync Option
Desktop: Has 7 sync buttons including "Force Full Sync" (bypasses freshness checks). Web: dashboard.html has standard "Full Sync" but no force option.

API: The /api/sync/data endpoint would need a ?force=true parameter to pass through
Template: dashboard.html — add a "Force Sync" button
Priority: MEDIUM
17. Gamecast — Missing Quarter-by-Quarter Line Score
Desktop: ScoreboardWidget shows per-quarter scores for both teams (Q1, Q2, Q3, Q4, OT columns). Web: gamecast.html only shows total scores.

API: Quarter data is available in the SSE stream (competitors.linescores)
Template: gamecast.html — add a quarter score table in the scoreboard
Priority: MEDIUM
18. Notification System
Desktop: Has NotificationBell widget in tab bar corner + InjuryMonitor service for real-time injury alerts. Web: No notification system at all.

API: Need a notifications SSE endpoint or polling endpoint (the backend src/notifications/ module exists)
Template: base.html — add notification bell/badge in navbar
Priority: MEDIUM
19. Gamecast — Missing Team Colors / Theming per Game
Desktop: Uses get_team_colors() from nba_colors.py to theme the scoreboard, court, and win probability bar with actual team brand colors. Web: Uses generic single-color styling.

API: Could serve team colors via API or embed in SSE response
Template: gamecast.html and style.css — add dynamic CSS variables
Priority: MEDIUM
Priority: LOW (Nice-to-Have / Polish)
20. All-Star Weekend View — Entirely Missing
Desktop: Full AllStarView with 4 sub-tabs: MVP predictions, 3PT Contest, Rising Stars, Game Winner — each with scoring models, betting tables, odds, and edge calculations. Web: No All-Star page exists at all. Not in nav, no template.

API: Need new /allstar route and template; data is currently hardcoded in the desktop view
Template: New allstar.html needed
Priority: LOW (seasonal/event feature)
21. Gamecast — Missing Play Feed Team Attribution
Desktop: PlayFeedWidget attributes plays to home/away team with team colors and ESPN team IDs. Web: Plays are flat text with clock only — no team attribution or color coding.

API: Team IDs are in the play data but not parsed in the JS
Template: gamecast.html — enhance updatePlays() JS
Priority: LOW
22. Live Games — Missing Auto-Refresh Toggle
Desktop: Has a "Auto-refresh (30s)" checkbox the user can toggle off. Web: live.html auto-refreshes unconditionally via setTimeout(() => location.reload(), 30000) — no way to pause. Also does full page reload rather than AJAX.

Template: live.html — add toggle checkbox, switch to fetch() instead of location.reload()
Priority: LOW
23. Live Games — Missing ET→Local Time Conversion
Desktop: Converts "7:00 PM ET" to local time (e.g. "4:00 PM PT") for scheduled games. Web: Shows raw status text with no timezone conversion.

API: Could be done server-side in the route or client-side with JS
Template: live.html — add JS time conversion
Priority: LOW
24. Dashboard — Sync Log Not Styled by Message Type
Desktop: Log messages are color-coded: errors in red, success in green, progress in blue. Web: dashboard.html uses startSSE() which renders all messages the same color in .log-line.

Template: base.html and style.css — parse message content and apply CSS classes
Priority: LOW
25. Dashboard — Missing Pipeline State / Full Pipeline Button Feedback
Desktop: The pipeline has detailed state tracking. Web has the button but minimal feedback.

API: Exists (/api/full-pipeline)
Template: dashboard.html — add a second log area or reuse the existing one
Priority: LOW
26. Web Nav Still Shows Accuracy/Autotune/Admin Links
User stated mobile doesn't need accuracy/tuning. base.html still shows Accuracy, Autotune, and Admin in the nav.

Template: base.html — remove or conditionally hide these links for the mobile-focused web version
Priority: LOW
27. Players — Truncated to 300 Rows
Web: players.html hard-limits rendering to 300 players (players[:300]). Desktop shows up to 600.

Template: players.html — implement virtual scrolling or pagination instead of hard truncation
Priority: LOW
28. Missing PWA / Mobile Optimization
Web: base.html has viewport meta and hamburger menu, but there's no service worker, no manifest, no offline support, and no touch-optimized interactions.

Template: Add manifest.json, service worker, and touch gesture support
Priority: LOW
Summary Table
#	Gap	Template(s)	API Status	Priority
1	Gamecast box score	gamecast.html	Exists	HIGH
2	Gamecast live prediction + win prob	gamecast.html	Needs new endpoint	HIGH
3	Gamecast court visualization	gamecast.html	Data exists in stream	HIGH
4	Live games spread/O/U/recommendations	live.html	Needs new endpoint	HIGH
5	Live → Gamecast navigation	live.html	None needed	HIGH
6	Matchup player rosters	matchups.html + app.py	Needs route changes	HIGH
7	Matchup game picker	matchups.html	Data exists	HIGH
8	Matchup confidence score	matchups.html	Data exists in result	HIGH
9	Players search/filter	players.html	Client-side JS	MEDIUM
10	Player photos	players.html + new route	Needs image serving	MEDIUM
11	Players play probability	players.html + app.py	Needs route changes	MEDIUM
12	Players impact classification	players.html + app.py	Needs route changes	MEDIUM
13	Schedule team logos	schedule.html + new route	Needs image serving	MEDIUM
14	Schedule date range picker	schedule.html + app.py	Needs route changes	MEDIUM
15	Schedule → Matchup loading indicator	matchups.html	None needed	MEDIUM
16	Dashboard force sync	dashboard.html + app.py	Needs param support	MEDIUM
17	Gamecast quarter scores	gamecast.html	Data exists in stream	MEDIUM
18	Notification system	base.html + new route	Needs new endpoint	MEDIUM
19	Team color theming	gamecast.html + style.css	Needs color API	MEDIUM
20	All-Star view	New template	Needs new route	LOW
21	Play feed team attribution	gamecast.html	Data in stream	LOW
22	Live auto-refresh toggle	live.html	None needed	LOW
23	ET→local time conversion	live.html	Client-side JS	LOW
24	Styled sync log messages	dashboard.html + style.css	None needed	LOW
25	Pipeline feedback	dashboard.html	Exists	LOW
26	Hide accuracy/autotune/admin nav	base.html	None needed	LOW
27	Player list pagination	players.html	None needed	LOW
28	PWA / offline support	base.html + new files	Needs new files	LOW
