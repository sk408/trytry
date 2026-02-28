# NBA Underdog Prediction System

## The Thesis

Every bettor can pick favorites. The lines are priced for it — a -300 favorite needs to win 75% of the time just to break even. The entire sportsbook industry is built on the assumption that the public will pile onto chalk. They're right.

The edge is on the other side. NBA underdogs in the 2–8 point spread zone win roughly 35–40% of their games outright. At average moneyline odds of +150 to +200, you only need to hit 40% to be profitable. The question isn't "who will win?" — it's "which dogs are underpriced?"

This system exists to answer that question.

---

## Architecture

```
                    ┌─────────────────────────────┐
                    │      RAW DATA LAYER          │
                    │  (never changes once stored)  │
                    ├─────────────────────────────┤
                    │  Player game logs (550+)      │
                    │  Team advanced metrics         │
                    │  Quarter-by-quarter scores     │
                    │  Injury history & status logs  │
                    │  Vegas odds (open + close)     │
                    │  Sharp money % vs public %     │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │    PRECOMPUTED GAME LAYER     │
                    │  (rebuild only on new games)   │
                    ├─────────────────────────────┤
                    │  Player splits (20-game window)│
                    │  Aggregate team projections     │
                    │  Defensive efficiency factors   │
                    │  Four Factors edges             │
                    │  Fatigue penalties              │
                    │  Clutch performance metrics     │
                    │  Hustle/effort differentials    │
                    │  Historical injury context      │
                    │  Home court advantage           │
                    │  Vegas lines & sharp money      │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │    EVALUATION LAYER           │
                    │  (applied fresh every run)     │
                    ├─────────────────────────────┤
                    │  Autotune corrections (from DB) │
                    │  30+ tunable weight params      │
                    │  Per-team weight overrides       │
                    │  ML ensemble blend (XGBoost)    │
                    │  Residual calibration buckets   │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │    VALUE IDENTIFICATION       │
                    │  (where the money is)          │
                    ├─────────────────────────────┤
                    │  Model spread vs Vegas spread   │
                    │  Dog pick detection (2-8 zone)  │
                    │  Sharp money agreement signal   │
                    │  Confidence scoring              │
                    │  ROI-weighted recommendations    │
                    └─────────────────────────────┘
```

The key architectural decision: **precomputed games are raw facts that never need rebuilding.** Tuning, weights, and ML predictions are applied at evaluation time. This means the optimizer can run 50 passes overnight without touching the precomputed cache. The raw data is the foundation; everything above it is a lens.

---

## The Prediction Engine

### Stage 1: Player-Level Projections

Every prediction starts with individual players, not team averages.

For each player on each team:
- **20-game rolling splits** with exponential decay (recent games matter more)
- **Location splits** — how they perform home vs away (graduated weighting)
- **Minutes budget** — 240 minutes per team, scaled by who's available
  - When a starter is out, bench players inherit minutes proportionally
  - Short-handed teams get a 0.70 dampening factor (fatigue compounds)
- **Opponent context** — defensive efficiency of the team they're facing

This gives us a points/rebounds/assists/steals/blocks/turnovers projection for every player, which aggregates into team-level projections.

### Stage 2: Team-Level Adjustments

Raw player aggregations get adjusted by team-level factors:

| Factor | What It Captures | Why It Matters for Dogs |
|--------|-----------------|------------------------|
| **Defensive efficiency** | Opponent's ability to suppress scoring | Dogs with elite defense keep games close |
| **Four Factors** | eFG%, TOV%, OReb%, FT rate | Process metrics predict outcomes better than results |
| **Fatigue** | Back-to-back, 3-in-4, 4-in-6 nights | Favorites on B2Bs are the #1 dog opportunity |
| **Clutch performance** | Last 6 min, within 6 points | Dogs that perform in crunch time win close games |
| **Hustle metrics** | Deflections, contested shots | Effort metrics predict upset likelihood |
| **Pace** | Possessions per game | High-pace games have more variance = dog-friendly |
| **Home court** | 3-point baseline, per-team tuned | Home dogs are the sweet spot |

### Stage 3: Market Comparison

The model's spread prediction is compared against the Vegas line:

```
Model says: Home team -3.5
Vegas says: Home team -7.0
Gap: 3.5 points

→ The away team (dog) is being overpriced by 3.5 points.
→ At +240 ML, this is a value play.
```

The "value zone" is spreads between 2 and 8 points. Too tight (< 2) and it's a toss-up with no real dog value. Too wide (> 8) and the favorite usually wins for a reason.

### Stage 4: Sharp Money Confirmation

Action Network provides public betting percentages vs money percentages:
- **Public %** = percentage of bets on each side (ticket count)
- **Money %** = percentage of dollars on each side (weighted by bet size)

When money % significantly exceeds public % on the dog side, sharp bettors (who bet bigger) are backing the dog. This is a confirmation signal, not a primary signal — but when the model AND sharp money agree on a dog, the hit rate goes up materially.

---

## The Optimization Philosophy

### What We Optimize For

The loss function for the "ml" (moneyline) target:

```
loss = spread_mae * 0.15        # Keep predictions reasonable
     + total_mae * 0.10         # Totals matter less for dog picks
     - dog_roi * 0.70           # PRIMARY: profitable underdogs
     - dog_hit_bonus * 0.30     # Hit rate above 40% break-even
     - ml_roi * 0.20            # Overall ML profitability (secondary)
     + fav_penalty              # Punish ML win rate > 60%
     + no_dog_penalty           # Punish models that never pick dogs
```

**The anti-favorite ramp** is critical. NBA favorites win ~67% of games. If the model's ML win rate climbs toward 67%, it's just picking chalk — which anyone can do. The penalty kicks in at 60% and scales linearly: at 67% it adds +2.8 to the loss. This forces the optimizer to find weight configurations where dogs hit profitably, even if overall win rate drops to 55-60%.

### What We Don't Optimize For

- **ATS (Against the Spread)** — Rewards being close to Vegas on either side. A model can win ATS by picking the favorite to cover OR the dog to cover. It doesn't care. This is useless for directional betting.
- **Spread MAE alone** — Minimizing prediction error pushes toward "agree with Vegas." Vegas is already extremely accurate. Matching them is not an edge.
- **Overall ML win rate** — Picking every favorite gives you ~67% win rate and negative ROI. A 55% win rate with smart dog picks is far more profitable.

### The Optimization Loop

```
Pipeline Pass 1 (full):
  1. Sync data (players, teams, injuries, odds)
  2. Build injury history
  3. Autotune per-team corrections
  4. Load precomputed games (from disk cache)
  5. Train XGBoost models (spread + total)
  6. Optimize weights (3000 Optuna trials, ML target)
  7. Per-team refinement (500 trials × 30 teams)
  8. Residual calibration
  9. Validation backtest

Pipeline Pass 2+ (overnight loop):
  A. Optimize weights (3000 trials, fresh random seed)
  B. Per-team refinement (500 trials)
  C. Residual calibration
  D. Backtest → compare DogROI to previous best
  (repeat until time budget exhausted)
```

Each pass uses fresh random seeds. The best result is tracked by DogROI (primary) with dog hit rate as tiebreak. The precomputed game cache is built once and reused across all passes.

---

## Sensitivity Analysis

Before trusting optimized weights, sweep each parameter across its full range and measure the impact on dog profitability:

For each of 30+ parameters:
1. Fix all other weights at their current values
2. Sweep this parameter from extreme low to extreme high (200 steps)
3. Record: loss, ML ROI, DogROI, dog hit rate, win% at each step
4. Find the optimal value for each metric
5. **Flag if the optimal value falls outside the optimizer's search range**

This last point is important. If sensitivity analysis shows that `clutch_scale = 0.8` maximizes DogROI but the optimizer only searches `[0.0, 0.5]`, we're leaving money on the table. The ranges need to be expanded.

Coordinate descent then iterates: pick the parameter with the most impact, set it to optimal, re-sweep everything. Repeat until convergence.

---

## What Makes a Good Dog Pick

The ideal underdog play has several of these characteristics:

1. **Spread 2–8 points** — Close enough that the dog wins outright 35-40% of the time
2. **Model disagrees with Vegas by 3+ points** — Our projection says this game is closer than the market thinks
3. **Home dog** — Home court advantage adds 3 points. A home team getting +5 is really only a 2-point dog by neutral-site standards
4. **Favorite on a back-to-back** — The market is slow to price fatigue. A team playing their 3rd game in 4 nights is significantly weaker than their season numbers suggest
5. **Sharp money on the dog** — When professional bettors are backing the dog, they see something the public doesn't
6. **Dog has elite defense** — Defensive teams keep games close. A team ranked top-10 in defensive efficiency that's getting +6 is dangerous
7. **High pace matchup** — More possessions = more variance = more opportunity for upsets
8. **Clutch performance** — Dogs that have historically performed well in crunch time (last 6 min, within 6 points) are more likely to close out tight games

Not every pick will have all of these. But the more boxes checked, the higher the confidence.

---

## Data Sources & Their Value

| Source | Data | Edge Value | Notes |
|--------|------|------------|-------|
| **NBA Stats API** | Player game logs, team metrics, advanced stats | Foundation | 550+ players, full season, 20-game rolling window |
| **ESPN Gamecast** | Live play-by-play, substitutions | Live adjustments | Real-time injury detection, momentum tracking |
| **Action Network** | Vegas odds, opening lines, sharp money splits | Market pricing | The line IS the market. Sharp money % is the confirmation signal |
| **Injury Reports** | Player status (Out, Doubtful, Questionable, Probable) | Mispricing detection | Market is slow to adjust for role players. Stars get priced in immediately |
| **Historical Injury Inference** | Who actually played in each game | Backtest accuracy | Can't backtest injury impact without knowing who was actually injured |
| **Quarter Scores** | Period-by-period scoring | Clutch validation | Proves which teams actually perform in close games |

---

## The XGBoost Ensemble

Two separate gradient-boosted models provide an independent prediction signal:

- **Spread model** — Predicts home team margin from 50+ game features
- **Total model** — Predicts combined score from pace/efficiency features

Both use: 500 trees, max_depth=3, L2 regularization. Conservative depth prevents overfitting to specific team matchups.

The ML ensemble is blended at a tunable weight (default 0.33) with the statistical prediction. This provides:
- **Regularization** — XGBoost captures nonlinear interactions the linear model misses
- **Diversification** — Two independent signals reduce variance
- **Feature discovery** — The model can find interactions (e.g., fast pace + poor defense + B2B = blowout) that manual weights miss

---

## Risk Management

### What We Track

| Metric | Target | Why |
|--------|--------|-----|
| Dog ROI | > 0% (positive) | Are we making money on dog picks? |
| Dog Hit Rate | > 40% | At +150 avg odds, 40% is roughly break-even |
| Dog Pick Rate | 15-30% of value zone games | Being selective, not betting everything |
| ML Win Rate | 55-62% | Below 55% is noise; above 62% is picking too many favorites |
| Spread MAE | < 10 points | Sanity check — predictions should be reasonable |

### What We Don't Track as Primary

| Metric | Why Not |
|--------|---------|
| ATS rate | Rewards agreement with Vegas on either side |
| Overall accuracy | Picking favorites gives 67%, which is unprofitable |
| Spread ≤5 rate | Being close to Vegas means you ARE Vegas |

---

## System Components

### Desktop Application (PySide6)

12-tab interface:
- **Dashboard** — Today's games, predictions, dog picks highlighted
- **Accuracy** — Backtest results with dog metrics prominently displayed (DogROI, DogHit, DogPicks as top-3 cards)
- **Schedule** — Full season schedule with predictions, odds, notes
- **Gamecast** — Live game tracking with play-by-play
- **Sensitivity** — Parameter sweep analysis with outside-range detection
- **Pipeline** — One-click full pipeline / overnight optimization

### Web Application (Flask)

Mobile-friendly interface with:
- Live gamecast with SSE streaming (scores, odds, game flow update every 10s)
- Accuracy dashboard matching desktop (dog metrics first)
- Responsive game selector (start times for pre-game, quarter/clock/score for live)

### Overnight Pipeline

Set it and forget it. Run `overnight` with an 8-hour budget:
1. Full data sync + pipeline (Pass 1)
2. Repeated optimization loops with fresh random seeds (Pass 2+)
3. Each pass backtests and compares DogROI to the best so far
4. Morning: weights are tuned for maximum dog profitability

---

## Ideas for the Future

### Lineup-Adjusted Projections

Current system uses injury reports (Out/Doubtful/etc.) and inferred absences. Next step: use actual starting lineup announcements (available ~30 min before tip) to adjust projections in real-time. A bench player starting is very different from a star starting.

### Closing Line Value (CLV) Tracking

CLV is the single best predictor of long-term betting profitability. If you consistently bet dogs at +200 and the line closes at +180, you're finding value before the market corrects. Track:
- Our bet line vs closing line for every dog pick
- CLV% over time (should be positive)
- Which game situations produce the most CLV

### Momentum / In-Game Modeling

Use quarter-by-quarter scoring patterns to identify:
- Teams that start slow but finish strong (good live-bet dog targets)
- Teams that blow leads (fade in 4Q)
- Scoring run patterns that predict final margin

### Weather / Travel Distance

Not typically thought of for basketball, but altitude (Denver), time zone changes (East Coast team playing late Pacific), and consecutive road games matter more than the market prices in.

### Referee Tendencies

Different ref crews call games differently. Some crews call more fouls (higher totals, more FTs = dog-friendly for teams that get to the line). Historical ref assignment data is available and could be a signal.

### Hedging / Parlay Strategy

Once individual game dog picks are solid:
- **Correlated parlays** — Two dogs in the same time slot whose odds are positively correlated (both playing favorites on B2Bs)
- **Live hedging** — If a dog pick leads at halftime, hedge with a live bet on the favorite to lock in profit
- **Kelly criterion** — Size bets based on edge confidence, not flat units

### Market Timing

Lines move. Open lines are set by algorithms; closing lines incorporate sharp money. The optimal time to bet dogs is:
- After opening (when lines are weakest)
- Before sharp money moves the line toward the dog (capturing maximum value)
- Track line movement patterns per bookmaker

---

## Philosophy

The system's job is not to predict basketball games accurately. Vegas already does that better than any model can. The system's job is to find the 15-25% of games where the market is wrong about underdogs — and to be right about those games often enough to be profitable.

We don't need to beat Vegas on every game. We need to beat them on the games we choose to bet. That's a fundamentally different problem, and it's the one this system is designed to solve.
