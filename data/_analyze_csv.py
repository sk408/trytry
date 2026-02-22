"""Comprehensive analysis of diagnostic CSV — Round 5 (all 30 teams)."""
import csv, sys, statistics, math
from collections import defaultdict

sys.path.insert(0, ".")

rows = []
with open("data/diagnostic_worst_teams.csv") as f:
    reader = csv.DictReader(f)
    for r in reader:
        for k in r:
            try:
                r[k] = float(r[k])
            except (ValueError, TypeError):
                pass
        rows.append(r)

N = len(rows)
out = []
def p(s=""): out.append(s)

p(f"Total rows: {N}")

# ════════════════════════════════════════════════
# 1. OVERALL ACCURACY
# ════════════════════════════════════════════════
wc = sum(1 for r in rows if r["winner_correct"] in (1.0, True, "True"))
w5 = sum(1 for r in rows if r["spread_within_5"] in (1.0, True, "True"))
se = [abs(r["spread_error"]) for r in rows]
te = [abs(r["total_error"]) for r in rows]
ps = [r["pred_spread"] for r in rows]
acs = [r["actual_spread"] for r in rows]

p("\n" + "=" * 70)
p("1. OVERALL ACCURACY")
p("=" * 70)
p(f"  Winner accuracy:   {wc}/{N} ({100*wc/N:.1f}%)")
p(f"  Spread within 5:   {w5}/{N} ({100*w5/N:.1f}%)")
p(f"  Spread MAE:        {sum(se)/N:.2f}")
p(f"  Spread median err: {statistics.median(se):.2f}")
p(f"  Total MAE:         {sum(te)/N:.2f}")
p(f"  Total median err:  {statistics.median(te):.2f}")
p(f"  Avg |pred_spread|: {sum(abs(s) for s in ps)/N:.1f}")
p(f"  Avg |actual_spread|: {sum(abs(s) for s in acs)/N:.1f}")
p(f"  Spread RMSE:       {math.sqrt(sum(e**2 for e in se)/N):.2f}")

# ════════════════════════════════════════════════
# 2. SCORE SANITY
# ════════════════════════════════════════════════
p("\n" + "=" * 70)
p("2. SCORE SANITY")
p("=" * 70)
ah = [r["actual_home_score"] for r in rows]
aa = [r["actual_away_score"] for r in rows]
ph = [r["pred_home_score"] for r in rows]
pa = [r["pred_away_score"] for r in rows]
at = [r["actual_total"] for r in rows]
pt = [r["pred_total"] for r in rows]
p(f"  Actual home: {sum(ah)/N:.1f}  pred home: {sum(ph)/N:.1f}")
p(f"  Actual away: {sum(aa)/N:.1f}  pred away: {sum(pa)/N:.1f}")
p(f"  Actual total: {sum(at)/N:.1f}  pred total: {sum(pt)/N:.1f}  bias: {(sum(pt)-sum(at))/N:+.1f}")

# ════════════════════════════════════════════════
# 3. SPREAD DISTRIBUTION
# ════════════════════════════════════════════════
p("\n" + "=" * 70)
p("3. SPREAD DISTRIBUTION (pred vs actual)")
p("=" * 70)
for lo, hi in [(0,3),(3,6),(6,10),(10,15),(15,25),(25,99)]:
    pc = sum(1 for s in ps if lo <= abs(s) < hi)
    ac = sum(1 for s in acs if lo <= abs(s) < hi)
    p(f"  |spread| {lo:2d}-{hi:2d}: pred {pc:4d} ({100*pc/N:5.1f}%)   actual {ac:4d} ({100*ac/N:5.1f}%)")

# ════════════════════════════════════════════════
# 4. ADJUSTMENTS: MAGNITUDE & HELPFULNESS
# ════════════════════════════════════════════════
p("\n" + "=" * 70)
p("4. ADJUSTMENT IMPACT ANALYSIS")
p("=" * 70)
adj_cols = ["home_court_adv", "fatigue_adj", "turnover_adj", "rebound_adj",
            "rating_matchup_adj", "four_factors_adj", "clutch_adj", "hustle_adj",
            "pace_adj", "ml_blend_adj"]
for col in adj_cols:
    vals = [r[col] for r in rows if isinstance(r[col], float)]
    abs_vals = [abs(v) for v in vals]
    nz = sum(1 for v in vals if abs(v) > 0.01)
    helps = hurts = net = 0
    for r in rows:
        adj = r[col]
        if abs(adj) < 0.01: continue
        actual = r["actual_spread"]
        err_with = abs(r["pred_spread"] - actual)
        err_without = abs((r["pred_spread"] - adj) - actual)
        d = err_without - err_with
        net += d
        if d > 0: helps += 1
        elif d < 0: hurts += 1
    tot = helps + hurts
    helpful_pct = 100*helps/tot if tot else 0
    p(f"  {col:25s}: |avg|={sum(abs_vals)/max(len(abs_vals),1):6.2f}  [{min(vals):+7.2f},{max(vals):+7.2f}]  "
      f"active={nz:3d}/{N}  helps={helps:3d} hurts={hurts:3d} ({helpful_pct:.0f}%) net={net:+.0f}pts")

# Combined adjustments impact
raw_pred = [r["home_proj_pts"] - r["away_proj_pts"] + r["home_court_adv"] for r in rows]
tuned_adj = sum(abs(r["home_tune_home_corr"]) + abs(r["away_tune_away_corr"]) for r in rows) / N
p(f"\n  Avg autotune magnitude: {tuned_adj:.2f} pts/game")
# Pred_spread contribution breakdown
avg_hca = sum(r["home_court_adv"] for r in rows) / N
avg_rate = sum(r["rating_matchup_adj"] for r in rows) / N
avg_ff = sum(r["four_factors_adj"] for r in rows) / N
avg_ml = sum(r["ml_blend_adj"] for r in rows) / N
avg_fat = sum(r["fatigue_adj"] for r in rows) / N
avg_to = sum(r["turnover_adj"] for r in rows) / N
avg_pace = sum(r["pace_adj"] for r in rows) / N
avg_cl = sum(r["clutch_adj"] for r in rows) / N
p(f"  Avg contribution (signed): HCA={avg_hca:+.2f} rate={avg_rate:+.2f} ff={avg_ff:+.2f} ml={avg_ml:+.2f} fat={avg_fat:+.2f} to={avg_to:+.2f} pace={avg_pace:+.2f} cl={avg_cl:+.2f}")

# ════════════════════════════════════════════════
# 5. SPREAD ERROR BY DIRECTION
# ════════════════════════════════════════════════
p("\n" + "=" * 70)
p("5. SPREAD DIRECTION & MAGNITUDE ANALYSIS")
p("=" * 70)
right = [r for r in rows if r["pred_spread"] * r["actual_spread"] > 0]
wrong = [r for r in rows if r["pred_spread"] * r["actual_spread"] < 0]
p(f"  Right winner: {len(right)}/{N} ({100*len(right)/N:.1f}%)")
if right:
    p(f"    avg |pred|={sum(abs(r['pred_spread']) for r in right)/len(right):.1f}  "
      f"avg |actual|={sum(abs(r['actual_spread']) for r in right)/len(right):.1f}  "
      f"MAE={sum(abs(r['spread_error']) for r in right)/len(right):.1f}")
p(f"  Wrong winner: {len(wrong)}/{N} ({100*len(wrong)/N:.1f}%)")
if wrong:
    p(f"    avg |pred|={sum(abs(r['pred_spread']) for r in wrong)/len(wrong):.1f}  "
      f"avg |actual|={sum(abs(r['actual_spread']) for r in wrong)/len(wrong):.1f}  "
      f"MAE={sum(abs(r['spread_error']) for r in wrong)/len(wrong):.1f}")

# Conservatism: when right direction, how under/over do we predict?
if right:
    under = sum(1 for r in right if abs(r["pred_spread"]) < abs(r["actual_spread"]))
    over = len(right) - under
    p(f"  When RIGHT: under-predict magnitude {under} / over-predict {over}")

# ════════════════════════════════════════════════
# 6. AUTOTUNE
# ════════════════════════════════════════════════
p("\n" + "=" * 70)
p("6. AUTOTUNE")
p("=" * 70)
hh = [r["home_tune_home_corr"] for r in rows]
aa_t = [r["away_tune_away_corr"] for r in rows]
at_cap_h = sum(1 for v in hh if abs(v) >= 5.99)
at_cap_a = sum(1 for v in aa_t if abs(v) >= 5.99)
p(f"  At ±6 cap: home={at_cap_h}/{N} ({100*at_cap_h/N:.0f}%), away={at_cap_a}/{N} ({100*at_cap_a/N:.0f}%)")
# Net autotune spread bias
net_tunes = [r["home_tune_home_corr"] - r["away_tune_away_corr"] for r in rows]
p(f"  Net tune spread bias: mean={sum(net_tunes)/N:+.2f}, |mean|={sum(abs(x) for x in net_tunes)/N:.2f}")

# Per team
team_tunes = defaultdict(list)
for r in rows:
    ft = r["focus_team"]
    team_tunes[ft].append(r)
for team in sorted(team_tunes.keys()):
    entries = team_tunes[team]
    h_games = [r for r in entries if r["home_team"] == team]
    a_games = [r for r in entries if r["away_team"] == team]
    ht = sum(r["home_tune_home_corr"] for r in h_games)/max(len(h_games),1) if h_games else 0
    at_v = sum(r["away_tune_away_corr"] for r in a_games)/max(len(a_games),1) if a_games else 0
    p(f"  {team}: home_corr={ht:+.2f} ({len(h_games)}g)  away_corr={at_v:+.2f} ({len(a_games)}g)")

# ════════════════════════════════════════════════
# 7. PROJECTION ANALYSIS
# ════════════════════════════════════════════════
p("\n" + "=" * 70)
p("7. BASE PROJECTION ANALYSIS")
p("=" * 70)
hp = [r["home_proj_pts"] for r in rows]
ap = [r["away_proj_pts"] for r in rows]
both_fb = sum(1 for r in rows if abs(r["home_proj_pts"]-112)<0.01 and abs(r["away_proj_pts"]-112)<0.01)
p(f"  Fallback (both=112): {both_fb}/{N} ({100*both_fb/N:.1f}%)")
# Compare projection spread vs actual
proj_diffs = [r["home_proj_pts"] - r["away_proj_pts"] for r in rows]
# Raw projection MAE (before adjustments)
raw_proj_mae = sum(abs(pd - r["actual_spread"]) for pd, r in zip(proj_diffs, rows)) / N
p(f"  Raw projection spread MAE: {raw_proj_mae:.2f}")
p(f"  Final prediction spread MAE: {sum(se)/N:.2f}")
p(f"  => Adjustments {'help' if sum(se)/N < raw_proj_mae else 'HURT'} by {abs(raw_proj_mae - sum(se)/N):.2f} pts")

# ════════════════════════════════════════════════
# 8. CONFIDENCE vs ACCURACY
# ════════════════════════════════════════════════
p("\n" + "=" * 70)
p("8. CONFIDENCE TIERS")
p("=" * 70)
for lo, hi, label in [(0,0.2,"<0.2"),(0.2,0.4,"0.2-0.4"),(0.4,0.6,"0.4-0.6"),(0.6,0.8,"0.6-0.8"),(0.8,1.1,">0.8")]:
    tier = [r for r in rows if isinstance(r["confidence"], float) and lo <= r["confidence"] < hi]
    if tier:
        mae = sum(abs(r["spread_error"]) for r in tier) / len(tier)
        w_pct = 100 * sum(1 for r in tier if r["winner_correct"] in (1.0, True, "True")) / len(tier)
        w5_pct = 100 * sum(1 for r in tier if r["spread_within_5"] in (1.0, True, "True")) / len(tier)
        p(f"  {label:8s}: {len(tier):3d} games, winner={w_pct:4.0f}%, MAE={mae:5.1f}, within5={w5_pct:4.0f}%")

# ════════════════════════════════════════════════
# 9. MONTH BREAKDOWN
# ════════════════════════════════════════════════
p("\n" + "=" * 70)
p("9. ACCURACY BY MONTH")
p("=" * 70)
ms = defaultdict(lambda: {"g": 0, "w": 0, "se": [], "w5": 0})
for r in rows:
    m = str(r["game_date"])[:7]
    ms[m]["g"] += 1
    if r["winner_correct"] in (1.0, True, "True"): ms[m]["w"] += 1
    ms[m]["se"].append(abs(r["spread_error"]))
    if r["spread_within_5"] in (1.0, True, "True"): ms[m]["w5"] += 1
for m in sorted(ms):
    s = ms[m]
    p(f"  {m}: {s['g']:3d}g  winner={100*s['w']/s['g']:5.1f}%  spread_MAE={sum(s['se'])/len(s['se']):5.1f}  within5={100*s['w5']/s['g']:4.0f}%")

# ════════════════════════════════════════════════
# 10. PER-TEAM
# ════════════════════════════════════════════════
p("\n" + "=" * 70)
p("10. PER-TEAM ACCURACY (sorted by spread MAE)")
p("=" * 70)
ts = defaultdict(lambda: {"g": 0, "w": 0, "se": [], "te": [], "w5": 0})
for r in rows:
    ft = r["focus_team"]
    ts[ft]["g"] += 1
    if r["winner_correct"] in (1.0, True, "True"): ts[ft]["w"] += 1
    ts[ft]["se"].append(abs(r["spread_error"]))
    ts[ft]["te"].append(abs(r["total_error"]))
    if r["spread_within_5"] in (1.0, True, "True"): ts[ft]["w5"] += 1
for t in sorted(ts, key=lambda t: sum(ts[t]["se"])/ts[t]["g"]):
    s = ts[t]
    p(f"  {t}: {s['g']:2d}g  winner={100*s['w']/s['g']:5.1f}%  spread_MAE={sum(s['se'])/len(s['se']):5.1f}  total_MAE={sum(s['te'])/len(s['te']):5.1f}  within5={100*s['w5']/s['g']:4.0f}%")

# ════════════════════════════════════════════════
# 11. TOP 20 WORST PREDICTIONS
# ════════════════════════════════════════════════
p("\n" + "=" * 70)
p("11. TOP 20 WORST SPREAD ERRORS")
p("=" * 70)
worst = sorted(rows, key=lambda r: abs(r["spread_error"]), reverse=True)[:20]
for r in worst:
    p(f"  {r['game_date']} {r['home_team']:3s}v{r['away_team']:3s}: "
      f"pred={r['pred_spread']:+6.1f} actual={r['actual_spread']:+6.1f} err={abs(r['spread_error']):5.1f} "
      f"hca={r['home_court_adv']:+5.1f} rate={r['rating_matchup_adj']:+5.1f} "
      f"ff={r['four_factors_adj']:+5.2f} fat={r['fatigue_adj']:+5.1f} ml={r['ml_blend_adj']:+5.2f} "
      f"tune_h={r['home_tune_home_corr']:+5.1f} tune_a={r['away_tune_away_corr']:+5.1f}")

# ════════════════════════════════════════════════
# 12. SPECIFIC FACTOR DEEP-DIVES
# ════════════════════════════════════════════════

# Rating matchup
p("\n" + "=" * 70)
p("12a. RATING MATCHUP DETAIL")
p("=" * 70)
rm = [r["rating_matchup_adj"] for r in rows]
net_diffs = [r["home_net_rating"] - r["away_net_rating"] for r in rows 
             if isinstance(r.get("home_net_rating"), float) and isinstance(r.get("away_net_rating"), float)]
p(f"  Rating adj: mean |adj|={sum(abs(v) for v in rm)/N:.2f}, range=[{min(rm):+.1f}, {max(rm):+.1f}]")
if net_diffs:
    p(f"  Net rating diffs: range=[{min(net_diffs):.1f}, {max(net_diffs):.1f}], avg |diff|={sum(abs(d) for d in net_diffs)/len(net_diffs):.1f}")

# Four factors
p("\n" + "=" * 70)
p("12b. FOUR FACTORS DETAIL")
p("=" * 70)
ff = [r["four_factors_adj"] for r in rows]
p(f"  FF adj: mean |adj|={sum(abs(v) for v in ff)/N:.3f}, range=[{min(ff):+.3f}, {max(ff):+.3f}]")
hfe = [r["home_ff_efg"] for r in rows if isinstance(r.get("home_ff_efg"), float)]
afe = [r["away_ff_efg"] for r in rows if isinstance(r.get("away_ff_efg"), float)]
if hfe:
    efg_diffs = [r["home_ff_efg"] - r["away_ff_efg"] for r in rows 
                 if isinstance(r.get("home_ff_efg"), float) and isinstance(r.get("away_ff_efg"), float)]
    p(f"  eFG diffs: range=[{min(efg_diffs):.3f}, {max(efg_diffs):.3f}], avg |diff|={sum(abs(d) for d in efg_diffs)/len(efg_diffs):.4f}")

# ML Blend
p("\n" + "=" * 70)
p("12c. ML BLEND DETAIL")
p("=" * 70)
ml_vals = [r["ml_blend_adj"] for r in rows if abs(r["ml_blend_adj"]) > 0.01]
p(f"  Active: {len(ml_vals)}/{N}")
if ml_vals:
    p(f"  Blend adj range: [{min(ml_vals):+.2f}, {max(ml_vals):+.2f}], avg |adj|={sum(abs(v) for v in ml_vals)/len(ml_vals):.2f}")
    ml_raw = [r["ml_spread"] for r in rows if abs(r["ml_blend_adj"]) > 0.01]
    p(f"  Raw ML spread range: [{min(ml_raw):.1f}, {max(ml_raw):.1f}]")
    # ML accuracy vs base model
    ml_closer = sum(1 for r in rows if abs(r["ml_blend_adj"]) > 0.01 and
                    abs(r["pred_spread"] - r["actual_spread"]) < abs((r["pred_spread"] - r["ml_blend_adj"]) - r["actual_spread"]))
    ml_worse = sum(1 for r in rows if abs(r["ml_blend_adj"]) > 0.01 and
                    abs(r["pred_spread"] - r["actual_spread"]) > abs((r["pred_spread"] - r["ml_blend_adj"]) - r["actual_spread"]))
    p(f"  ML closer: {ml_closer}  ML worse: {ml_worse}  ({100*ml_closer/max(ml_closer+ml_worse,1):.0f}% helpful)")

# Fatigue
p("\n" + "=" * 70)
p("12d. FATIGUE DETAIL")
p("=" * 70)
fat = [r["fatigue_adj"] for r in rows]
nz_fat = [v for v in fat if abs(v) > 0.01]
p(f"  Active: {len(nz_fat)}/{N} ({100*len(nz_fat)/N:.0f}%)")
if nz_fat:
    p(f"  Range: [{min(nz_fat):+.1f}, {max(nz_fat):+.1f}], avg |adj|={sum(abs(v) for v in nz_fat)/len(nz_fat):.2f}")

# Clutch
p("\n" + "=" * 70)
p("12e. CLUTCH DETAIL")
p("=" * 70)
cl = [r["clutch_adj"] for r in rows]
nz_cl = [v for v in cl if abs(v) > 0.001]
p(f"  Active: {len(nz_cl)}/{N} ({100*len(nz_cl)/N:.0f}%)")
if nz_cl:
    p(f"  Range: [{min(nz_cl):+.3f}, {max(nz_cl):+.3f}]")

# ════════════════════════════════════════════════
# 13. WEIGHTS
# ════════════════════════════════════════════════
p("\n" + "=" * 70)
p("13. WEIGHT CONFIG (from row 0)")
p("=" * 70)
r0 = rows[0]
for k in sorted(k for k in r0 if k.startswith("w_")):
    p(f"  {k}: {r0[k]}")

# ════════════════════════════════════════════════
# 14. KEY ISSUES SUMMARY
# ════════════════════════════════════════════════
p("\n" + "=" * 70)
p("14. KEY ISSUES SUMMARY")
p("=" * 70)
p(f"  [1] Spread conservatism: avg |pred|={sum(abs(s) for s in ps)/N:.1f} vs avg |actual|={sum(abs(s) for s in acs)/N:.1f} (off by {sum(abs(s) for s in acs)/N - sum(abs(s) for s in ps)/N:.1f} pts)")
p(f"  [2] Raw projection vs final: proj_MAE={raw_proj_mae:.2f} final_MAE={sum(se)/N:.2f} => adjustments {'IMPROVE' if sum(se)/N < raw_proj_mae else 'HURT'} by {abs(raw_proj_mae - sum(se)/N):.2f}")
p(f"  [3] Autotune at cap: {at_cap_h} home, {at_cap_a} away")
p(f"  [4] Four factors: mean |adj|={sum(abs(v) for v in ff)/N:.3f}")
p(f"  [5] Fallback projections: {both_fb}/{N} ({100*both_fb/N:.0f}%)")
p(f"  [6] Total bias: {(sum(pt)-sum(at))/N:+.1f}")

# ════════════════════════════════════════════════
# 15. BLOWOUT ANALYSIS (actual spread > 20)
# ════════════════════════════════════════════════
p("\n" + "=" * 70)
p("15. BLOWOUT ANALYSIS (|actual_spread| > 20)")
p("=" * 70)
blowouts = [r for r in rows if abs(r["actual_spread"]) > 20]
non_blow = [r for r in rows if abs(r["actual_spread"]) <= 20]
p(f"  Blowout games: {len(blowouts)}/{N} ({100*len(blowouts)/N:.1f}%)")
if blowouts:
    blow_mae = sum(abs(r["spread_error"]) for r in blowouts) / len(blowouts)
    blow_w = sum(1 for r in blowouts if r["winner_correct"] in (1.0, True, "True"))
    blow_avg_pred = sum(abs(r["pred_spread"]) for r in blowouts) / len(blowouts)
    blow_avg_act = sum(abs(r["actual_spread"]) for r in blowouts) / len(blowouts)
    p(f"  Blowout: winner={100*blow_w/len(blowouts):.0f}%  MAE={blow_mae:.1f}  avg|pred|={blow_avg_pred:.1f}  avg|act|={blow_avg_act:.1f}")
if non_blow:
    nb_mae = sum(abs(r["spread_error"]) for r in non_blow) / len(non_blow)
    nb_w = sum(1 for r in non_blow if r["winner_correct"] in (1.0, True, "True"))
    nb_avg_pred = sum(abs(r["pred_spread"]) for r in non_blow) / len(non_blow)
    nb_avg_act = sum(abs(r["actual_spread"]) for r in non_blow) / len(non_blow)
    p(f"  Non-blow: winner={100*nb_w/len(non_blow):.0f}%  MAE={nb_mae:.1f}  avg|pred|={nb_avg_pred:.1f}  avg|act|={nb_avg_act:.1f}")

# Close games (actual spread < 10)
close = [r for r in rows if abs(r["actual_spread"]) <= 10]
if close:
    cl_mae = sum(abs(r["spread_error"]) for r in close) / len(close)
    cl_w = sum(1 for r in close if r["winner_correct"] in (1.0, True, "True"))
    cl_w5 = sum(1 for r in close if r["spread_within_5"] in (1.0, True, "True"))
    p(f"  Close (<10): {len(close)}g  winner={100*cl_w/len(close):.0f}%  MAE={cl_mae:.1f}  within5={100*cl_w5/len(close):.0f}%")

# ════════════════════════════════════════════════
# 16. HOME vs AWAY PREDICTION BIAS
# ════════════════════════════════════════════════
p("\n" + "=" * 70)
p("16. HOME vs AWAY PREDICTION BIAS")
p("=" * 70)
home_wins_actual = sum(1 for r in rows if r["actual_spread"] > 0)
home_wins_pred = sum(1 for r in rows if r["pred_spread"] > 0)
p(f"  Actually home wins: {home_wins_actual}/{N} ({100*home_wins_actual/N:.1f}%)")
p(f"  Predicted home wins: {home_wins_pred}/{N} ({100*home_wins_pred/N:.1f}%)")
p(f"  Home pred bias: {(home_wins_pred - home_wins_actual):+d} games")

# ════════════════════════════════════════════════
# 17. GAMES WITH FALLBACK PROJECTIONS
# ════════════════════════════════════════════════
p("\n" + "=" * 70)
p("17. FALLBACK PROJECTION GAMES DETAIL")
p("=" * 70)
fb_games = [r for r in rows if abs(r["home_proj_pts"]-112)<0.5 and abs(r["away_proj_pts"]-112)<0.5]
p(f"  Both-side fallback games: {len(fb_games)}")
for r in fb_games[:10]:
    p(f"    {r['game_date']} {r['home_team']:3s}v{r['away_team']:3s}: pred={r['pred_spread']:+.1f} actual={r['actual_spread']:+.1f}")

# One side fallback
one_fb = [r for r in rows if (abs(r["home_proj_pts"]-112)<0.5) != (abs(r["away_proj_pts"]-112)<0.5)]
p(f"  One-side fallback games: {len(one_fb)}")

# ════════════════════════════════════════════════
# 18. SEASON PROGRESSION (rolling 50-game window)
# ════════════════════════════════════════════════
p("\n" + "=" * 70)
p("18. ROLLING ACCURACY (50-game window)")
p("=" * 70)
sorted_rows = sorted(rows, key=lambda r: str(r["game_date"]))
win_size = 50
for i in range(0, len(sorted_rows) - win_size + 1, win_size):
    window = sorted_rows[i:i+win_size]
    w_mae = sum(abs(r["spread_error"]) for r in window) / len(window)
    w_win = sum(1 for r in window if r["winner_correct"] in (1.0, True, "True")) / len(window)
    w_w5 = sum(1 for r in window if r["spread_within_5"] in (1.0, True, "True")) / len(window)
    w_pred_mag = sum(abs(r["pred_spread"]) for r in window) / len(window)
    p(f"  {window[0]['game_date']}-{window[-1]['game_date']}: win={100*w_win:.0f}% MAE={w_mae:.1f} w5={100*w_w5:.0f}% |pred|={w_pred_mag:.1f}")

# Write to file
with open("data/_analysis_output.txt", "w") as f:
    f.write("\n".join(out))
print("\n".join(out))
