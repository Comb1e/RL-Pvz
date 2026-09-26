# Mechanics, economics and measurement scope in 0.24.0

The engine's checked [gait and geometry derivation](https://github.com/Comb1e/pvz-cuda-work/blob/fdab989fe20ad3415c2b8975fccdeec834d0196e/docs/math/gait-and-geometry.md)
defines the source-pixel convention, rational rate calculation, phase endpoint,
fixed-point rounding, collision boundaries and remaining approximations. The
independent CPU fraction integrator verifies 5,000 ticks for every gait with and
without chilling; CUDA controls compare state, events, RNG and snapshot replay.

One tile is 80 source pixels. Public zombie x is the body's left edge; the
ordinary body is 42 pixels wide. With N frames and source velocity v, continuous
cycle mean speed is `47 v N / (80 (N−1))` tiles/s. This is only a mean: zero and
negative ground intervals remain observable as nonuniform motion. Ordinary
walkers range .1380625–.1920869565 tiles/s; flag speed parameter is fixed, with
mean .2701222826. Chilling halves animation rate. Jump translation has a separate
clock and a 150-pixel completion shift. Private gait state does not add inputs.

## Reward accounting after the mechanics audit

The objective and units in [training objective](training-objective.md) are unchanged.
Damage is effective plant/projectile body plus armor damage; overkill, mower damage
and autonomous decay do not add plant-damage value. Head loss can finish a game
while uncredited residual HP remains. One effective HP is worth 50/270 net sun;
one net sun contributes 1/30,000 reward. Thus 20 HP contributes
`(20*50/270)/30000 = 0.0001234567901`, not the old 200-HP conversion.

For the pinned normal rosters, 50 initial sun and a 120,000-tick cutoff:

| Conservative bound | Calculation | Value |
|---|---|---:|
| Final assets | 9990 + 45×200 | 18,990 |
| Actual sky income | earliest first drop 425; next gap min(950,425+10n) | ≤3,525 |
| Available body+armor, easy/standard/hard | sum configured roster HP | 4,050 / 17,240 / 38,130 |
| Effective damage value, hard | 38130×50/270 | ≤7,061.111111 |
| Cumulative net lower | −50−3525−5×200 | −4,575 |
| Cumulative net upper | 18990−50+38130×50/270 | +26,001.111111 |
| Naturally completed win reward | 1−4575/30000 | ≥+0.8475 |
| Naturally completed loss reward | −2+26001.111111/30000 | ≤−1.1332962963 |

The independent maintained bound test counts minimum sky intervals and configured
rosters without invoking the reward implementation. These outcome-separation
bounds exclude arbitrary custom configurations, initial boards and incomplete
episodes. Gait changes do not enlarge health pools or asset caps, so they do not
invalidate these conservative bounds. They do change realized timing and damage.

## Saving feasibility and counterexamples

Saving is unchanged: 100 sun, all eight plants, no natural income or mowers,
three lanes and waves at 75/87/99 seconds. The cash/production controls remain in
[saving feasibility](saving-and-actions.md). The public-only five-flower witness
and wait/no-flower/dig/four-mine failures are checked for all ten lane combinations.
CPU/CUDA agreement includes complete states, public observations and reward.

Do not reuse the old one-tile mine argument. A mine has a 60-pixel circular radius,
the victim has a 42-pixel body, and independently sampled gaits change spacing.
Mine triggering and blast eligibility are separate tests, including pole/airborne
exclusions and exact tangencies. The maintained controls establish these specific
witnesses and counterexamples, not an exhaustive proof for every seed or mine
placement. The lesson was not retuned to hide a regression.

## Cohort and diagnostic accounting

The unchanged [sequential-Q loss](sequential-q-control.md) is fit after 128 complete
games. No network update happens during collection; finished workers pause. Larger
cohorts can improve execution throughput while repeating behavior longer and
delaying feedback. The old environment-count measurements used three warmed,
viewer-disabled 10-second cutoffs, not normal complete games.

History records the actual pre-action ten-vector and legal branches with the
decision sequence, tick, executed/greedy action, exploration coins and outcome.
Sequence numbers count waits too; history rows omit waits, so gaps are intentional.
No historical score is recomputed after fitting. Journals are output-only and do
not enter Q targets, memory tokens, observations or compatibility signatures.
