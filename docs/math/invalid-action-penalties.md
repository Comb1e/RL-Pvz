# Ten-way selection and rejection penalties

The controller compares the ten first-level values

\[
 Q_0(s), Q_{sunflower}(s),\ldots,Q_{repeater}(s),Q_{dig}(s)
\]

for every active state. The controller applies deterministic tie order (wait,
the configured plant order, then dig), then selects a tile. A plant tile mask
contains empty board tiles only; digging considers all 45 tiles. The branch
comparison therefore never changes merely because sun is insufficient or a card
is cooling down. The selected plant proposal is kept through CPU and CUDA
validation. On a full board it still gets the deterministic tile-zero proposal,
so the pinned simulator reports occupancy rejection rather than a lower-Q
fallback and advances one per-tick wait when the command cannot execute.

Rejection prices have one shared definition in `train.toml`, used by autonomous
training, demonstration reward reconstruction and reporting:

\[
 p_{plant}=0.00001,
\]

and

\[
 p_{empty\ dig}=0.000003.
\]

For an action result, these signed components are

\[
 R_{invalid\ plant}=-p_{plant}\,1[Place\land\neg accepted],\qquad
 R_{empty\ dig}=-p_{empty\ dig}\,1[Dig\land reason=empty\_tile].
\]

The penalty is independent of net-value accounting: it does not alter assets,
sky income, effective damage, mower expenditure, terminal rewards, or the
discounted-return identity. With gamma one, fitting sums the six signed components
defined in [reward accounting](training-objective.md), multiplying only development
by ten. Rejection costs are never multiplied. CPU computes
the indicators from the public action result; CUDA uses the same action index,
accepted flag, and pinned rejection reason (`empty_tile`, reason 4). Independent
controls cover zero-sun and cooldown plants, occupied/full-board proposals, empty
digs, one-tick advancement, terminal conservation, and CPU/CUDA equality.
The trajectory stores the selected proposal so its complete-return target learns
the penalty. Recurrent input stores the resolved execution action separately:
the proposal after acceptance, or wait (`0`) after rejection, together with
`accepted` and `ticks_advanced`. Rejection therefore produces `(0, 0, 1)`
for the next recurrent input. The cap does not change branch selection or penalties.

These defaults are experimental costs, not learned calibration. At 100 decisions
per simulated second, repeatedly selecting an invalid plant costs 0.001 reward per
second, and repeated empty digs cost 0.0003 per second. A 1,200-second cutoff
permits at most 120,000 rejected actions because every rejection advances a tick;
therefore the combined default rejection cost is bounded below by -1.2 reward
units (the larger plant charge), independently of the economic bounds. Successful
instantaneous commands receive neither rejection charge.

The normal-level positive-win/negative-loss reward separation applies
only to outcome plus economic development, or to trajectories with no penalties.
It is **not a guarantee for total reward**: repeated invalid proposals can
make a win's total reward negative. Outcome reward itself and the bounds on
assets/net value remain unchanged. The design asks regression to learn rejection
cost from complete returns; it does not guarantee that immediate digging or
other learning failures will disappear.
