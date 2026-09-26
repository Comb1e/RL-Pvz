# Ten-way selection and rejection penalties

Research 0.25.0 compares the ten first-level values

\[
 Q_0(s), Q_{sunflower}(s),\ldots,Q_{repeater}(s),Q_{dig}(s)
\]

for every active state. The controller applies deterministic tie order (wait,
the configured plant order, then dig), then selects a tile. A plant tile mask
contains empty board tiles only; digging considers all 45 tiles. The branch
comparison therefore never changes merely because sun is insufficient, a card is
cooling down, or the board is full. Research lesson restrictions no longer limit
the controller or simulator roster. The pinned simulator rejects the proposed
command and advances one per-tick wait when the
command cannot execute.

The development scale is `progress_weight / value_scale = 0.01 / 300`. Thus the
configured rejected-plant charge is

\[
 p_{plant}=0.001 = 30/30000,
\]

and the empty-dig charge is

\[
 p_{empty\ dig}=0.0003333333333333333 = 10/30000.
\]

For an action result \(r\), the complete transition reward is

\[
 R = R_{terminal}+R_{development}
     -p_{plant}\,1[Place\land\neg accepted]
     -p_{empty\ dig}\,1[Dig\land reason=empty\_tile].
\]

The penalty is independent of net-value accounting: it does not alter assets,
sky income, effective damage, mower expenditure, terminal rewards, or the
discounted-return identity. With gamma one, the return target remains the sum of
the actual transition rewards, including these explicit charges. CPU computes
the indicators from the public action result; CUDA uses the same action index,
accepted flag, and pinned rejection reason (`empty_tile`, reason 4). Independent
controls cover zero-sun and cooldown plants, occupied/out-of-board proposals,
empty digs, one-tick advancement, terminal conservation, and CPU/CUDA equality.

Historical Q records are immutable snapshots of the pre-action vector. Paging
uses an absolute `start` offset and request identifier; selecting a row stores
the row without truncating its page. A response is accepted only when panel,
environment, episode, generation, and request all match. Consequently a newer
network evaluation or a delayed page cannot rewrite a selected decision.

These defaults are experimental costs, not learned calibration. At 100 decisions
per simulated second, repeatedly selecting an invalid plant costs 0.1 reward per
second, and repeated empty digs cost 1/30 reward per second. A 1,200-second cutoff
permits at most 120,000 rejected actions because every rejection advances a tick;
therefore the combined default rejection cost is bounded below by -120 reward
units (the larger plant charge), not by the net-value economic bounds. Successful
instantaneous commands receive neither rejection charge.

The earlier normal-level positive-win/negative-loss reward separation applies
only to outcome plus economic development, or to trajectories with no penalties.
It is **not a guarantee for 0.25.0 total reward**: repeated invalid proposals can
make a win's total reward negative. Outcome reward itself and the old bounds on
assets/net value remain unchanged. The design asks regression to learn rejection
cost from complete returns; it does not guarantee that immediate digging or
other learning failures will disappear.

When no empty tile exists, a plant proposal deterministically uses tile zero,
which the simulator rejects. This explicit fallback avoids an all-masked tile
argmax. The tile exploration coin still fires at its configured rate, but there
is no alternate tile to select. Species exploration still covers all eight
species; evaluation uses neither exploration coin.
