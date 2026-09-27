# Sequential complete-return Q control

The implemented objective in research 0.25.0 uses complete episodes, gamma one,
and one shared encoder. A decision assembles a branch b (wait, eight species,
dig) and, except for wait, a tile t. Only the assembled command advances the
simulator or enters public history. The 1,200-second cutoff is a labelled failure.

## Targets and interpretation

For transition i in a completed game, G_i = sum_{j=i}^{T-1} r_j. Backward
accumulation uses float64; stored fitting targets use float32. No value bootstrap,
replay, target network or intermediate reward is used. Gamma one makes the first
target equal to episode reward, including outcome, net-value development and
[rejection penalties](invalid-action-penalties.md).

Q1(s,b) estimates the remaining return after selecting b and continuing with
the collecting tile selector. Q2(s,b,t) estimates it after also selecting t.
For a fixed continuation policy, Q1(s,b) = sum_t p(t|s,b) Q2(s,b,t).
This is an expectation during exploration, not necessarily max_t Q2(s,b,t).
For example, tile values [-2, 1, 0.25] and epsilon 0.051316701949 give an
epsilon-greedy expected value of 0.935854122563, below the greedy value 1.
Approximate shared-target regression does not enforce exact equality, ensure
optimal choices, or supply information about unvisited alternatives.

Per-transition loss is (Q1-G)^2 for wait and
[(Q1-G)^2 + (Q2-G)^2]/2 otherwise. The predictions are never added.
The selected-head derivatives are 2(Q1-G) for wait, and Q1-G and Q2-G for
non-wait, before group and minibatch scaling. Unselected output entries receive
no direct target gradient; shared encoder parameters can still change them.

For populated groups g in {wait, plant, dig}, counts n_g, total N, and K
populated groups, w_i = N/(K n_g). The full objective is
L = (1/N) sum_i w_i loss_i = (1/K) sum_g mean_{i in g} loss_i.
Each minibatch sum is divided by min(batch_size,N), including the short final
batch. This preserves relative sample weights across an epoch; sequential Adam
steps are not equivalent to one full-cohort optimizer step.
Control: five waiting errors of 4, two planting errors of 1 and one digging
error of 0 give L=(4+1+0)/3=5/3, irrespective of group frequencies.

## Exploration

The branch controller always takes the maximum of all ten Q1 outputs, breaking
ties by wait, configured species order, then dig. The branch is never explored,
so a high-value plant remains the submitted proposal even when the simulator
will reject it for affordability or cooldown.

After the branch wins, a non-wait branch chooses a tile with one exploration
coin of probability e, where e is the cohort's tile budget. If the greedy tile
is t* and the candidate set has n tiles, then

\[
p(t\mid s,b)=(1-e)1[t=t^*]+e/n.
\]

Plant candidates are empty tiles; dig candidates are all 45 tiles. A full board
uses the explicit rejected tile-zero proposal described in the penalty
derivation, so there is no alternate tile to sample. Coin firing can still
select the greedy tile and is logged separately from whether the command
changed. At e=0 selection is fully greedy; at e=1 the selected branch's tile is
uniform. Exploration cannot force a different branch when waiting has the
greatest value.

For stage games g, e(g)=e_start (e_floor/e_start)^min(1,g/3000), with
defaults 0.1 and 0.001. Resolve once before a cohort and save that resolved state
on interruption. Evaluation sets e=0 without changing the saved training budget.

## Scope and controls

Independent controls in the maintained math/policy suites enumerate probabilities,
calculate returns and weighted losses, and compare gradients and Adam updates.
Negative waiting targets can make initially zero planting values preferable;
this is a controlled example, not a guarantee of coverage or successful learning.
Greedy control can remain wrong when values are inaccurate or untried.

Metz et al.'s sequential-Q method motivates assembling actions in stages; its
bootstrapped off-policy objective is different from this Monte Carlo adaptation.
See the inspected-method evidence and SB3 argmax source in ../references.md.


The default cohort is now 128 complete games. All 128 environments use the same
frozen network until each has finished; an early finisher waits. Increasing the
cohort changes the number/diversity of games between fits, not this target or loss
formula. It can reduce fitting frequency per completed game and increase stale
behavior within a cohort. Short-cutoff throughput does not quantify learning gains.
