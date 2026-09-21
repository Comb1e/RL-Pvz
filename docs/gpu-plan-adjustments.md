# GPU implementation choices and follow-up work

The learning strategy remains unchanged: shared policy, existing observation and
reward formulas, ordinary action indices, completed-game schedules, PPO networks,
learning rate, discount/GAE, minibatch 256 and four optimization epochs. The GPU
profile uses 128 decisions per parallel game, so changing parallelism changes
the total rollout exactly as requested.

Two implementation choices make the initial port reviewable and preserve exact
game behavior:

1. **Ordered combat within each game.** A warp is assigned to each game and one
   lane executes combat in engine order. Games run in parallel; masks, networks
   and minibatches use further parallelism. Concurrent damage atomics would make
   killing credit and outcomes depend on scheduling. More intra-game parallelism
   should follow profiling and new ordering proofs.
2. **A compact host synchronization remains.** Each decision transfers three
   integers per game for completion, timeout and elapsed ticks. Observations,
   masks, actions, rewards, values, log probabilities, GAE and returns stay on the
   GPU. Completed-game totals transfer as a batch. Reset selection uses current
   CPU curriculum state through a bounded staging queue. This preserves the
   existing episode-boundary semantics without speculatively choosing future
   tasks using stale curriculum weights. A future on-device queue/reset scheduler
   could remove this synchronization, but needs explicit queue-exhaustion and
   curriculum-boundary tests. It is not disguised as a fully asynchronous loop.

The CUDA engine accepts pinned combat rules and declared signed-int64 capacities.
Custom scenarios are resolved with the existing seeded generator. Modified rules
and scripted hybrid placement use Python. Arbitrary mid-game snapshots need
additional projectile headroom beyond the proved bound for newly generated shots.

Benchmark runs use the final 20/40/40 difficulty mixture and at least 4,096
decisions per game so collection includes combat and resets. This is an isolated
throughput workload, not a curriculum or hyperparameter change to real training.
Compilation/warmup and CUDA event instrumentation are reported separately.
Default promotion requires both correctness and a measured whole-pipeline gain;
the performance document records the outcome and limitations.

No imitation, demonstrations for learning, reward edits, larger networks, mixed
precision, extra PPO epochs, or formal research training were introduced. All
development used isolated checkouts/environments. After verification, the research
branch and dependencies were consolidated into the main research checkout's
`.venv`, and the temporary environment was deleted as requested. Existing runs
and checkpoints remain intact. The original game checkout is unchanged; the
optional backend is on a separate committed game feature branch.
