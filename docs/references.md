# Research references

Papers and projects that informed design or research directions, listed once per
source. Entries distinguish adopted ideas from algorithms that remain comparison
references. Historical entries preserve attribution for removed designs.
Routine library/API documentation, implementation details and version history are
outside this index.

## Design influences

- Metz et al., [Discrete Sequential Prediction of Continuous Actions for Deep RL](https://arxiv.org/html/1705.05035v3):
  conditional action decomposition and cross-stage value consistency informed the
  species/tile model. Its Bellman algorithm is not adopted; consistency alone
  does not independently ground either head's values.
- van Hasselt et al., [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/html/1509.06461v3):
  online selection and separate EMA evaluation inform species consistency targets.
  Correlated estimators can share errors; agreement does not establish calibration.
- Zambaldi et al., [Relational Deep Reinforcement Learning](https://arxiv.org/abs/1806.01830):
  entity/spatial attention informed reasoning over the current public board.
- Lee et al., [Set Transformer](https://arxiv.org/abs/1810.00825) and
  [project](https://github.com/juho-lee/set_transformer): permutation-invariant
  entity attention and query readouts informed the independent encoders.
- Vinyals et al., [AlphaStar](https://www.nature.com/articles/s41586-019-1724-z),
  its [unit encoder](https://github.com/google-deepmind/alphastar/blob/700b1e74364ed5dfc66f6cd2574c5ffac2fa474e/alphastar/architectures/components/units.py)
  and [mini-AlphaStar §3.2](https://arxiv.org/html/2104.06890v2#S3.SS2): entity
  embeddings, masks and structured action arguments informed encoding; recurrence
  and league training are not used.
- Hester et al., [Deep Q-learning from Demonstrations](https://arxiv.org/html/1704.03732v4):
  supervised action preferences alongside value regression informed demo/annotation
  initialization. Finite legality margins and rankings from verified comparisons
  are research proposals; a winning trajectory does not make every action expert.
- [AlphaStar Unplugged](https://arxiv.org/abs/2308.03526): offline return learning
  motivated demonstration initialization, without reproducing its algorithm or data.
- Dabney et al., [Temporally-Extended epsilon-Greedy Exploration](https://arxiv.org/abs/2006.01782):
  temporal persistence informed species commitments that reduce dithering.
- Bacon et al., [The Option-Critic Architecture](https://arxiv.org/abs/1609.05140)
  and [project](https://github.com/jeanharb/option_critic): explicit termination
  informed readiness-based commitments; options and termination are not learned.
- Cour, Sapp and Taskar, [Learning from Partial Labels](https://www.jmlr.org/papers/v12/cour11a.html):
  candidate-label supervision informed accepting any annotated feasible tile.
  Multiple tiles can be correct here; the local set-ranking loss is an adaptation.
- Sutton, [Learning to Predict by the Methods of Temporal Differences](https://www.cs.utexas.edu/~shivaram/readings/b2hd-Sutton1988.html):
  credit-assignment background for chronological returns and delayed effects.
  The implementation uses simulator provenance rather than learned traces.
- Ng, Harada and Russell, [Policy Invariance Under Reward Transformations](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf):
  distinguishes potential shaping from intentional objective changes. With gamma
  one and zero terminal potentials, its sum is constant for a fixed initial state;
  this alone supplies no new ranking evidence to complete-return regression.
- van Hasselt et al., [Learning values across many orders of magnitude](https://arxiv.org/abs/1602.07714):
  scale sensitivity motivated separating raw ledger values from fitted rewards;
  PopArt is not implemented.
- Cobbe et al., [Leveraging Procedural Generation to Benchmark Reinforcement Learning](https://arxiv.org/abs/1912.01588)
  and [Procgen](https://github.com/openai/procgen): procedural diversity and held-out
  scenarios informed independent training/validation splits and reachable sampling.
- Agarwal et al., [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264)
  and [rliable](https://github.com/google-research/rliable/tree/3ccd9f4dea577a04d3d2b557f259aac08badbd81):
  independent runs, uncertainty and equal-budget comparisons informed evaluation;
  the crossed paired bootstrap is local.
- [WarpDrive: Fast End-to-End Deep Multi-Agent Reinforcement Learning on a GPU](https://jmlr.org/papers/v23/22-0185.html)
  and [project](https://github.com/salesforce/warp-drive/blob/master/README.md):
  device-resident simulation and reduced host/device round trips informed batching.
  No code or published throughput result is adopted.

## Research comparisons

- Ben-Baruch et al., [Asymmetric Loss For Multi-Label Classification](https://arxiv.org/html/2009.14119v4):
  dense positive/negative supervision and reduced easy-negative weight motivate a
  Cherry Bomb classification experiment. This remains a proposal, not the adopted
  top-1 hinge; simulator hit labels are complete, so missing-label filtering is unnecessary.
- Shaw et al., [Self-Attention with Relative Position Representations](https://aclanthology.org/N18-2074/):
  relation-aware attention motivates explicit tile/entity spatial offsets as an
  alternative to learning blast geometry from absolute coordinates. Language-task
  results do not establish improvement on this board.
- Dong, Cordonnier and Loukas, [Attention is not all you need: pure attention loses rank doubly exponentially with depth](https://proceedings.mlr.press/v139/dong21a.html):
  token-uniformity analysis motivates shallow readouts and spatial skip-path controls
  for nearly identical tile features. The pure-attention theorem does not establish
  collapse in the current residual Transformer with MLPs.
- Kaushik, Hovy and Lipton, [Learning the Difference that Makes a Difference with Counterfactually-Augmented Data](https://arxiv.org/abs/1909.12434):
  minimally changed examples motivate paired boards with moved zombies and
  simulator-recomputed labels to challenge location shortcuts. This is an untested
  adaptation from text classification, not counterfactual game-return evaluation.
- Anthony, Tian and Barber, [Thinking Fast and Slow with Deep Learning and Tree Search](https://arxiv.org/html/1705.08439v1):
  expert iteration informs the bounded simulator-search teacher whose
  action comparisons train the existing networks. Search and distillation gains
  in Hex do not establish improvement or computational feasibility here.
- [DeepMind mctx search policies](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/policies.py):
  inspected Gumbel root sampling, sequential halving and explicit simulation/depth
  budgets as allocation references. Neither JAX nor learned dynamics is used;
  the linked Gumbel paper's PDF was blocked, so no theorem is transferred.
- Meulemans et al., [Would I have gotten that reward?](https://proceedings.neurips.cc/paper_files/paper/2023/file/d8bd445c2abe1343cce0e14b361b2fb3-Paper-Conference.pdf):
  COCOA's reward-outcome contributions inform distinguishing strategic value from
  reproducing a particular later board or action sequence. Its learned contribution
  estimator is not adopted by the simulator-search proposal.
- Triantafyllou et al., [Counterfactual Effect Decomposition in Multi-Agent Sequential Decision Making](https://arxiv.org/html/2410.12539v1):
  effects through later actions and state transitions inform preserving income and
  occupancy consequences. This is conceptual guidance for a single-agent planner,
  without implementing the paper's mediation or Shapley estimators.
- Arjona-Medina et al., [RUDDER: Return Decomposition for Delayed Rewards](https://papers.nips.cc/paper/2019/file/16105fb9cc614fc29e1bda00dab60d41-Paper.pdf):
  return equivalence informs checking reward redistribution against the original
  objective. Redistribution alone neither supplies missing counterfactual evidence
  nor repairs an undesirable reward ordering; no extra sequence model is proposed.
- Silver and Veness, [Monte-Carlo Planning in Large POMDPs](https://papers.nips.cc/paper_files/paper/2010/file/edfbe1afcf9246bb0d40eb4d8027d90f-Paper.pdf):
  sampled hidden states and observation-history search inform evaluating plausible
  futures without giving action selection private schedule information. POMCP's
  belief/history machinery and convergence guarantees are not adopted wholesale.
- Cowling, Powley and Whitehouse, [Information Set Monte Carlo Tree Search](https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf):
  strategy fusion motivates a teacher control: indistinguishable public inputs
  cannot justify different actions using hidden futures. Full information-set
  search is a comparison reference, not a second training framework.
- Hatch and Boots, [The Value of Planning for Infinite-Horizon Model Predictive Control](https://homes.cs.washington.edu/~bboots/files/ValueOfPlanning.pdf):
  horizon myopia and terminal-value accuracy inform testing search beyond Sunflower
  payback and defense delays. Robot-navigation results do not validate this game's
  current value estimates or a shortened rollout horizon.
- Sunberg and Kochenderfer, [Online Algorithms for POMDPs with Continuous State, Action, and Observation Spaces](https://cdn.aaai.org/ojs/13882/13882-40-17400-1-2-20201228.pdf):
  observation-tree particle collapse motivates public-state sharing and fresh-world
  confirmation controls. The teacher uses an approximate public-board prior,
  not POMCPOW's weighted belief filter or its convergence claims.
- Harutyunyan et al., [Hindsight Credit Assignment](https://papers.neurips.cc/paper/9413-hindsight-credit-assignment.pdf):
  relevance-weighted advantages motivated same-board action comparisons that
  separate useful decisions from continuation noise. Paired simulator rollouts are
  a local proposal and estimate the chosen continuation policy, not optimal values.
- Liu et al., [Towards a practical measure of interference for reinforcement learning](https://arxiv.org/html/2007.03807v1):
  before/after-update policy returns motivated measuring actual behavioral damage;
  average losses can conceal damaging updates. Its TD-error proxy is not assumed
  valid for complete-return fitting.
- Yu et al., [Gradient Surgery for Multi-Task Learning](https://arxiv.org/html/2001.06782v3)
  and [PCGrad](https://github.com/tianheyu927/PCGrad): per-loss gradient projection
  is a candidate if harmful interference is demonstrated. Negative alignment alone
  is insufficient evidence; grouping this agent's losses is an unverified adaptation.
- Skalse et al., [Defining and Characterizing Reward Hacking](https://ar5iv.labs.arxiv.org/html/2209.13085):
  policy-order comparisons motivated checking shaped returns against victory rates.
  Its restrictive assumptions do not establish reward hacking in this agent.
- Ecoffet et al., [First return, then explore](https://arxiv.org/html/2004.12919v5)
  and [Go-Explore](https://github.com/uber-research/go-explore): archive/return/explore
  separation motivated coherent exploration from reachable late-game states.
- Lan et al., [Maxmin Q-learning: Controlling the Estimation Bias of Q-learning](https://arxiv.org/html/2002.06487v2):
  minima across estimators motivated conservative target comparisons, with possible
  underestimation; its tabular results do not guarantee Transformer accuracy.
- Anschel et al., [Averaged-DQN](https://arxiv.org/html/1611.01929v4): averaging past
  predictions motivated variance-reduction comparisons. Averaging nonlinear model
  weights is not equivalent to averaging predictions and cannot remove shared bias.
- Feinberg et al., [Model-Based Value Estimation for Efficient Model-Free Reinforcement Learning](https://arxiv.org/html/1803.00101v1)
  and [project](https://github.com/vlad17/mve/blob/19835bba87a2e5abc9bca653d011bf2eed68dd62/README.md):
  bounded rewards plus a tail estimate informed earlier probes and later calibration
  proposals. Learned dynamics are not used; gamma one does not discount tail error.
- [Explorer](https://github.com/qlan3/Explorer): Double, Maxmin and Averaged DQN
  implementations supplied comparison references; no code or dependency is adopted.
- Foerster et al., [Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/html/1705.08926v3)
  and [PyMARL's COMA learner](https://github.com/oxwhirl/pymarl/blob/c971afdceb34635d31b778021b0ef90d7af51e86/src/learners/coma_learner.py):
  holding one action component fixed informed placement-credit comparisons.
  Species and tiles are not independent agents; advantages cannot replace absolute
  value targets without a matching objective.
- Tavakoli et al., [Action Branching Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1711.08946)
  and [project](https://github.com/atavakol/action-branching-agents/tree/1e026011505779e0ad30060ab3422f97eff08300):
  contrasting factorized actions highlighted the need to retain species/tile interactions.
- Settles, [Active Learning Literature Survey](https://burrsettles.com/pub/settles.activelearning.pdf):
  pool-based selection informed possible annotation-budget reduction; active
  selection remains unimplemented.
- Rabe and Staats, [Self-attention Does Not Need O(n²) Memory](https://arxiv.org/abs/2112.05682):
  abstract-only background distinguished attention arithmetic from storage cost;
  no published performance ratios are transferred to this project.

## Simulator projects

- [pvz_game inspected source](https://github.com/Comb1e/pvz-cuda-work/tree/1424b4d802e783a36772091c49d96aa7c6036d7a):
  public observations, legality, events and differential controls informed adapter
  contracts and CPU/CUDA checks. The [engine lock](../src/pvz_rl/data/engine-lock.json)
  is authoritative for the current dependency pin.
- [Patoke reconstruction](https://github.com/Patoke/re-plants-vs-zombies/tree/c4692036c5e11d227c8fb7c593b734dac96da028)
  and [Bamcane animation reference](https://github.com/Bamcane/re-plants-vs-zombies/tree/0f6bbd39302acf69484ba8b3e071724e35cfba17/pak/reanim):
  collision geometry, cadence and explosion/contact behavior informed the engine
  mechanics audit; reconstruction does not prove original-binary equivalence.

## Historical influences

- Campos et al., [Skip RNN: Learning to Skip State Updates in Recurrent Neural Networks](https://arxiv.org/abs/1708.06834)
  and [project](https://imatge-upc.github.io/skiprnn-2017-telecombcn/): informed the
  former selective event-memory design. Current policies are stateless.
- Hausknecht and Stone, [Deep Recurrent Q-Learning](https://arxiv.org/abs/1507.06527):
  abstract-only motivation for the removed recurrent policy.
- [Sample Factory](https://arxiv.org/abs/2006.11751) and
  [project](https://github.com/alex-petrenko/sample-factory): independent runnable
  work informed the former background counterfactual scheduler.
- [EnvPool](https://arxiv.org/abs/2206.10558) and
  [project](https://github.com/sail-sg/envpool): batched/asynchronous environments
  informed independent lanes in that removed scheduler.
