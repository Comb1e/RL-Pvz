## Separate tile stages and comparative search — 2026-10-31

- Learning from winning games remains unverified after an attempted correction.
  The user confirms the preceding version's defect: winning play failed to
  reinforce useful choices, and correct actions could be suppressed. The
  correction has not yet undergone actual training, so recovery is unproven.
  Possible causes of the confirmed defect remain hypotheses:
  - **Game outcomes may obscure the causal value of individual choices.** A
    useful decision can inherit an eventual loss, while incidental decisions
    share a win; fitting those outcomes can reward or suppress the wrong choices.
  - **Accurate value fitting may still fail to establish useful preferences.**
    Winning choices need to outrank alternatives, and shared updates may erase
    that ordering even while reducing prediction error.
- Cherry Bomb still chooses occupied plots. The user reports largely learning
  useful bomb placement after about 50,000 environments, but these invalid
  selections keep the stage training rather than reaching completion.
  **The policy may learn attack-value cues without learning occupancy as a
  general constraint.** Strong target-location preferences could dominate plant
  presence, leaving a small but persistent class of invalid choices despite
  broad tactical improvement. This explanation is speculative.
- Performance gains beyond Cherry Bomb remain untested. Its reported near-100x
  improvement suggests substantial optimization opportunities elsewhere, but
  neither transferable speedups nor equivalent bottlenecks are established.
  **Other stages may share preparation and coordination costs, or spend most
  of their time on fundamentally different work.** Cherry Bomb's specialized
  workload may therefore expose either a common inefficiency or a stage-specific
  advantage; this is a hypothesis, not a measured finding about other stages.
