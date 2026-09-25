"""Resolve the two historical module names stored in compatible checkpoints."""

import sys


def register_checkpoint_imports():
    """Preserve pickled class identity without duplicate modules or conversion.

    0.19.0 stored policy/features and rollout-buffer classes under flat module
    names. New checkpoints record the canonical package names. Protocol validation
    still decides whether a checkpoint may load; aliases never bypass it.
    """
    from pvz_rl.learning import cuda_buffer
    from pvz_rl.policy import spatial_policy

    sys.modules.setdefault("pvz_rl.spatial_policy", spatial_policy)
    sys.modules.setdefault("pvz_rl.cuda_buffer", cuda_buffer)
