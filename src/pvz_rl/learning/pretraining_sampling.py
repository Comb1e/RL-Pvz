"""Model-free private-pretraining samples shared by legacy and device execution."""

import random

import numpy as np
from pvz_game import LevelSpec, Spawn

from pvz_rl.envs.scratch_game import EngineSnapshot, scratch_game, set_initial_mowers


def prepare_roots(cfg, rules, seed, origin, count):
    snapshots = []
    game = scratch_game(rules)
    for index in range(count):
        rng = random.Random((int(seed or 0) << 32) ^ (origin + index) ^ 0xF0EC457)
        zombies = rng.randint(1, 8)
        level = LevelSpec(
            "forecast-pretraining",
            tuple(
                Spawn(
                    3501,
                    rng.choice(cfg["environment"]["zombies"]),
                    rng.randrange(5),
                    x=rng.randrange(9 * rules.game["units_per_tile"]),
                )
                for _ in range(zombies)
            ),
            initial_sun=500,
            mowers=False,
        )
        game.reset(level, rng.randrange(2**31))
        set_initial_mowers(game, [row for row in range(5) if rng.random() < 0.5])
        snapshots.append(EngineSnapshot.capture(game))
    return snapshots


def play_commands(legal, seed, origin, tick, count):
    commands = np.zeros(len(legal), np.int64)
    for slot in range(count):
        choices = np.flatnonzero(legal[slot])
        rng = np.random.default_rng([int(seed or 0), origin + slot, tick, 17])
        if (origin + slot) % 2 == 0:
            commands[slot] = int(rng.choice(choices))
        else:
            for branch in (1, 2, 8, 6, 3, 5, 7, 4):
                candidates = choices[(choices > 0) & ((choices - 1) // 45 + 1 == branch)]
                if len(candidates):
                    commands[slot] = candidates[0]
                    break
    return commands


def sample_actions(legal, seed, origin, count):
    actions, valid, jobs = np.zeros((count, 5), np.int64), np.zeros((count, 5), bool), []
    for root in range(count):
        rng = np.random.default_rng([int(seed or 0), origin + root, 71])
        candidates = np.flatnonzero(legal[root])[1:]
        selected = rng.choice(candidates, min(5, len(candidates)), replace=False)
        actions[root, : len(selected)] = selected
        valid[root, : len(selected)] = True
        jobs.extend((root, token) for token in range(len(selected)))
    return actions, valid, jobs
