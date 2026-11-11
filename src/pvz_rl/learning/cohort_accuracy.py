"""Cohort-fitting accuracy. Validation panels are a separate measurement."""

from pvz_rl.learning.forecast_labels import scene_class_matches
from pvz_rl.learning.placement_labels import positive_set_mask
from pvz_rl.policy.sequential_q import greedy_choice, tile_candidates


def cohort_perfect(correct, total, expected):
    """True when every expected board was scored and every one matched."""
    if (
        type(correct) is not int
        or type(total) is not int
        or type(expected) is not int
    ):
        return False
    return expected > 0 and correct == total == expected


def before_fit_rate(correct, scored):
    """Fraction from the check that runs before any fitting pass."""
    if (
        type(correct) is not int
        or type(scored) is not int
        or scored < 1
        or correct < 0
        or correct > scored
    ):
        return None
    return correct / scored


def before_fit_counts(saved_correct, saved_scored, epoch, phase, check_correct, check_scored):
    """Restore the before-fit pair. A later check is not a substitute."""
    if before_fit_rate(saved_correct, saved_scored) is not None:
        return saved_correct, saved_scored
    if (
        type(epoch) is int
        and epoch == 0
        and phase in ("pass", "done")
        and before_fit_rate(check_correct, check_scored) is not None
    ):
        return check_correct, check_scored
    return None, None


def scene_hits(logits, targets):
    """Scenes whose field, breach, and probe classes all match."""
    matched = scene_class_matches(logits, targets)
    return matched.all(-1).all(-1).sum()


def _same_count(count, expected):
    return type(count) is int and type(expected) is int and expected > 0 and count == expected


def _strict_rate(numer, denom, threshold):
    if type(numer) is not int or type(denom) is not int or denom < 1:
        return False
    if isinstance(threshold, bool) or type(threshold) not in (int, float):
        return False
    return numer / denom > threshold


def _tasks_clear(task_wins, tasks):
    if not tasks:
        return False
    recorded = task_wins or {}
    for task in tasks:
        pair = recorded.get(task)
        if not isinstance(pair, tuple) or len(pair) != 2:
            return False
        wins, games = pair
        if type(wins) is not int or type(games) is not int or games < 1 or wins != games:
            return False
    return True


def cohort_meets_stage_bar(
    stage,
    *,
    correct=None,
    scored=None,
    expected=None,
    wins=None,
    games=None,
    task_wins=None,
    tasks=None,
    threshold=0.95,
):
    """Old validation bar on one collected cohort.

    ``True`` meets the bar, ``False`` is a finished miss, and ``None`` is a
    short cohort that does not move the streak.
    """
    from pvz_rl.learning.curriculum import COHORT_PERFECT_STAGES, STAGES

    if stage in COHORT_PERFECT_STAGES:
        if not _same_count(scored, expected):
            return None
        if stage == "cherry_bomb":
            return _strict_rate(correct, scored, threshold)
        return cohort_perfect(correct, scored, expected)
    if stage == "peashooter":
        if not _same_count(games, expected):
            return None
        return _strict_rate(wins, games, threshold)
    if stage in STAGES:
        if not _same_count(games, expected):
            return None
        return _tasks_clear(task_wins, tasks)
    return False


def legal_tile_hits(values, occupied, branches, feasible):
    """Boards whose greedy legal tile lies in that board's successful set."""
    if values.ndim != 2 or values.shape[0] != len(feasible):
        raise ValueError("Tile scores must match the labeled boards")
    candidates = tile_candidates(occupied, branches)
    chosen = greedy_choice(values, candidates)
    positive = positive_set_mask(feasible, values.shape[1]).to(values.device)
    return positive.gather(1, chosen[:, None]).flatten().sum()
