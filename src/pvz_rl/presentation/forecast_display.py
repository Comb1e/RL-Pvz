"""Shared display facts for forecasts. Numbers only; the view draws the rest."""


def fully_correct_text(correct, total):
    """Scene rate where every lane matches field, breach, and probe."""
    if correct is None or total is None:
        return None
    total = int(total)
    if total <= 0:
        return None
    correct = int(round(float(correct)))
    return f"{correct / total:.1%} ({correct}/{total})"


def scores_recorded(record):
    """True when the journal holds action-head scores rather than a zero placeholder."""
    if not record:
        return False
    return any(value not in (None, 0, 0.0) for value in (record.get("q") or ()))


def tile_label(action):
    action = int(action or 0)
    if action <= 0:
        return ""
    tile = (action - 1) % 45
    return f"{chr(65 + tile // 9)}{tile % 9 + 1}"
