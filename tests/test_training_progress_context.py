from types import SimpleNamespace

from pvz_rl.config import load_config
from pvz_rl.curriculum import CurriculumState
from pvz_rl.training import ResearchCallback


def test_progress_identifies_mower_free_lessons_and_does_not_round_rare_kills(tmp_path, capsys):
    cfg = load_config()
    callback = ResearchCallback(cfg, "masked", 101, tmp_path)
    callback.model = SimpleNamespace(num_timesteps=1000, training_games=100)
    callback.curriculum = CurriculumState()
    try:
        episode = dict(
            family="placement",
            level="easy",
            decisions=100,
            win=0,
            simulated_seconds=50,
            invalid_actions=0,
            plant_kills=0,
            mower_kills=0,
            attacker_purchases=3,
            early_voluntary_digs=3,
            **{"return": -2},
        )
        callback.recent.extend([episode.copy() for _ in range(100)])
        callback.recent[-1]["plant_kills"] = 1
        callback.log_progress(force=True)
        output = capsys.readouterr().out
        assert (
            "stage placement; exploration 0.000%; tasks placement=100; mowers disabled in these lessons"
            in output
        )
        assert "plant/mower kills per game 0.01/0.00" in output
        assert "early digs/game 3.00" in output
        assert "plant+dig/game" not in output  # Missing archived metrics stay missing.
        for row in callback.recent:
            row["agent_actions"] = 6
        callback.recent[-1].update(family="preset", level="easy", mower_kills=2)
        callback.log_progress(force=True)
        output = capsys.readouterr().out
        assert "mowers disabled" not in output
        assert "plant/mower kills per game 0.01/0.02" in output
        assert "plant+dig/game 6.00" in output
        assert "transitions/s" in output
        callback.recent.clear()
        callback.log_progress(force=True)
        output = capsys.readouterr().out
        assert "pending" in output and "kills per game" not in output
    finally:
        callback.close()
