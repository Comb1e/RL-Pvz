from types import SimpleNamespace

from pvz_rl.config import load_config
from pvz_rl.learning.curriculum import CurriculumState
from pvz_rl.learning.training import ResearchCallback


def test_progress_has_aligned_recent_and_run_metrics_and_unlimited_has_no_eta(tmp_path, capsys):
    cfg = load_config()
    cfg["curriculum"]["run_stage"] = "saving"
    cfg["training"]["until_stage_complete"] = True
    callback = ResearchCallback(cfg, "masked", 101, tmp_path)
    callback.model = SimpleNamespace(
        num_timesteps=1000, training_games=100, phase="collect", exploration_phase="formal"
    )
    callback.curriculum = CurriculumState()
    try:
        callback.log_progress(force=True)
        text = capsys.readouterr().out
        assert "Stage       saving | collect" in text
        assert "until stage mastery" in text and "ETA" not in text
        assert "Hardware    GPU n/a" in text
        assert "Run average" in text and "Recent play" in text
        assert "next mastery probe 2,000 games" in text
        assert "Reward      n/a | discounted return n/a" in text
        assert "Net value   n/a" in text
        callback.model.phase = "fit"
        callback.recent.append(
            dict(
                family="saving",
                level="easy",
                decisions=100,
                win=1,
                simulated_seconds=136.52,
                invalid_actions=0,
                attacker_purchases=3,
                early_voluntary_digs=1,
                plant_usage={"sunflower": 5, "peashooter": 3},
                discounted_return=1.1,
                cumulative_net_value=3000,
                terminal=1,
                development=0.1,
                **{"return": 1.1},
            )
        )
        callback.log_progress(force=True)
        text = capsys.readouterr().out
        assert "| fit" in text and "early digs/plant 12.50%" in text
        assert "attackers/game 3.00" in text and "discounted return +1.10000" in text
        assert "Reward      +1.10000" in text and "Net value   +3000.00" in text
    finally:
        callback.close()
