"""Explicit, restartable orchestration for the single-policy research protocol.

This module never starts automatically. Every attempt uses a fresh directory, and
the journal keeps previous attempts and their evidence intact.
"""

import copy
import json
from pathlib import Path

from pvz_rl.config import digest, research_config, seed_values
from pvz_rl.evaluation.runner import BASELINES, evaluate
from pvz_rl.learning.training import load_policy, train
from pvz_rl.learning.training_requirements import TRAINING_CONDITIONS, require_cuda_training
from pvz_rl.presentation.reporting import make_report
from pvz_rl.provenance import file_hash, metadata, write_json


def run_suite(cfg, output, *, resume=False):
    cfg = copy.deepcopy(cfg)
    require_cuda_training(cfg)
    cfg["conditions"] = {"masked": cfg["conditions"]["masked"]}
    output = Path(output)
    journal_path = output / "suite.json"
    if resume:
        journal = json.loads(journal_path.read_text("utf-8"))
        original = json.loads((output / "metadata.json").read_text("utf-8"))["config"]
        require_cuda_training(original)
        if any(not k.startswith("masked-") for k in journal["jobs"]):
            raise ValueError(
                "This suite contains retired training modes; start a new single-policy suite."
            )
        original["conditions"] = {"masked": original["conditions"]["masked"]}
        if research_config(original) != research_config(cfg):
            raise ValueError("Suite resume requires its original configuration")
    else:
        output.mkdir(parents=True, exist_ok=False)
        journal = {"state": "training", "config_hash": digest(cfg), "jobs": {}, "evaluations": {}}
        write_json(output / "metadata.json", metadata(cfg, kind="formal-suite"))
        write_json(journal_path, journal)
    if journal["state"] == "complete":
        return journal
    try:
        for condition in TRAINING_CONDITIONS:
            if condition not in cfg["conditions"]:
                continue
            job_cfg = copy.deepcopy(cfg)
            for seed in cfg["training"]["learner_seeds"]:
                key = f"{condition}-{seed}"
                attempts = journal["jobs"].setdefault(key, [])
                checkpoint = None
                if attempts:
                    last = Path(attempts[-1])
                    status_path = last / "status.json"
                    status = (
                        json.loads(status_path.read_text("utf-8")) if status_path.exists() else {}
                    )
                    if status.get("state") == "complete":
                        continue
                    for name in ("interrupted.zip", "latest.zip"):
                        if (last / name).exists():
                            checkpoint = last / name
                            break
                destination = output / "training" / key / f"attempt-{len(attempts) + 1}"
                attempts.append(str(destination.resolve()))
                journal.update(state="training", active=key)
                write_json(journal_path, journal)
                train(job_cfg, condition, seed, destination, resume=checkpoint)

        # Test results are exposed only after all training configurations are frozen and run.
        journal["state"] = "evaluating"
        write_json(journal_path, journal)
        evaluations = journal["evaluations"]
        families = ["preset", *cfg["evaluation"]["ood_families"]]
        for policy in [*BASELINES, *journal["jobs"]]:
            evaluation_cfg = cfg
            if policy in BASELINES:
                model, condition, learner_seed, baseline, checkpoint_hash = (
                    None,
                    "masked",
                    None,
                    policy,
                    None,
                )
            else:
                run = Path(journal["jobs"][policy][-1])
                if not (run / "best.zip").exists():
                    journal.setdefault("skipped_evaluations", {})[policy] = (
                        "No validated checkpoint: no stage passed or its evaluation is pending."
                    )
                    write_json(journal_path, journal)
                    continue
                model, data = load_policy(run / "best.zip")
                evaluation_cfg = data["config"]
                condition, learner_seed, baseline = data["condition"], data["learner_seed"], None
                checkpoint_hash = file_hash(run / "best.zip")
            for family in families:
                key = f"{policy}/{family}"
                attempts = evaluations.setdefault(key, [])
                if attempts:
                    status = Path(attempts[-1]) / "status.json"
                    if (
                        status.exists()
                        and json.loads(status.read_text("utf-8")).get("state") == "complete"
                    ):
                        continue
                destination = (
                    output / "evaluation" / policy / family / f"attempt-{len(attempts) + 1}"
                )
                attempts.append(str(destination.resolve()))
                journal["active"] = key
                write_json(journal_path, journal)
                split = "test" if family == "preset" else "ood"
                levels = cfg["evaluation"]["levels" if family == "preset" else "ood_levels"]
                evaluate(
                    evaluation_cfg,
                    policy=model,
                    condition=condition,
                    baseline=baseline,
                    learner_seed=learner_seed,
                    seeds=seed_values(cfg, split),
                    levels=levels,
                    family=family,
                    split=split,
                    output=destination,
                    record=True,
                    training_steps=model.num_timesteps if model else 0,
                    checkpoint_hash=checkpoint_hash,
                )
                write_json(
                    destination / "metadata.json",
                    {
                        "config_hash": digest(cfg),
                        "checkpoint_hash": checkpoint_hash,
                        "policy": policy,
                    },
                )
        paths = [Path(attempts[-1]) / "episodes.jsonl" for attempts in evaluations.values()]
        curves = [
            Path(attempts[-1]) / "learning-curve.jsonl"
            for attempts in journal["jobs"].values()
            if (Path(attempts[-1]) / "learning-curve.jsonl").exists()
        ]
        reports = journal.setdefault("reports", [])
        report_dir = output / f"report-{len(reports) + 1}"
        reports.append(str(report_dir.resolve()))
        journal["state"] = "reporting"
        write_json(journal_path, journal)
        make_report(paths, report_dir, cfg, curves)
        journal.update(state="complete", active=None)
        write_json(journal_path, journal)
        return journal
    except BaseException as exc:
        journal.update(
            state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=repr(exc)
        )
        write_json(journal_path, journal)
        raise
