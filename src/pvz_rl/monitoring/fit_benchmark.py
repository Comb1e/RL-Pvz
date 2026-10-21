"""Bounded real-loss fitting calibration; synthetic public boards, no quality claim."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from pvz_rl.config import load_config
from pvz_rl.learning.action_routing import classify_actions, workload_counts
from pvz_rl.learning.batch_transport import BatchPrefetch, FittingSpool
from pvz_rl.learning.cuda_buffer import CompleteGameBuffer
from pvz_rl.learning.flat_q import CohortCoefficients, batch_loss
from pvz_rl.monitoring.entity_benchmark import observation
from pvz_rl.policy.two_stage_transformer import TwoStageTransformerPolicy
from pvz_rl.provenance import write_json

# Keep the workload labels explicit in the artifact.  The sparse and exact
# bucket-boundary cases exercise the observed fit distribution and padding
# transitions; medium and dense cases fill in the interior up to the public
# 256-entity cap so a preset is not selected from sparse boards alone.
WORKLOAD_WIDTHS = {
    "sparse": (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21),
    "medium": (24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192),
    "dense": (200, 208, 216, 224, 232, 240, 248, 252, 255, 256),
    "boundary": (31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256),
}


ACTION_MIXES = ("wait", "mixed", "tile", "species_invalid", "tile_invalid", "rejected_heavy")


def benchmark_rows(cfg, widths, rows_per_width, action_mix, dtype):
    """Coherent public board/card/outcome fixtures for the real fitting path."""
    from pvz_game import Rules
    from pvz_game.cuda.schema import REASONS

    from pvz_rl.envs.encoding import ObservationEncoder

    encoder = ObservationEncoder(cfg, Rules())
    rows = np.zeros(len(widths) * rows_per_width, dtype=dtype)
    rows["active"] = True
    boards = []
    for index, width in enumerate(np.repeat(widths, rows_per_width)):
        board = observation(cfg, int(width))[0]
        entities = board["entities"].copy()
        # Tile-error fixtures need an occupied plot even at the sparse boundary.
        if action_mix in ("tile_invalid", "rejected_heavy") and not np.any(
            (entities[:, 0] >= 1) & (entities[:, 0] <= 8)
        ):
            entities[0] = [1, encoder.states["plant:ready"], 0, 0, 300, 0, 0, 0, 0, 0, 0]
        board = dict(entities=entities, globals=board["globals"].copy())
        board["globals"][0] = 10.0
        board["globals"][6:14] = 0
        occupied = np.zeros(45, bool)
        for entity in entities:
            if 1 <= entity[0] <= 8:
                occupied[entity[2] * 9 + entity[3] // Rules().game["units_per_tile"]] = True
        board["globals"][14] = occupied.sum() / encoder.count_scale
        legal = np.r_[True, np.tile(~occupied, 8), occupied]
        branch = index % 8
        empty = int(np.flatnonzero(~occupied)[0])
        action = 1 + branch * 45 + empty
        accepted, reason = True, None
        kind = action_mix
        branch_available = np.ones(10, dtype=bool)
        if kind == "mixed":
            kind = "tile" if index % 14 == 0 else "wait"
        elif kind == "rejected_heavy":
            kind = ("species_invalid", "tile_invalid", "species_invalid", "tile_invalid", "tile")[
                index % 5
            ]
        if kind == "wait":
            action = 0
        elif kind == "species_invalid":
            # Public sun/cooldown failure: impossible species, even if occupied.
            board["globals"][0] = 0
            board["globals"][6 + branch] = 0.5
            branch_available[1 + branch] = False
            legal[1:361] = False
            accepted, reason = False, "card_recharging"
        elif kind == "tile_invalid":
            planted = int(np.flatnonzero(occupied)[0])
            if index % 2:
                action, reason = 361 + empty, "empty_tile"
            else:
                action, reason = 1 + branch * 45 + planted, "occupied_tile"
            accepted = False
        elif occupied.any() and index % 3 == 0:
            action = 361 + int(np.flatnonzero(occupied)[0])
        route, usable = classify_actions(
            [action],
            [accepted],
            legal[None],
            [reason],
            branch_available=branch_available[None],
        )
        rows[index]["action"], rows[index]["accepted"] = action, accepted
        rows[index]["route"], rows[index]["branch_usable"] = route[0], usable[0]
        rows[index]["rejection_reason"] = REASONS.index(reason)
        rows[index]["executed_action"] = action if accepted else 0
        rows[index]["target"] = (index % 7 - 3) / 4
        boards.append(board)
    return rows, boards


def measure(cfg, *, repeats=3, rows_per_width=16, trajectory=None, initial=None):
    """Each repetition commits four complete passes with fixed cohort denominators."""
    torch.set_num_threads(cfg["training"]["torch_threads"])
    torch.manual_seed(101)
    model = TwoStageTransformerPolicy(cfg).cuda().train()
    if initial is not None:
        from pvz_rl.learning.training import initial_weights

        weights, _ = initial_weights(initial, cfg)
        model.load_state_dict(weights)
    teacher = TwoStageTransformerPolicy(cfg).cuda().eval().requires_grad_(False)
    teacher.load_state_dict(model.runtime_state_dict())
    performance = cfg["training"]["performance"]
    for network in (model, teacher):
        for encoder in network.encoders:
            encoder.microbatch = performance.get(
                f"fit_{encoder.stage}_microbatch", performance["fit_encoder_microbatch"]
            )
            encoder.token_budget = performance["fit_token_budget"]
            if hasattr(encoder, "fitting"):
                encoder.fitting = True
        if performance["compile_fitting"]:
            network.enable_compilation()
    optimizer = torch.optim.Adam(model.master_parameters(), lr=cfg["training"]["learning_rate"])
    measurements = []
    workloads = {"trajectory": ()} if trajectory is not None else WORKLOAD_WIDTHS
    for name, widths in workloads.items():
        for action_mix in ("actual",) if trajectory is not None else ACTION_MIXES:
            with TemporaryDirectory(prefix="pvz-fit-benchmark-") as temporary:
                if trajectory is None:
                    buffer = CompleteGameBuffer(temporary, 1, cfg=cfg)
                    rows, boards = benchmark_rows(
                        cfg, widths, rows_per_width, action_mix, buffer.dtype
                    )
                    buffer.append(rows, boards)
                else:
                    state = json.loads((Path(trajectory) / "metadata.json").read_text())
                    buffer = CompleteGameBuffer.restore(trajectory, state, temporary)
                    rows = None
                prepared = perf_counter()
                spool = FittingSpool(buffer)
                spool_seconds = perf_counter() - prepared
                counts = spool.outcome_counts
                coefficients = CohortCoefficients.prepare(
                    counts,
                    spool.consistency_counts,
                    "cuda",
                    cfg["training"]["objective"]["accepted_outcome_share"],
                )

                def passes():
                    for _ in range(cfg["training"]["n_epochs"]):
                        model.zero_grad(set_to_none=True)
                        with_transport = BatchPrefetch(
                            buffer,
                            cfg["training"]["batch_size"] * performance["fit_batch_groups"],
                            "cuda",
                            cfg["encoding"]["max_entities"],
                            spool=spool,
                        )
                        try:
                            for chunk, obs, fields in with_transport:
                                loss, *_ = batch_loss(
                                    model,
                                    chunk,
                                    counts,
                                    obs,
                                    fields,
                                    teacher=teacher,
                                    consistency_counts=spool.consistency_counts,
                                    coefficients=coefficients,
                                )
                                loss.backward()
                                model.accumulate_master_gradients()
                                model.clear_runtime_grad(set_to_none=True)
                        finally:
                            with_transport.close()
                        torch.nn.utils.clip_grad_norm_(
                            model.master_parameters(),
                            cfg["training"]["max_grad_norm"],
                            error_if_nonfinite=True,
                        )
                        optimizer.step()
                        model.sync_runtime_from_masters()
                        with torch.no_grad():
                            for ema, online in zip(
                                teacher.parameters(), model.parameters(), strict=True
                            ):
                                ema.lerp_(online, 1 - cfg["training"]["objective"]["ema_decay"])

                torch.cuda.synchronize()
                cold_start = perf_counter()
                passes()
                torch.cuda.synchronize()
                cold_seconds = perf_counter() - cold_start
                repetitions = []
                for _ in range(repeats):
                    torch.cuda.reset_peak_memory_stats()
                    started = perf_counter()
                    passes()
                    torch.cuda.synchronize()
                    free, total = torch.cuda.mem_get_info()
                    repetitions.append(
                        dict(
                            seconds=perf_counter() - started,
                            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                            peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                            free_mib=free / 2**20,
                            total_mib=total / 2**20,
                        )
                    )
                measurements.append(
                    dict(
                        workload=name,
                        actions=action_mix,
                        widths=widths,
                        rows=buffer.size,
                        representatives=len(spool),
                        compaction=spool.compaction,
                        passes=cfg["training"]["n_epochs"],
                        spool_seconds=spool_seconds,
                        cold_seconds=cold_seconds,
                        routed_workload=workload_counts(spool.rows),
                        repetitions=repetitions,
                        transport=dict(buffer.transport_metrics),
                        encoders=[
                            dict(
                                stage=e.stage,
                                network=n,
                                compiled_shapes=list(e._compiled_shapes),
                                collection=e.collection_metrics,
                                fitting=e._fitting_inference_cache.metrics,
                                status=e.compilation_status,
                                compilation_backend=e.compilation_backend,
                                compilation_error=e.compilation_error,
                            )
                            for n, net in (("online", model), ("ema", teacher))
                            for e in net.encoders
                        ],
                    )
                )
                spool.close()
                buffer.close()
    return dict(
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        config=cfg,
        complete=True,
        measurements=measurements,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--rows-per-width", type=int, default=16)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--init-from", type=Path)
    parser.add_argument("--species-microbatch", type=int)
    parser.add_argument("--tile-microbatch", type=int)
    parser.add_argument("--token-budget", type=int)
    parser.add_argument("--groups", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="use eager fitting for a low-memory correctness rehearsal",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    from pvz_rl.gpu import apply_gpu_profile, select_gpu

    device = select_gpu(args.gpu)
    cfg, _ = apply_gpu_profile(cfg, device=device, path=args.config)
    performance = cfg["training"]["performance"]
    if args.species_microbatch:
        performance["fit_species_microbatch"] = args.species_microbatch
    if args.tile_microbatch:
        performance["fit_tile_microbatch"] = args.tile_microbatch
    if args.token_budget:
        performance["fit_token_budget"] = args.token_budget
    if args.groups:
        performance["fit_batch_groups"] = args.groups
    if args.threads:
        cfg["training"]["torch_threads"] = args.threads
    if args.no_compile:
        performance["compile_fitting"] = False
    write_json(
        args.output,
        measure(
            cfg,
            repeats=args.repeats,
            rows_per_width=args.rows_per_width,
            trajectory=args.trajectory,
            initial=args.init_from,
        ),
    )


if __name__ == "__main__":
    main()
