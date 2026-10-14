"""Bounded real-loss fitting calibration; synthetic public boards, no quality claim."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from pvz_rl.config import load_config
from pvz_rl.learning.batch_transport import BatchPrefetch, FittingSpool
from pvz_rl.learning.cuda_buffer import CompleteGameBuffer
from pvz_rl.learning.flat_q import batch_loss
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


def measure(cfg, *, repeats=3, rows_per_width=16):
    """Each repetition commits four complete passes with fixed cohort denominators."""
    torch.set_num_threads(cfg["training"]["torch_threads"])
    torch.manual_seed(101)
    model = TwoStageTransformerPolicy(cfg).cuda().train()
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
        if performance["compile_kernels"]:
            network.enable_compilation()
    optimizer = torch.optim.Adam(model.master_parameters(), lr=cfg["training"]["learning_rate"])
    measurements = []
    for name, widths in WORKLOAD_WIDTHS.items():
        for action_mix in ("wait", "mixed", "tile"):
            with TemporaryDirectory(prefix="pvz-fit-benchmark-") as temporary:
                buffer = CompleteGameBuffer(temporary, 1, cfg=cfg)
                public = [observation(cfg, width)[0] for width in widths]
                boards = [board for board in public for _ in range(rows_per_width)]
                rows = np.zeros(len(boards), dtype=buffer.dtype)
                rows["active"] = True
                indices = np.arange(len(rows))
                rows["action"] = (indices % 9) * 45 + 1
                if action_mix == "wait":
                    rows["action"] = 0
                elif action_mix == "mixed":
                    rows["action"][indices % 14 != 0] = 0
                rows["accepted"] = indices % 2 == 0
                rows["target"] = (indices % 7 - 3) / 4
                buffer.append(rows, boards)
                branches = np.where(rows["action"] == 0, 0,
                                    np.where(rows["action"] <= 360, 1, 2))
                counts = np.zeros((3, 2), dtype=np.int64)
                np.add.at(counts, (branches, rows["accepted"].astype(int)), 1)
                prepared = perf_counter()
                spool = FittingSpool(buffer)
                spool_seconds = perf_counter() - prepared

                def passes():
                    for _ in range(cfg["training"]["n_epochs"]):
                        model.zero_grad(set_to_none=True)
                        with_transport = BatchPrefetch(
                            buffer, cfg["training"]["batch_size"] * performance["fit_batch_groups"],
                            "cuda", cfg["encoding"]["max_entities"], spool=spool,
                        )
                        try:
                            for chunk, obs, fields in with_transport:
                                loss, *_ = batch_loss(model, chunk, counts, obs, fields,
                                                       teacher=teacher, consistency_counts=counts[1:])
                                loss.backward()
                                model.accumulate_master_gradients()
                                model.clear_runtime_grad(set_to_none=True)
                        finally:
                            with_transport.close()
                        torch.nn.utils.clip_grad_norm_(model.master_parameters(),
                            cfg["training"]["max_grad_norm"], error_if_nonfinite=True)
                        optimizer.step()
                        model.sync_runtime_from_masters()
                        with torch.no_grad():
                            for ema, online in zip(teacher.parameters(), model.parameters(), strict=True):
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
                    repetitions.append(dict(seconds=perf_counter() - started,
                        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                        free_mib=free / 2**20, total_mib=total / 2**20))
                measurements.append(dict(workload=name, actions=action_mix, widths=widths,
                    rows=len(rows), passes=cfg["training"]["n_epochs"],
                    spool_seconds=spool_seconds, cold_seconds=cold_seconds,
                    repetitions=repetitions, transport=dict(buffer.transport_metrics),
                    encoders=[dict(stage=e.stage, network=n,
                        compiled_shapes=list(e._compiled_shapes),
                        collection=e.collection_metrics,
                        fitting=e._fitting_inference_cache.metrics,
                        status=e.compilation_status,
                        compilation_backend=e.compilation_backend,
                        compilation_error=e.compilation_error)
                        for n, net in (("online", model), ("ema", teacher))
                        for e in net.encoders]))
                spool.close()
                buffer.close()
    return dict(device=torch.cuda.get_device_name(), torch=torch.__version__,
                config=cfg, complete=True, measurements=measurements)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--rows-per-width", type=int, default=16)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--species-microbatch", type=int)
    parser.add_argument("--tile-microbatch", type=int)
    parser.add_argument("--token-budget", type=int)
    parser.add_argument("--groups", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-compile", action="store_true",
                        help="use eager fitting for a low-memory correctness rehearsal")
    args = parser.parse_args()
    cfg = load_config(args.config)
    from pvz_rl.gpu import select_gpu

    select_gpu(args.gpu)
    cfg["training"]["batch_size"] = min(64, cfg["training"]["batch_size"])
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
        performance["compile_kernels"] = False
    write_json(args.output, measure(cfg, repeats=args.repeats, rows_per_width=args.rows_per_width))


if __name__ == "__main__":
    main()
