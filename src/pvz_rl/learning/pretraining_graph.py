"""Bounded eager CUDA graph for private forward/backward and FP32 gradient clipping."""

import copy

import torch

from pvz_rl.envs.encoding import EntityBatch


class PretrainingGraph:
    """Optimizer/EMA commits stay outside capture; warmup never updates weights."""

    def __init__(self, policy, observations, targets, max_grad_norm):
        self.policy = policy
        self.observations = EntityBatch(
            *(value.clone() for value in observations.tensors()), validated=True
        )
        self.targets = copy.copy(targets)
        for key, value in vars(targets).items():
            if isinstance(value, torch.Tensor):
                setattr(self.targets, key, value.clone())
        self.runtime = tuple(policy.runtime_parameters("effect_pretrain"))
        self.masters = tuple(policy.master_parameters("effect_pretrain"))
        self.max_grad_norm = max_grad_norm
        self.stream = torch.cuda.Stream(device=policy.device)
        current = torch.cuda.current_stream(policy.device)
        self.stream.wait_stream(current)
        policy.zero_grad(set_to_none=True)
        with torch.cuda.stream(self.stream):
            # Warmup only changes scratch gradients, which capture clears again.
            for _ in range(3):
                policy.zero_grad(set_to_none=True)
                self._backward()
            self.pairs = tuple(
                (m, p)
                for m, p in zip(self.masters, self.runtime, strict=True)
                if p.grad is not None
            )
            for master, parameter in self.pairs:
                master.grad = torch.empty_like(master)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=self.stream):
                for parameter in self.runtime:
                    if parameter.grad is not None:
                        parameter.grad.zero_()
                self._backward()
                torch._foreach_copy_(
                    [m.grad for m, _ in self.pairs], [p.grad for _, p in self.pairs]
                )
                norm = torch.nn.utils.clip_grad_norm_(self.masters, self.max_grad_norm)
                self.finite = torch.isfinite(self.loss.detach()) & torch.isfinite(norm)
        current.wait_stream(self.stream)

    def _backward(self):
        with self.policy.board_context(self.observations) as memory:
            self.loss = self.targets.loss(self.policy, self.observations, memory)
        self.loss.backward()

    def replay(self, observations, targets):
        for destination, source in zip(
            self.observations.tensors(), observations.tensors(), strict=True
        ):
            destination.copy_(source)
        for key, value in vars(targets).items():
            if isinstance(value, torch.Tensor):
                getattr(self.targets, key).copy_(value)
        self.graph.replay()
        return self.loss, self.finite

    def close(self):
        torch.cuda.synchronize(self.policy.device)
        self.graph.reset()
        self.policy.zero_grad(set_to_none=True)
