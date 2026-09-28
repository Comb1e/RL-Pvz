# Precision and execution controls

The optimization changes execution, not the entity representation or complete-return
objective. For each cohort, the same active decisions, group denominators and
targets contribute to each of four whole-pass gradients. Chronological 256-step
chunks retain the original detached recurrent boundaries. A 64-frame encoder
microbatch does not define an optimizer minibatch.

Let `theta` be FP32 master weights and `Q_b(theta, x)` the network with eligible
feature operations evaluated using BF16 computation copies. Fitting accumulates

\[
g = \sum_{c \in \text{chronological chunks}}\nabla_\theta L_c(Q_b(\theta,x),G),
\qquad \theta' = \operatorname{Adam}(\theta,\operatorname{clip}(g)).
\]

The gradient is accumulated in FP32; no loss scale is applied. BF16's exponent
range matches FP32, but its shorter significand changes rounding. This changes
numerical results and can change subsequent learned decisions. It does not
establish equivalent long-term training trajectories or learning quality.
Keeping master weights in FP32 preserves small updates that would disappear if
rounded immediately to BF16 storage.

Entity embedding outputs and eligible feature matrix operations use BF16.
Embedding sums, residual accumulation, normalization reductions, explicit fallback
softmax, recurrent computation/state, Q heads, objective and optimizer remain
FP32. Encoder outputs are converted to FP32 before the recurrent and tile-head
boundaries. FP64 independent controls retain their precision.

Independent tests compare BF16 branch/tile outputs against FP32 with absolute
tolerance 0.002 and relative tolerance 0.01. The aggregate gradient relative L2
error must be at most 2% on seeded sparse/crowded controls. Internal cell states
can accumulate input quantization differences; a separate control evaluates the
FP32 LSTM on the actual BF16-derived feature inputs and checks its state exactly.
FP32 attention, sequence and gradient controls remain independently tested.
For Adam, the independent first-step formula is `delta = lr*g/(abs(g)+epsilon)`.
Softmax makes a shared key bias mathematically irrelevant, so its tiny FP32
gradients can be rounding noise. Adam can amplify differences in those near-zero
gradients; controls bound update differences by this formula rather than assuming
bitwise-identical parameters across different reduction orders.

An uncommitted pass contributes no durable optimizer update. Nonfinite BF16 loss
or gradients discard all accumulated gradients and restart that pass in FP32.
The remainder of the cohort stays FP32. A nonfinite FP32 retry fails. Allocation
failure uses successively smaller encoder microbatches, then one outer encoder
checkpoint; it never changes observations, targets or recurrent chunk boundaries.
Restarted passes are not added to completed-pass or optimizer-step counters.

Two reusable pinned slots are charged to the trajectory RAM allowance. A single
producer preserves sequence order; each slot's transfer event completes before
it is overwritten. The compute stream waits for transfer completion, and copied
device tensors retain stream ownership until their consumers finish. Checkpoints
drain queued copies and retain durable arrays plus effective execution settings;
prefetched chunks and partial gradients are reconstructed after interruption.

Performance evidence must separate setup/warmup, collection, host preparation,
transfer wait, transfer device time, fitting, optimizer time and total wall time.
Preparation and transfer can overlap computation, so their durations are not
additive. Memory reports distinguish Torch allocation, CuPy pools and host RAM.
Short throughput checks are not formal training or learning-quality estimates.
