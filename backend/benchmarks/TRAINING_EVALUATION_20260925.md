# Held-out training evaluation — September 25, 2026

Native training now accepts a separate validation dataset for random-init
pretraining, CPT, SFT and DPO. It evaluates the current policy before the first
update (after loading a resume checkpoint), at a global optimizer-step interval,
and after the final requested update. Evaluation uses the same tokenizer,
context windows, response masks and objective as training.

```powershell
build-cpu/fyodor-train.exe --data train.txt --output trained.gguf --checkpoint run.ckpt --metrics run.csv --steps 100 --eval-data validation.txt --eval-every 10 --eval-records 0
```

`--eval-data` enables evaluation. `--eval-every` defaults to 10 and must be
positive. `--eval-records` defaults to 0 (all records/windows); positive values
select a fixed prefix on every pass. The record limit is not random sampling.
These two numeric options require an evaluation dataset in CLI invocations.
Training and validation each retain the existing 64 MiB input bound; structured
files retain the 100,000-line bound. Both datasets are loaded and validated
upfront. The record limit bounds the evaluation pass, not dataset parsing or
initial DPO reference caching. Dataset allocations are outside graph budgets.

Cross-entropy aggregates by actual supervised-token count, including unequal
sequence lengths and a shorter final corpus window. Masked SFT prompt labels
do not contribute. DPO averages pairs, using the original base/initial policy
as its fixed reference. Reference scores are cached before resume restoration;
the resumed policy must not become its own reference. There is no perplexity
label on DPO loss and no model-quality claim from a decreasing short-run loss.

## State and interruption contract

Evaluation never clears gradients, runs backward, updates parameters/moments,
advances the optimizer step or consumes initialization randomness. Evaluation
data, interval and prefix limit are excluded from checkpoint identity because
they do not affect training. They may change on resume. Existing NYARUN v1
checkpoints remain compatible.

The control pipe is polled between evaluation records. A Stop request discards
the incomplete validation result and saves the last completed training state.
It does not publish a partial mean as a complete validation loss. Dataset
parsing and DPO reference preparation still precede these polling boundaries.
Evaluation failure before the first update fails the run. A later evaluation
failure stops further updates, attempts normal checkpoint/model saving, and
returns failure so the desktop retains its error log. Output I/O can still
fail; this is not an atomic checkpoint or power-loss guarantee.

`evaluation_start` announces a pass; completed `eval_step` lines contain loss,
processed tokens, supervised units/pairs, evaluated records, elapsed milliseconds
and maximum graph bytes. CSV adds `validation_loss`, `eval_ms`, `eval_tokens`
and `eval_units` to update rows. Unscheduled/interrupted passes leave these
fields blank. The initial validation result is in the log, with no fabricated
optimizer-update CSV row. Training throughput continues to exclude evaluation;
validation time is reported separately and adds to whole-run elapsed time.

The desktop exposes the optional dataset, interval and prefix limit. Validation
loss has its own step and record count, separate from the training-loss trace.
Telemetry is scoped to the latest run-start marker so a new run cannot inherit
an older run's validation value. Missing evaluation remains blank.

## Forward-only graph implementation

`nya_train_graph_create_for_evaluation(limit, executor)` creates an explicit
graph-local mode. Parameter leaves borrow values without exposing parameter
gradients. Intermediate gradient allocations and saved embedding/label/RoPE
metadata are omitted. Attention reuses one probability row instead of retaining
all heads' probability matrices, with identical score, softmax and value
reduction order. Forward tensors still belong to the graph until destruction;
this is not an activation-liveness allocator. Backward on this graph is rejected.
Ordinary training graph behavior and tolerances remain unchanged.

The implementation is original native C. Source study of PyTorch's
[gradient-mode implementation](https://github.com/pytorch/pytorch/blob/main/torch/autograd/grad_mode.py)
informed the separation of forward evaluation from gradient tracking. Fyodor
uses explicit graph ownership rather than a thread-local mode. No upstream code
was copied, no Python/C++ runtime was added, and the graph remains CPU-only.
The supported decoders currently have no dropout or batch-normalization state
requiring an additional train/eval transition.

## Correctness evidence

The [36-process suite](training-evaluation-20260925/process.json) runs all four
modes with accumulation 3. Enabling evaluation must preserve every training
loss, full checkpoint hash and GGUF hash, including split/resumed runs. It also
checks fixed-prefix selection, changing the prefix and worker count on resume,
global evaluation intervals, CSV blanks, Unicode validation paths, malformed
or missing data, invalid options, and stopping during a long initial pass with
exact step-zero recovery. [Driver](../tests/train_eval.py).

Native unit tests compare unequal-mask CE and unequal-length DPO against an
independent weighted objective. Entire serialized parameter/gradient/moment
state must remain byte-identical after full, limited and memory-failing
evaluation. Forward-only and ordinary graphs match bitwise for branched
objectives, RoPE/GQA with causal/window/vision-group masks, LLaMA full/LoRA
decoders and Gemma full/LoRA decoders. Evaluation-only memory budgets accept
the forward graph while rejecting the larger training graph. Existing
finite-difference, optimizer, export and inference checks remain unchanged.

All 15 stages of the [validation matrix](training-evaluation-20260925/validation.json)
passed: C17 CPU 47 passed/4 unavailable-hardware skips; C17/C23 CUDA+Vulkan
72/4 each; ONNX 48/4; ASan+UBSan 47/4; physical optcpp/CUDA removal 47/4 each;
and CUTLASS removal 67/4. The four skips are real ROCm/MLX hardware checks,
not failed tests counted as success. Frontend tests passed 11/11, its production
build passed, and both Rust tests passed with `FYODOR_TEST_TRAINER` explicitly
pointing at the current native trainer. The [27-file source snapshot](training-evaluation-20260925/source-hashes.json)
matches the validated production/test sources.

The rebuilt Windows NSIS package is checked by
[the extraction/process verifier](training-evaluation-20260925/verify_package.py):
all resource hashes must match, and the desktop binary permits only the expected
Tauri bundle marker substitution. Its trainer runs the 36 evaluation processes,
54 stop processes, 24 Unicode processes and complete CLI workflow script.
The [package record](training-evaluation-20260925/desktop-package.json) stores
the installer hash and exact commands. The installer was not installed, and
interactive UI, POSIX and MSVC backend execution are not covered by these tests.

### Real TinyLlama checkpoint

The [real-model driver](training-evaluation-20260925/real_evaluation.py) resumes
the preserved pre-executor step-3 TinyLlama Q4_K_M rank-2 CPT checkpoint with
three CPU workers, context 8 and one further update. It evaluates a separate
fixed eight-token validation window before and after that update. Results:

- Validation loss: 7.46900272 at step 3; 7.40202045 at step 4.
- Evaluation graph: 72,064,540 bytes (68.7 MiB); training graph: 143,521,224
  bytes (136.9 MiB). This compares graph allocations, not process RAM.
- Evaluation elapsed: 495.6 and 456.6 ms. These are two observations, not a
  repeated throughput benchmark.
- Export SHA-256: `1f9f5d9a4198399f36b4063b7c22a057182ea87c940a697a58869f807fb48005`
- Checkpoint SHA-256: `3a1a21efdade6dc4d4afadd52252df1ac15ca00718bebd4e4e6ca2e3e94423ca`

Both hashes match the earlier no-evaluation four-update result exactly. The
driver removed only its own large export after hashing and retained checkpoint,
CSV and [resource evidence](training-evaluation-20260925/real-evaluation.json).
Windows process queries every 50 ms observed 812,371,968 bytes OS peak working
set and 180,277,248 bytes sampled peak private commit over the whole run,
including export. Working set includes mapped/shared pages; private commit is
not physical RSS. No GPU memory or useful-convergence claim is made.

## Disabled-evaluation regression measurements

Same Windows/Ryzen 5 9600X six-core host, GCC 16.2 C17 Release `-O3`, CPU only;
before is the preserved pre-evaluation trainer, after includes this change.
Pairs alternate process order, with no concurrent build/test work. All losses,
checkpoints and exports match exactly. Means ± sample SD in tokens/sec:

| Workload | Before | After |
|---|---:|---:|
| Small | 9,473.28 ± 297.19 | 9,149.16 ± 249.76 |
| Medium | 2,131.54 ± 524.67 | 2,486.84 ± 139.23 |
| Long | 4,453.68 ± 107.23 | 4,501.24 ± 35.73 |
| Tail | 14,521.40 ± 447.15 | 14,441.53 ± 198.69 |

[Raw paired evidence](training-evaluation-20260925/disabled-paired/metadata.json):
four processes per version/shape, 14 updates, first two excluded from timing.
The small average is 3.4% lower; medium has a large unexplained baseline outlier.
No sample is discarded and no disabled-training speedup is claimed.
A [longer small-model measurement](training-evaluation-20260925/disabled-small-sustained/metadata.json),
eight pairs with 200 updates and 20 warmup, gave 9,358.47 ± 900.75 before and
9,297.83 ± 572.43 after (0.994×). Substantial shared-desktop variation remains;
this does not establish zero overhead. The benefit here is state-preserving
validation and lower evaluation graph memory.

GPU training, schedulers, periodic atomic checkpoints, activation recomputation,
streaming datasets and inference throughput gaps remain outstanding. Evaluation
currently shares the training graph's decoder/tokenizer/objective support; it is
not a general external benchmark or task-accuracy harness.
