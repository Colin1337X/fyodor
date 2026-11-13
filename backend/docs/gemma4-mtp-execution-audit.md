# Gemma 4 MTP execution audit, 2026-11-13

Current code is a correctness implementation of speculative sampling, not an accelerated batched verifier. This audit concerns the native runtime; no external framework is proposed as its replacement.

## Verified current behavior

- `backend/core/llm_cpu.c:nya_llm_generate_draft` frees both resident plans for MTP, because the assistant reads the target normalized hidden vector and borrows target KV from host arrays. The target sets `reference_only` and `nya_llm_forward` consequently passes a context with compute disabled to the host forward. Assistant calls go directly through `nya_llm_forward_impl`; its matvec calls can still use the configured assisted provider, but its state and attention remain on the host. It is therefore inaccurate to call this a resident CUDA MTP path.
- Assistant layers bind the last non-sharing target layer of the matching full/sliding attention kind, checking head dimension and KV-head count. The target prefix length is frozen for a proposal round; recurrent assistant hidden states use the same cache prefix.
- Verification visits proposals one at a time and invokes a target forward after each emitted token if generation continues. Acceptance does not currently replace those target passes with one batched verification call. `target_steps` counts processed target positions, including the prompt; it is not a count of saved launches or measured acceleration.
- The CUDA planner explicitly declines assistant models in `CUDA/resident.inc:cuda_plan_create`. Its batch path projects only the last position to logits, so it cannot currently return all per-position distributions needed by speculative acceptance. The compute-plan C interface exposes KV reading but no shared device hidden-state/borrowed-KV binding for an assistant.
- The existing real-model evidence in `../benchmarks/gemma4-local-20261029/mtp-template.log` demonstrates greedy parity for four generated tokens, two draft tokens and one acceptance. It does not establish performance, useful sustained acceptance rate, or sampling-distribution parity at scale.

## Implementation requirements before a speed claim

1. Add a bounded C-owned verification result interface that can request logits for each selected position in a batch, with explicit capacities and overflow/allocation checks. Preserve the current final-position-only path for ordinary inference. Backend support must be capability-checked and optional.
2. Validate batched target logits against serial decode across causal/window boundaries, shared KV, graph/reset transitions, and rejection rewinds. Keep token positions and logits alignment explicit; each proposal is checked against the distribution before consuming that proposal.
3. Bind assistant hidden state and borrowed KV through opaque device-plan interfaces with explicit lifetime and synchronization rules. Do not download the full target cache each round. Keep CPU fallback available, including when optional backend directories are absent.
4. Integrate batched acceptance with existing greedy and probabilistic rejection/residual semantics. Rejected speculative suffixes must not poison future cache reads; ring retention must support the maximum proposal/rollback window.
5. Measure sustained acceptance and total generated tokens/s including draft work, verification, rewinds and synchronization, against the same target without speculation. Use matched model/context/sampling settings and clean environment acceptance. Do not reuse the existing last-logit-only bench as evidence for batched speculative speed.

These are remaining requirements, not completed capabilities. The short-decode 125% target remains unmet; kernel-level diagnostics do not close this execution gap.
