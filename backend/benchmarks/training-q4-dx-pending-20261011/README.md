# Q4_K-specific backward tile candidate: parked

Following the rejected universal 64 experiment, a template specialization used
a 64-wide reduction tile only for Q4_K and retained 32 for every other type.
The candidate passed all 82 CUDA training tests. The exact unaccepted diff is
`candidate.patch`; the early whole-step before/after outputs are retained in
`interrupted-results.json`.

The user requested an immediate benchmark-score refresh after clearing GPU
competitors. The in-progress candidate benchmark was deliberately stopped and
its incomplete results are excluded from performance acceptance. The production
kernel source was restored to the committed narrow-forward implementation.
This candidate is not shipped or claimed as an improvement. It remains a
follow-up experiment needing complete controlled comparisons and sanitizer
checks before acceptance.

The reusable comparison driver now accepts output directory, baseline binary,
contexts, steps, warmup count and repetitions, and records those settings plus
revision/source hashes. Existing defaults preserve the original narrow-forward
experiment. The device tests additionally check Q4_K backward results bitwise
against ordered F32 C products/additions, including two accumulated passes and
reduction boundaries 63/64/65/127/128/129. `--q4-dx` isolates these tests.

After restoring the production kernel, all six focused boundary cases passed;
see `restored-kernel-tests.log`.
`nya-test-training-device --cuda` also passed the complete resident device suite on the restored kernel; see `restored-full-device-tests.log`.

The later [controlled Q4_K-only experiment](../training-q4-dx-20261011/README.md)
completed validation and accepted the specialization. This directory retains
the original interrupted evidence; its timings remain excluded.
