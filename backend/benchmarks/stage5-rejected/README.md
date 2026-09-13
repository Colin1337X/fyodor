# Rejected F16 cache experiment

This patch is research evidence, not an enabled backend feature. It replaces parts of the stage 5 F32 fusion/diagnostic implementation to reproduce the F16 experiment. Do not apply it to a working tree with other changes. Full-model parity failed the unchanged 0.001 scaled-error bound: TinyLlama 0.0020303, Gemma 12B 0.0173245. Synthetic tests alone did not establish model correctness. The normal runtime remains F32.

The boundary trace shows CPU -2.2431640625 and GPU -2.24316692352 rounding to different adjacent half values, which amplify downstream. Conversion and small fixtures passed, but the complete model comparison did not. No performance acceptance is claimed.
