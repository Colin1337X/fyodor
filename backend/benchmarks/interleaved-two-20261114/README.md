# Batch-two interleaved activation experiment

Original Fyodor prototype in optcpp/matvec/interleaved_two.cu. It converts two
activation vectors to float2 input while retaining native Q4_K/Q6_K weight
arithmetic and each output accumulation/reduction order. Not integrated into
runtime or CMake. No production behavior changes in this experiment.

All 17 raw correctness cases pass exact comparison, finite/output guards and
independent CPU transpose/guards. All 11 actual Gemma projection shapes and
six small cases are included, with varied signed/non-binary F16 block scales.
Small cases also pass the unchanged independent float64 bound. No dedicated
batch-two CUDA memory checker or real-model integration test has run yet.

Graph microbenchmarks include the conversion in every candidate invocation.
Five samples in each variant order, three warmup graphs, 100 projections per
sample, synthetic reused weights. These are diagnostic timings, not accepted
decode benchmarks or predictions of whole-generation speed.

| Type | Columns | Rows | Native ms | Convert + interleaved ms | Speed % |
| --- | ---: | ---: | ---: | ---: | ---: |
| 12 | 3840 | 512 | 0.006820 | 0.006999 | 97.434 |
| 12 | 3840 | 2048 | 0.019741 | 0.020023 | 98.589 |
| 12 | 3840 | 4096 | 0.035067 | 0.035183 | 99.671 |
| 12 | 3840 | 8192 | 0.066017 | 0.065790 | 100.344 |
| 12 | 3840 | 15360 | 0.124000 | 0.122595 | 101.146 |
| 12 | 4096 | 3840 | 0.035098 | 0.035168 | 99.802 |
| 12 | 8192 | 3840 | 0.068441 | 0.068297 | 100.210 |
| 12 | 15360 | 3840 | 0.138173 | 0.138644 | 99.661 |
| 14 | 3840 | 2048 | 0.024108 | 0.019704 | 122.349 |
| 14 | 3840 | 262144 | 2.744699 | 2.211233 | 124.125 |
| 14 | 15360 | 3840 | 0.179095 | 0.139706 | 128.194 |

Q4 does not justify integration: gains are small and some shapes regress. Q6
improves 22.349-28.194% in this sweep and merits further validation as a
format-specific candidate. No blanket batch-two dispatch change is warranted.
The correctness workload was also observed by a failed PowerShell monitoring
experiment; this was not a timing run. Monitor limitations are documented in
gpu-monitor-20261114. All measured results, including regressions, are retained.
