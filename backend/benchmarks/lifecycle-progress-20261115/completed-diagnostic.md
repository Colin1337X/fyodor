# Completed short sanitizer diagnostic

Memcheck257 (original session 50712) completed successfully in 3068.23 seconds. Both prefix-257 reset comparisons had max_scaled=2.19546455e-05 against the unchanged 0.0005 bound. The unique final sanitizer summary reports zero errors, the child exits zero, and all build-manifest inputs remain unchanged. This is the filtered cached-query command in a fallback context; it is NOT full-model instrumentation or proof of long cached-query kernel execution. Do not satisfy the pending 8K candidate gate with this result.

The three completed prefill phase records and serial token checkpoints establish sustained slow progress. First reference prefill took 622.483 seconds for 257 tokens (0.412863 tokens/s). Serial samples took roughly 4.31–4.35 seconds per token. These are instrumented diagnostic rates, never accepted inference throughput.

The unsanitized CheckC17 and CheckC23 diagnostics completed in about 20.7 seconds. An additional completed run named Sanitizer257 accidentally omitted --memcheck; its command and manifest correctly record memcheck=false. It is an additional unsanitized run, not sanitizer evidence, and is retained without relabeling or overwriting.

The subsequent isolated five-case filter probe passed: the regex does select/exclude its named fixture kernels. The exact cause of the severe model overhead remains unresolved. There are no active GPU jobs from these completed runs.
