# Position within the reference prefill

A follow-up read-only GDB inspection selected the cuda_plan_prefill frame by function name and read its saved registers. r12=7936 is the current batch end (i+chunk), r13=8192 is count, and rbx=0 is initial position. The plan maximum_batch at r14+0x228 is 512. The original disassembly establishes these assignments; original resident.inc establishes the loop and initial 256-token alignment boundary.

Consequently the sampled batch covers zero-based positions 7424 through 7935. Earlier batches in this first prefix call have returned successfully. This is evidence of setup progress, not a numerical pass or an estimate of remaining wall time. It does not establish whether a particular kernel is stalled within this batch. The final 256-token batch and subsequent reference/candidate passes remain. No process was restarted. Both live debugger inspections detached; the offline disassembly does not execute the binary.
