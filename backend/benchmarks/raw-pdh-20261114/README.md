# Raw PDH counter diagnostic

This isolated observer reads formatted and raw wildcard GPU-engine arrays from
one collection, without another collection between the two reads. It observes
only the selected adapter and workload PID plus System PID 4. It launches no GPU
work and makes no benchmark acceptance decision. Existing collectors and
coverage rules are unchanged.

The rejected FFN-reuse run returned status 3221228474 (0xC0000BBA), named
PDH_CSTATUS_INVALID_DATA in the installed Windows headers. Microsoft's
[counter-status documentation](https://learn.microsoft.com/en-us/windows/win32/perfctrs/checking-pdh-interface-return-values)
requires checking counter status before using its value and describes invalid
data as distinct from a failed API call. Possible causes do not establish the
cause on this machine.

The observer uses
[PdhGetRawCounterArrayW](https://learn.microsoft.com/en-us/windows/win32/api/pdh/nf-pdh-pdhgetrawcounterarrayw)
with size-query/retry handling and preserves the
[raw structure](https://learn.microsoft.com/en-us/windows/win32/api/pdh/ns-pdh-pdh_raw_counter)
status, timestamp, first/second values and multi-count. It does not derive a
utilization rate from invalid data. `abi.c` compiled and ran against the installed
Windows headers, confirming the Windows x64 layout used by ctypes.

`readback-long.jsonl` contains 30 one-second-spaced collections during the
existing readback long-generation correctness run (PID 23360). The observer
exited zero; its terminal record reports no failure or changed inputs. Header
hashes were independently verified. `analysis.json` records no invalid formatted
values and no decreasing valid raw counters in this window. The workload CUDA
engine and System copy engine both have positive values in all 30 samples.
This does not reproduce the earlier invalid-data condition or causally attribute
System work. It does not establish clean decode coverage or change any score.

The generation process was already running before this observer started. A
future diagnostic spanning context creation/destruction is needed to investigate
whether counter transitions explain the intermittent invalid-data records.
