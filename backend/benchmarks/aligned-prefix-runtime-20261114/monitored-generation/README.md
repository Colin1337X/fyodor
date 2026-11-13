# Rejected monitored generation diagnostic

Both short and long children exited zero; all 24 cases passed output parity
(three rounds, four generation modes, two contexts, 32 generated tokens).
The launch idle gate failed. The PDH collector then failed after 168 samples
with status 0xc0000bba; the old launcher continued without checking its health.
Coverage also reports unattributed System PID 4 activity, invalid target counter
values and missing brackets. Preserve the original data and reject all timings.
These results do not update the accepted llama.cpp score. The subsequent launcher
health checks stop the owned workload if its collector dies.
