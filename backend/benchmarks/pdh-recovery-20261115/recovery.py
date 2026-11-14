"""Bounded diagnostic recovery; a recovered capture still fails coverage."""
TRANSIENT = 'PdhGetFormattedCounterArrayW(data): PDH status 0xc0000bba'

class Recovery:
    def __init__(self):
        self.consecutive = 0
        self.total = 0

    def collect(self, query, emit, index, clock):
        before = clock()
        try:
            rows = query.collect()
        except RuntimeError as error:
            if str(error) != TRANSIENT:
                raise
            self.consecutive += 1
            self.total += 1
            emit(dict(kind='error', index=index, message=str(error),
                      qpc_before_collect=before, qpc_observed_seconds=clock(),
                      recovery_attempt=self.consecutive, accepted_benchmark=False))
            if self.consecutive >= 3:
                raise RuntimeError('Three consecutive formatted PDH failures') from error
            return None
        self.consecutive = 0
        return before, clock(), rows
