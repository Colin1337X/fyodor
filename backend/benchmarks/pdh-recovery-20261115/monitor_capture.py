"""Own one cooperative PDH collector around a diagnostic workload."""
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read_complete(path):
    if not path.exists():
        return []
    # A writer may be between write and flush; do not parse a partial last line.
    return [json.loads(line) for line in path.read_text().split('\n')[:-1] if line]


class MonitorState(dict):
    """Serializable evidence with a live, non-serialized collector handle."""
    worker = None

    def check_alive(self):
        if self.worker is None:
            raise RuntimeError('Monitor has no collector')
        status = self.worker.poll()
        if status is not None:
            raise RuntimeError(f'Monitor exited during workload (exit {status})')


@contextmanager
def capture(folder):
    if os.name != 'nt':
        raise RuntimeError('PDH capture requires Windows')
    folder = Path(folder)
    directory = Path(__file__).resolve().parent
    paths = {key: folder / name for key, name in
             [('trace', 'gpu-pdh.jsonl'), ('stop', 'gpu-pdh.stop'),
              ('manifest', 'gpu-monitor.json'), ('log', 'gpu-monitor.log')]}
    if any(path.exists() for path in paths.values()):
        raise RuntimeError('Refusing to overwrite monitor evidence')
    source = directory.parent / 'gpu-activity-20261114/cuda_identity.py'
    spec = importlib.util.spec_from_file_location('cuda_identity', source)
    identity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(identity)
    devices = identity.identities()
    # Fyodor's current CUDA provider selects ordinal zero. Multi-node adapter
    # attribution needs a separate policy; do not silently treat it as node zero.
    adapter = next(device for device in devices['devices'] if device['ordinal'] == 0)
    if adapter['device_node_mask'] != 1:
        raise RuntimeError('Multi-node adapter attribution is not implemented')
    command = [sys.executable, str(directory / 'capture_pdh.py'), '--output', str(paths['trace']),
               '--stop', str(paths['stop']), '--luid', adapter['counter_luid'], '--samples', '3600']
    metadata = MonitorState(accepted_benchmark=False, identity=devices, adapter=adapter,
                    command=command, errors=[], ready=False)
    worker = None
    with paths['log'].open('x') as log:
        try:
            worker = subprocess.Popen(command, stdout=log, stderr=log,
                                      creationflags=subprocess.CREATE_NO_WINDOW)
            metadata.worker = worker
            metadata['process_id'] = worker.pid
            deadline = time.monotonic() + 20
            while True:
                if worker.poll() is not None:
                    raise RuntimeError('Monitor exited before workload')
                rows = read_complete(paths['trace'])
                if any(row.get('kind') == 'error' for row in rows):
                    raise RuntimeError('Monitor initialization failed')
                samples = [row for row in rows if row.get('kind') == 'sample']
                if len(samples) >= 2:
                    if (rows[0].get('provider') != 'PDH wildcard array' or
                        rows[0].get('adapter_luid') != adapter['counter_luid']):
                        raise RuntimeError('Monitor adapter identity mismatch')
                    metadata['ready'] = True
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError('Monitor readiness timeout')
                time.sleep(.1)
            paths['manifest'].write_text(json.dumps(metadata, indent=2))
            yield metadata
            metadata.check_alive()
        except BaseException as error:
            metadata['errors'].append(str(error) or type(error).__name__)
            raise
        finally:
            if worker is not None:
                # A full post-workload interval brackets the final decode even
                # when the benchmark exits immediately after its last output.
                if metadata['ready'] and worker.poll() is None:
                    time.sleep(2)
                paths['stop'].touch(exist_ok=False)
                try:
                    metadata['exit_code'] = worker.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    worker.terminate()
                    metadata['forced_termination'] = True
                    metadata['exit_code'] = worker.wait(timeout=15)
                    metadata['errors'].append('Monitor failed cooperative shutdown')
                try:
                    rows = read_complete(paths['trace'])
                    metadata['successful_stop'] = bool(rows and rows[-1].get('kind') == 'end' and
                        rows[-1].get('requested_stop') is True and rows[-1].get('failed') is False and
                        metadata['exit_code'] == 0 and not metadata.get('forced_termination'))
                except (OSError, ValueError) as error:
                    metadata['successful_stop'] = False
                    metadata['errors'].append(str(error))
            if not metadata.get('successful_stop'):
                metadata['errors'].append('Monitor failed cooperative shutdown; timing rejected')
            paths['manifest'].write_text(json.dumps(metadata, indent=2))

    if not metadata.get('successful_stop'):
        raise RuntimeError('Monitor failed cooperative shutdown; timing rejected')
