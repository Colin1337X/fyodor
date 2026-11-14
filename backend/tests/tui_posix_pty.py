"""Exercise resource-browser navigation through a real POSIX terminal."""
import os
import json
import tempfile
from pathlib import Path
import sys
if os.name != 'posix':
    sys.exit(77)
import errno
import fcntl
import pty
import select
import struct
import subprocess
import termios
import time

master, slave = pty.openpty()
original = termios.tcgetattr(slave)
fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 28, 100, 0, 0))
child = None

def expect(text):
    received = bytearray()
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if select.select([master], [], [], 0.1)[0]:
            try:
                chunk = os.read(master, 65536)
            except OSError as error:
                if error.errno == errno.EIO:
                    break
                raise
            if not chunk:
                break
            received.extend(chunk)
            if text in received:
                return
        if child.poll() is not None:
            break
    raise AssertionError(f'Missing terminal output {text!r}; exit={child.poll()}')

try:
    with tempfile.TemporaryDirectory(prefix='fyodor-resource-pty-') as folder:
        db = str(Path(folder) / 'workspace.db')
        base = ['--store', db, '--namespace', 'test']
        for i in range(2):
            package = dict(schema=1, uri=f'fyodor://writing/notes/00000000-0000-4000-8000-{i+1:012d}',
                           title=f'Note {i}', content=f'UNIQUE_DETAIL_{i}', metadata={}, provenance={})
            if i == 0:
                package['content'] += ''.join(f'\nLONG_LINE_{n:03}' for n in range(50))
            run = subprocess.run([sys.argv[2], *base, 'resource', 'import'],
                                 input=json.dumps(package), text=True, capture_output=True, timeout=10)
            assert run.returncode == 0, run.stderr
        child = subprocess.Popen([sys.argv[1], *base, '--color', 'none'],
                                 stdin=slave, stdout=slave, stderr=slave, close_fds=True)
        expect(b'local administrator')
        os.write(master, b'\x1b[B\r')
        expect(b'UNIQUE_DETAIL_1')
        os.write(master, b'b\x1bOA\r')
        expect(b'UNIQUE_DETAIL_0')
        os.write(master, b'\x1b[F')
        expect(b'LONG_LINE_049')
        os.write(master, b'\x1b[H')
        expect(b'UNIQUE_DETAIL_0')
        os.write(master, b'\x1b[6~')
        expect(b'LONG_LINE_025')
        os.write(master, b'\x1b[5~')
        expect(b'UNIQUE_DETAIL_0')
        os.write(master, b'b\x1b[6~\r')
        expect(b'UNIQUE_DETAIL_1')
        os.write(master, b'b\x1b[5~\r')
        expect(b'UNIQUE_DETAIL_0')
        os.write(master, b':resource lis\x1b[3~t\r')
        expect(b'Command exit 0')
        os.write(master, b':wrong\x1b:resource list\r')
        expect(b'Command exit 0')
        os.write(master, b"?jjj:resource 'unterminated\r")
        expect(b'Command syntax error:')
        os.write(master, b'?jjj:resource import\r')
        expect(b'Import packages through the CLI.')
        os.write(master, b':' + b'x' * 160 + b'TAIL_MARKER')
        expect(b'TAIL_MARKER')
        os.write(master, b'\x7fZ')
        expect(b'TAIL_MARKEZ')
        os.write(master, b'\x1b')
        os.write(master, b'q')
        expect(b'\x1b[?1049l')
        assert child.wait(timeout=5) == 0
        assert termios.tcgetattr(slave) == original, 'Terminal settings were not restored'
        print('Resource PTY navigation, command errors, Escape lookahead and restoration passed')
finally:
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)
    os.close(master)
    os.close(slave)
