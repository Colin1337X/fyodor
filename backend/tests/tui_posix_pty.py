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


def send_large(data):
    # Drain repaint output while writing, so neither side fills the PTY buffer.
    flags = fcntl.fcntl(master, fcntl.F_GETFL)
    fcntl.fcntl(master, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    offset = 0
    deadline = time.monotonic() + 10
    try:
        while offset < len(data):
            assert time.monotonic() < deadline, 'PTY input stalled'
            readable, writable, _ = select.select([master], [master], [], .1)
            if readable:
                os.read(master, 65536)
            if writable:
                try:
                    offset += os.write(master, data[offset:offset+256])
                except BlockingIOError:
                    pass
    finally:
        fcntl.fcntl(master, fcntl.F_SETFL, flags)

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
        os.write(master, b':DRAFT_MARKER\x1b[A')
        expect(b':resource import')
        os.write(master, b'\x1b[A')
        expect(b":resource 'unterminated")
        os.write(master, b'\x1b[B\x1b[B')
        expect(b':DRAFT_MARKER')
        os.write(master, b'\x1b:resource list\r')
        expect(b'Command exit 0')
        os.write(master, b':\x1b[A\r')
        expect(b'Command exit 0')
        os.write(master, b':\x1b[A\x1b[A')
        expect(b':resource import')  # Consecutive list submissions collapse.
        os.write(master, b'\x1b')
        for i in range(18):
            os.write(master, f':HISTORY_{i:02}\r'.encode())
            expect(b'Command exit')
        os.write(master, b':' + b'\x1b[A' * 20)
        expect(b':HISTORY_02')  # Only the newest 16 submissions survive.
        os.write(master, b'\x1b')
        os.write(master, b':' + b'x' * 160 + b'TAIL_MARKER')
        expect(b'TAIL_MARKER')
        os.write(master, b'\x7fZ')
        expect(b'TAIL_MARKEZ')
        os.write(master, b'\x1b')
        send_large(b':' + b'x' * 8192 + b'\x1b[A\x1b[B\r')
        expect(b'Command too long; discarded.')
        os.write(master, b':\x1b[A')
        expect(b':HISTORY_17')  # Neither cancelled nor overflowed input is saved.
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
