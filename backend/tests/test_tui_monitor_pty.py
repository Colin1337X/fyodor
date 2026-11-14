"""Exercise monitor input and terminal restoration through a private POSIX PTY."""
import os
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
    child = subprocess.Popen([sys.argv[1], '--monitor', '--color', 'none'],
                             stdin=slave, stdout=slave, stderr=slave, close_fds=True)
    expect(b'WORKSPACE')
    os.write(master, b':wrong\x1b:open /nonexistent/fyodor-input-test.gguf\r')
    expect(b'/nonexistent/fyodor-input-test.gguf')
    os.write(master, b':overview\xe2\r')
    expect(b'WORKSPACE')
    os.write(master, b':mod\x1b[3~el\r')
    expect(b'MODEL / CONTAINER INSPECTION')
    os.write(master, ':open /nonexistent/\u6587.gguf\r'.encode())
    expect('/nonexistent/\u6587.gguf'.encode())
    os.write(master, b':quit\r')
    expect(b'\x1b[?1049l')
    assert child.wait(timeout=5) == 0
    assert termios.tcgetattr(slave) == original, 'Terminal settings were not restored'
    print('POSIX PTY input recovery, Unicode, unsupported keys and restoration passed')
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
