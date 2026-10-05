"""Windows console lifecycle test in a private hidden console, never the user's."""
import ctypes as C
from ctypes import wintypes as W
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import itertools

if sys.platform != "win32":
    sys.exit(77)
if "--child" not in sys.argv:
    startup = subprocess.STARTUPINFO()
    startup.dwFlags = subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    run = subprocess.run([sys.executable, str(Path(__file__).resolve()), sys.argv[1], "--child"],
                         creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=startup, timeout=25, capture_output=True)
    if run.returncode:
        sys.stderr.buffer.write(run.stdout + run.stderr)
    sys.exit(run.returncode)

k = C.WinDLL("kernel32", use_last_error=True)
class Security(C.Structure):
    _fields_ = [("length", W.DWORD), ("descriptor", W.LPVOID), ("inherit", W.BOOL)]
class Coord(C.Structure):
    _fields_ = [("x", C.c_short), ("y", C.c_short)]
class Rect(C.Structure):
    _fields_ = [("left", C.c_short), ("top", C.c_short), ("right", C.c_short), ("bottom", C.c_short)]
class Info(C.Structure):
    _fields_ = [("size", Coord), ("cursor", Coord), ("attributes", W.WORD), ("window", Rect), ("maximum", Coord)]
class Key(C.Structure):
    _fields_ = [("down", W.BOOL), ("repeat", W.WORD), ("virtual", W.WORD), ("scan", W.WORD), ("char", W.WCHAR), ("control", W.DWORD)]
class Event(C.Union):
    _fields_ = [("key", Key), ("padding", C.c_byte * 16)]
class Input(C.Structure):
    _fields_ = [("type", W.WORD), ("event", Event)]
k.CreateFileW.argtypes = [W.LPCWSTR, W.DWORD, W.DWORD, C.POINTER(Security), W.DWORD, W.DWORD, W.HANDLE]
k.CreateFileW.restype = W.HANDLE
k.GetConsoleMode.argtypes = [W.HANDLE, C.POINTER(W.DWORD)]
k.CloseHandle.argtypes = [W.HANDLE]
k.WriteConsoleInputW.argtypes = [W.HANDLE, C.POINTER(Input), W.DWORD, C.POINTER(W.DWORD)]
k.GetConsoleScreenBufferInfo.argtypes = [W.HANDLE, C.POINTER(Info)]
k.ReadConsoleOutputCharacterW.argtypes = [W.HANDLE, W.LPWSTR, W.DWORD, Coord, C.POINTER(W.DWORD)]

def handle(name):
    security = Security(C.sizeof(Security), None, True)
    result = k.CreateFileW(name, 0xC0000000, 3, C.byref(security), 3, 0, None)
    assert result != W.HANDLE(-1).value, C.get_last_error()
    return result

def mode(h):
    value = W.DWORD()
    assert k.GetConsoleMode(h, C.byref(value)), C.get_last_error()
    return value.value

def wait_for(check):
    until = time.monotonic() + 5
    while time.monotonic() < until:
        if check():
            return
        time.sleep(0.025)
    raise AssertionError("Console did not reach expected state")

def visible():
    h = handle("CONOUT$")
    try:
        info = Info()
        assert k.GetConsoleScreenBufferInfo(h, C.byref(info))
        size = info.size.x * info.size.y
        text, got = C.create_unicode_buffer(size + 1), W.DWORD()
        assert k.ReadConsoleOutputCharacterW(h, text, size, Coord(0, 0), C.byref(got))
        return text.value
    finally:
        k.CloseHandle(h)

def key(h, char):
    record, written = Input(), W.DWORD()
    record.type = 1
    record.event.key = Key(True, 1, ord(char.upper()) if char.isalpha() else 0, 0, char, 0)
    assert k.WriteConsoleInputW(h, C.byref(record), 1, C.byref(written)) and written.value == 1

hin, hout = handle("CONIN$"), handle("CONOUT$")
try:
    original = mode(hin), mode(hout)
    with tempfile.TemporaryDirectory(prefix="fyodor-console-") as folder:
        for quit_key, color in itertools.product(("q", "\x03"), ("truecolor", "256", "16", "none")):
            startup = subprocess.STARTUPINFO()
            startup.dwFlags = subprocess.STARTF_USESTDHANDLES
            startup.hStdInput, startup.hStdOutput, startup.hStdError = hin, hout, hout
            child = subprocess.Popen([str(Path(sys.argv[1]).resolve()), "--store", str(Path(folder) / "test.db"), "--namespace", "test", "--theme", "fyodor", "--color", color],
                                     startupinfo=startup, close_fds=False, env={**os.environ, "TERM": "xterm"})
            try:
                wait_for(lambda: mode(hin) & 6 == 0)
                wait_for(lambda: "Fyodor | Resources" in visible())
                key(hin, "?")
                wait_for(lambda: "Browse: j/k" in visible())
                for char in ":resource list\r":
                    key(hin, char)
                wait_for(lambda: "Command exit 0" in visible())
                key(hin, "t")
                wait_for(lambda: "Theme: Fyodor Dark" in visible())
                key(hin, quit_key)
                assert child.wait(timeout=5) == 0
                assert (mode(hin), mode(hout)) == original, "Terminal modes not restored"
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()
finally:
    k.CloseHandle(hin)
    k.CloseHandle(hout)
