"""Regenerate the checked-in Vulkan shader using an explicitly chosen glslang.

This developer tool is optional; ordinary C builds use the checked-in header.
Usage: python optcpp/attention/generate_spirv.py PATH/TO/glslang [--check]
"""
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('compiler', type=Path)
parser.add_argument('--check', action='store_true')
args = parser.parse_args()
root = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='fyodor-shader-') as tmp:
    binary = Path(tmp) / 'portable.spv'
    subprocess.run([str(args.compiler.resolve()), '-V', str(root / 'portable_f32.comp'),
                    '-o', str(binary)], check=True)
    data = binary.read_bytes()
words = struct.unpack('<' + 'I' * (len(data) // 4), data)
assert words[0] == 0x07230203
header = '/* Generated from portable_f32.comp with glslang -V; no runtime shader compiler. */\n'
header += 'static const uint32_t nya_attention_spv[] = {\n'
header += '\n'.join(','.join(f'0x{word:08x}U' for word in words[i:i+8]) + ','
                    for i in range(0, len(words), 8)) + '\n};\n'
target = root / 'portable_spv.h'
if args.check:
    if target.read_text(encoding='utf-8') != header:
        raise SystemExit('Generated shader differs; inspect compiler version and source changes.')
else:
    target.write_text(header, encoding='utf-8')
