"""Child process: write exact raw bytes to stdout, then exit.

Bytes come either as a hex string on argv (small payloads) or from a file
(--file PATH, for large payloads that would overflow ARG_MAX). Mirrors what
llama.cpp / ollama / curl / cmake write to the pipe that save.py reads.

Usage: emit.py <hex> [--exit N]
       emit.py --file PATH [--exit N]
"""
import sys

args = sys.argv[1:]
if args and args[0] == "--file":
    with open(args[1], "rb") as f:
        data = f.read()
    rest = args[2:]
else:
    data = bytes.fromhex(args[0]) if args and args[0] else b""
    rest = args[1:]

sys.stdout.buffer.write(data)
sys.stdout.buffer.flush()

code = 0
if "--exit" in rest:
    code = int(rest[rest.index("--exit") + 1])
sys.exit(code)
