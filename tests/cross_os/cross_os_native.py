"""Native cross-OS reproduction of unslothai/unsloth#6218.

Runs on whatever OS the GitHub runner provides and reproduces the EXACT save.py
call shapes (text-mode subprocess.run capture + Popen streaming) against a child
that emits the same UTF-8 bytes llama.cpp / ollama emit (box-drawing, the curly
quote whose UTF-8 contains 0x9D, an accented path, CJK).

It uses the OS's REAL default encoding for the "before" behaviour -- no codec
forcing -- so:
  * on Windows (cp1252 default) BEFORE must crash with UnicodeDecodeError -> the
    bug is real and reproduced natively; the fix (utf-8 + replace) must be clean.
  * on Linux / macOS (utf-8 default) BEFORE and AFTER are both clean -> the fix
    is a no-op and nothing is broken.

Exits non-zero if the OS-appropriate expectation is not met (so CI goes red on a
genuine failure). stdlib only; no torch / GPU / unsloth import.
"""
import locale
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EMIT = str(HERE / "emit.py")
PY = sys.executable

# Make our own diagnostics safe + readable regardless of the console code page.
# This only affects what WE print; the BEFORE test below decodes the *child's*
# output (the decode side the PR fixes), which is independent of our stdout.
ORIG_STDOUT_ENC = sys.stdout.encoding
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

PAYLOAD = ("[██━] quantizing blk.0 ” C:\\Users\\José\\café.gguf モデル\n" * 5).encode("utf-8")
EXPECTED = PAYLOAD.decode("utf-8")


def child():
    return [PY, EMIT, PAYLOAD.hex()]


def run_capture(encoding, errors):
    r = subprocess.run(child(), capture_output=True, text=True, encoding=encoding, errors=errors)
    return r.stdout


def popen_stream(encoding, errors):
    out = []
    with subprocess.Popen(child(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding=encoding, errors=errors, bufsize=1) as sp:
        for line in sp.stdout:
            out.append(line)
        sp.wait()
    return "".join(out)


SHAPES = {"run_capture": run_capture, "popen_stream": popen_stream}


def main():
    pref = locale.getpreferredencoding(False)
    is_utf8 = pref.lower().replace("-", "") in {"utf8", "utf_8"} or sys.flags.utf8_mode == 1
    print("=" * 74)
    print(f"CROSS-OS NATIVE  system={platform.system()} {platform.machine()}  python={platform.python_version()}")
    print(f"  locale.getpreferredencoding={pref}  sys.stdout(orig)={ORIG_STDOUT_ENC}  utf8_mode={sys.flags.utf8_mode}")
    print(f"  host class: {'UTF-8 (Linux/macOS/WSL)' if is_utf8 else 'non-UTF-8 (Windows code page / C locale)'}")
    print("=" * 74)

    fails = []
    for sname, fn in SHAPES.items():
        # BEFORE = pre-PR behaviour: text mode, no encoding -> OS locale default, strict.
        try:
            before = fn(None, None)
            bstate, bdetail = "clean", ""
        except UnicodeDecodeError as e:
            bstate, bdetail = "CRASH_UnicodeDecodeError", str(e).splitlines()[0]

        # AFTER = the PR fix.
        try:
            after = fn("utf-8", "replace")
            acrash = None
        except Exception as e:  # noqa: BLE001
            after, acrash = None, f"{type(e).__name__}: {e}"
        acorrect = after == EXPECTED

        print(f"[{sname:12s}] BEFORE (OS default decode): {bstate}" + (f"  ({bdetail[:48]})" if bstate != "clean" else ""))
        print(f"[{sname:12s}] AFTER  (utf-8 + replace)  : {'clean, decoded correctly' if (acrash is None and acorrect) else ('CRASH ' + str(acrash)) if acrash else 'clean but WRONG decode'}")

        if acrash is not None:
            fails.append(f"{sname}: AFTER(fix) crashed: {acrash}")
        elif not acorrect:
            fails.append(f"{sname}: AFTER(fix) decoded incorrectly")

        if is_utf8:
            if bstate != "clean":
                fails.append(f"{sname}: expected BEFORE clean on UTF-8 host, got {bstate}")
        else:
            if not bstate.startswith("CRASH"):
                fails.append(f"{sname}: expected BEFORE to crash on non-UTF-8 host (bug not reproduced!), got {bstate}")

    print("-" * 74)
    if is_utf8:
        print("BASIS: UTF-8 host -> BEFORE and AFTER both clean => fix is a no-op; Linux/macOS NOT broken.")
    else:
        print("BASIS: non-UTF-8 host -> BEFORE crashes (#6218 reproduced natively), AFTER clean => fix works on Windows.")

    if fails:
        print(f"RESULT: FAIL ({len(fails)})")
        for f in fails:
            print("  FAIL:", f)
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
