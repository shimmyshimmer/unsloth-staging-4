import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile


def item(source, name):
    start = source.index(name)
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def run_arm(path, scratch):
    source = path.read_text()
    struct = item(source, "struct BackendLiveness")
    function = item(source, "fn watchdog_confirm_keeps_backend")
    rust = (
        "#[derive(Default)]\n"
        + struct
        + "\n"
        + function
        + r"""
fn main() {
    for alive in [false, true] {
        for active in [false, true] {
            for timeout in [false, true] {
                let value = BackendLiveness {
                    alive, inference_active: active, probe_timed_out: timeout,
                    ..BackendLiveness::default()
                };
                println!("{alive},{active},{timeout},{}", watchdog_confirm_keeps_backend(&value));
            }
        }
    }
}
"""
    )
    src = scratch / "probe.rs"
    binary = scratch / ("probe.exe" if os.name == "nt" else "probe")
    src.write_text(rust)
    subprocess.run(["rustc", "--edition=2021", str(src), "-o", str(binary)], check = True)
    output = subprocess.check_output([str(binary.resolve())], text = True)
    rows = [[field == "true" for field in line.split(",")] for line in output.splitlines()]
    expected = [False, False, False, True, True, True, True, True]
    assert len(rows) == 8 and all(len(row) == 4 for row in rows), rows
    failures = [row[:3] for row, want in zip(rows, expected) if row[3] != want]
    return {
        "source": str(path.resolve()),
        "sha256": hashlib.sha256(source.encode()).hexdigest(),
        "function_sha256": hashlib.sha256(function.encode()).hexdigest(),
        "rows": rows,
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type = Path, required = True)
    parser.add_argument("--head", type = Path, required = True)
    parser.add_argument("--out", type = Path, required = True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents = True, exist_ok = True)
    with tempfile.TemporaryDirectory(dir = args.out.parent) as directory:
        scratch = Path(directory)
        base = run_arm(args.base, scratch)
        head = run_arm(args.head, scratch)
    result = {
        "platform": platform.platform(),
        "rustc": subprocess.check_output(["rustc", "-vV"], text = True),
        "base": base,
        "head": head,
    }
    args.out.write_text(json.dumps(result, indent = 2))
    print(json.dumps(result, indent = 2))
    assert base["failures"] == [[True, False, False], [True, False, True]], base
    assert head["failures"] == [], head


if __name__ == "__main__":
    main()
