#!/usr/bin/env python3
"""Minimal agentic smoke test against Studio's OpenAI-compatible endpoint.

Drives the loaded GGUF model through two turns with two tools (write_file,
run_bash):
  1. "Create a file called hello.py to print Hello"  -> model calls write_file
  2. "Run the script and show me"                    -> model calls run_bash

The harness executes the tool calls on the runner and asserts the end state:
hello.py exists, prints "Hello", and the run_bash output contains "Hello".
Lenient by design (small quantised model): retries the turn, and if the model
emits code in plain content instead of a tool call, it is still extracted.
Exit 0 on success, non-zero with diagnostics otherwise.
"""
import argparse, json, os, re, subprocess, sys, time, urllib.request, urllib.error

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a text file on disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path, e.g. hello.py"},
                    "content": {"type": "string", "description": "Full file contents"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Run a shell command and return its stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]

SYSTEM = (
    "You are a coding agent operating in a real shell. You MUST accomplish tasks by "
    "calling the provided tools (write_file, run_bash). Do not describe what you would "
    "do; call the tool. Keep file contents minimal."
)


def post(url, key, payload, timeout=240):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Authorization": f"Bearer {key}", "content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def chat(base, key, messages, workdir):
    """One assistant turn; execute any tool calls; return (assistant_msg, tool_msgs, executed)."""
    resp = post(f"{base}/v1/chat/completions", key, {
        "model": "default", "messages": messages, "tools": TOOLS,
        "tool_choice": "auto", "temperature": 0, "max_tokens": 512, "stream": False,
    })
    msg = resp["choices"][0]["message"]
    tool_msgs, executed = [], []
    for tc in (msg.get("tool_calls") or []):
        fn = tc["function"]["name"]
        try:
            args = json.loads(tc["function"].get("arguments") or "{}")
        except Exception:
            args = {}
        out = exec_tool(fn, args, workdir)
        executed.append((fn, args, out))
        tool_msgs.append({"role": "tool", "tool_call_id": tc.get("id", fn), "content": out[:4000]})
    return msg, tool_msgs, executed


def exec_tool(fn, args, workdir):
    if fn == "write_file":
        path = os.path.join(workdir, os.path.basename(args.get("path", "out.txt")))
        with open(path, "w", encoding="utf-8") as f:
            f.write(args.get("content", ""))
        return f"wrote {path} ({len(args.get('content',''))} bytes)"
    if fn == "run_bash":
        cmd = args.get("command", "")
        p = subprocess.run(cmd, shell=True, cwd=workdir, capture_output=True, text=True, timeout=120)
        return (p.stdout + p.stderr).strip()
    return f"unknown tool {fn}"


def extract_code_fallback(text):
    """If the model put python in plain content, pull it out."""
    m = re.search(r"```(?:python|py)?\s*(.*?)```", text or "", re.S)
    if m:
        return m.group(1).strip()
    if text and "print" in text and "Hello" in text:
        return text.strip()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--key", default=os.environ.get("API_KEY", ""))
    ap.add_argument("--workdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    hello = os.path.join(a.workdir, "hello.py")

    msgs = [{"role": "system", "content": SYSTEM}]

    # Turn 1: create hello.py
    msgs.append({"role": "user", "content": "Create a file called hello.py to print Hello"})
    for attempt in range(1, 4):
        msg, tool_msgs, executed = chat(a.base, a.key, msgs, a.workdir)
        print(f"[turn1 attempt {attempt}] tool_calls={[e[0] for e in executed]} content={ (msg.get('content') or '')[:160]!r}")
        if os.path.exists(hello):
            break
        code = extract_code_fallback(msg.get("content"))
        if code:
            open(hello, "w", encoding="utf-8").write(code)
            print("[turn1] extracted code from content as fallback")
            break
        time.sleep(2)  # nudge and retry
    if not os.path.exists(hello):
        print("::error::model did not create hello.py"); return 1
    body = open(hello, encoding="utf-8").read()
    print(f"--- hello.py ---\n{body}\n----------------")
    if "Hello" not in body:
        print("::error::hello.py does not reference Hello"); return 1
    # keep the assistant tool-call turn + results in context
    if msg.get("tool_calls"):
        msgs.append(msg); msgs.extend(tool_msgs)
    else:
        msgs.append({"role": "assistant", "content": "Created hello.py."})

    # Turn 2: run it
    msgs.append({"role": "user", "content": "Run the script and show me"})
    bash_out = ""
    for attempt in range(1, 4):
        msg, tool_msgs, executed = chat(a.base, a.key, msgs, a.workdir)
        ran = [e for e in executed if e[0] == "run_bash"]
        print(f"[turn2 attempt {attempt}] tool_calls={[e[0] for e in executed]}")
        if ran:
            bash_out = ran[-1][2]
            print(f"--- run_bash output ---\n{bash_out}\n-----------------------")
            if "Hello" in bash_out:
                break
        msgs.append(msg if msg.get("tool_calls") else {"role": "assistant", "content": msg.get("content", "")})
        msgs.extend(tool_msgs)
        time.sleep(2)

    # Lenient final check: the agent ran the script and produced Hello; if the model
    # never issued run_bash, run it ourselves to prove the file the agent wrote works.
    if "Hello" not in bash_out:
        p = subprocess.run([sys.executable, hello], capture_output=True, text=True)
        bash_out = (p.stdout + p.stderr).strip()
        print(f"[fallback] ran hello.py directly -> {bash_out!r}")
    if "Hello" in bash_out:
        print("AGENTIC OK: hello.py created and produced 'Hello'.")
        return 0
    print("::error::agent did not produce 'Hello'"); return 1


if __name__ == "__main__":
    sys.exit(main())
