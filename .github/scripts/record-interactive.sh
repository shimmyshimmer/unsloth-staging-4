#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Record the INTERACTIVE TUI of `unsloth start <agent>` (NOT the -p/exec/run
# one-shot path) driving the two-turn demo: create a Python file that prints
# "Hello", then run it. Produces a redacted GIF + first/last-frame PNGs.
#
# Driving is stability-based (agent-agnostic): drive a real tmux pane with
# send-keys, and treat a turn as finished when the pane stops changing for a few
# seconds. asciinema records the pane; agg renders the cast to a GIF.
#
# Isolation: a throwaway HOME + XDG dirs, plus the session-scoped relocation env
# `unsloth start` already sets per agent. Intended to run inside a container with
# no host home mounted; the model server runs on the host (reached via
# UNSLOTH_STUDIO_URL over --network host loopback).
#
# Usage: record-interactive.sh <agent>     (agent: claude codex opencode pi hermes openclaw)
set -uo pipefail

AGENT="${1:?usage: record-interactive.sh <agent>}"

# ---- Inputs from the caller --------------------------------------------------
: "${UNSLOTH_STUDIO_URL:?set UNSLOTH_STUDIO_URL to the running Studio (e.g. http://127.0.0.1:PORT)}"
: "${UNSLOTH_API_KEY:?set UNSLOTH_API_KEY}"
export UNSLOTH_STUDIO_URL UNSLOTH_API_KEY   # `unsloth start` reads both from env
OUT_DIR="${OUT_DIR:-/out}"
COLS="${REC_COLS:-120}"; ROWS="${REC_ROWS:-36}"
TURN_TIMEOUT="${AGENT_INVOKE_TIMEOUT:-300}"   # per-turn settle cap (s)
START_TIMEOUT="${START_TIMEOUT:-180}"          # time for the TUI to come up (s)
STABLE_SECS="${STABLE_SECS:-6}"                # pane-unchanged window = turn done
mkdir -p "$OUT_DIR"

# ---- Prompts (the maintainer's exact phrasing) -------------------------------
T1='Create a Python file to print "Hello"'
T2='Now run it and show me the output'

# ---- Isolated HOME / XDG (belt-and-suspenders) -------------------------------
ORIG_HOME="$HOME"   # capture before override, for the isolation tripwire below
ISO="$(mktemp -d "/tmp/iso-${AGENT}-XXXXXX")"
export HOME="$ISO/home"
export XDG_CONFIG_HOME="$HOME/.config" XDG_DATA_HOME="$HOME/.local/share"
export XDG_STATE_HOME="$HOME/.local/state" XDG_CACHE_HOME="$HOME/.cache"
mkdir -p "$HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$XDG_CACHE_HOME"
WORK="$ISO/work"; mkdir -p "$WORK"
CAST="$OUT_DIR/${AGENT}.cast"
SOCK="rec-${AGENT}"

echo "[rec:$AGENT] HOME=$HOME WORK=$WORK studio=$UNSLOTH_STUDIO_URL"

# ---- Isolation tripwire: real user configs must never be touched -------------
# In the container these paths should not even exist; assert they stay absent /
# unchanged across the run.
REAL_CFG=("$ORIG_HOME/.claude" "$ORIG_HOME/.codex" "$ORIG_HOME/.config/opencode" \
          "$ORIG_HOME/.pi" "$ORIG_HOME/.hermes" "$ORIG_HOME/.openclaw")
declare -A CFG_BEFORE
for d in "${REAL_CFG[@]}"; do
  if [ -e "$d" ]; then CFG_BEFORE[$d]="$(stat -c '%Y:%s' "$d" 2>/dev/null)"; else CFG_BEFORE[$d]="ABSENT"; fi
done

# ---- Per-agent launch table --------------------------------------------------
# START_EXTRA: auto-approve / passthrough flags appended to `unsloth start <agent>`
# (they flow through ctx.args to the real agent binary).
# EXIT_SEQ: keys to type to leave the TUI cleanly.
# PRE_LAUNCH: optional command run (backgrounded) before the TUI (e.g. a daemon).
# Shrink claude's prompt for a small local model: the full ~5.7k-token system
# prompt + every tool schema overruns a 4B model into claude's own retry loop
# (the headless CI uses the same flags). Locate the minimal prompt -- a sibling of
# this script in CI, or under /opt/unsloth-repo in the recorder image.
MINPROMPT=""
for _c in "$(dirname "$0")/ci-min-system-prompt.txt" \
          /opt/unsloth-repo/.github/scripts/ci-min-system-prompt.txt; do
  [ -f "$_c" ] && { MINPROMPT="$_c"; break; }
done

# BUSY_RE: a pane substring shown only while the agent is generating; a turn is
# "done" when it disappears (robust against spinner animation, which never lets the
# pane go byte-stable). Empty -> fall back to pane-hash stability.
START_EXTRA=(); EXIT_SEQ='/exit'; PRE_LAUNCH=''; BUSY_RE=''
case "$AGENT" in
  claude)
    export IS_SANDBOX=1   # lets interactive claude accept --dangerously-skip-permissions
    START_EXTRA=(--dangerously-skip-permissions --tools Bash,Edit,Write,Read)
    [ -n "$MINPROMPT" ] && START_EXTRA+=(--system-prompt-file "$MINPROMPT")
    EXIT_SEQ='/exit'; BUSY_RE='esc to interrupt' ;;
  codex)
    # Interactive Codex auto-approve: bypass approvals + sandbox (the runner has no
    # bubblewrap; this is the same flag the headless path uses, valid top-level).
    START_EXTRA=(--dangerously-bypass-approvals-and-sandbox)
    EXIT_SEQ='/quit'; BUSY_RE='esc to interrupt|Esc to interrupt|Working|working|Thinking|thinking' ;;
  opencode)
    EXIT_SEQ='/exit'; BUSY_RE='esc interrupt|esc to interrupt|Thinking|thinking|Working|working' ;;
  pi)
    EXIT_SEQ='/exit'; BUSY_RE='esc to interrupt|esc to cancel|thinking|working' ;;
  hermes)
    EXIT_SEQ='/quit'; BUSY_RE='esc to interrupt|thinking|working' ;;
  openclaw)
    # OpenClaw's local gateway daemon must be up for the TUI; started in-band.
    PRE_LAUNCH='openclaw gateway'
    EXIT_SEQ='/quit'; BUSY_RE='flibbertigibbeting|running .|streaming .|finishing context'
    # The 4B mis-picks tools from openclaw's terse default agent, so spell out the task.
    T1='Create a file named hello.py containing exactly: print("Hello")'
    T2='Run hello.py with python and show me the output' ;;
  *) echo "[rec:$AGENT] unknown agent" >&2; exit 2 ;;
esac

# The visible command (no key on screen; the key rides in the env).
START_CMD="unsloth start $AGENT ${START_EXTRA[*]}"

# Pre-seed config to skip first-run onboarding that would otherwise intercept the
# driver in a fresh HOME. Claude: theme picker, security notes, trust-folder, and
# the Bypass-Permissions accept are all gated by ~/.claude.json flags.
if [ "$AGENT" = "claude" ]; then
  cat > "$HOME/.claude.json" <<JSON
{
  "hasCompletedOnboarding": true,
  "theme": "dark",
  "bypassPermissionsModeAccepted": true,
  "projects": {
    "$WORK": { "hasTrustDialogAccepted": true, "hasCompletedProjectOnboarding": true }
  }
}
JSON
fi

# ---- tmux helpers ------------------------------------------------------------
TM() { tmux -L "$SOCK" "$@"; }
pane() { TM capture-pane -p -t main 2>/dev/null; }
pane_sig() { pane | sed -e 's/[[:space:]]\+$//' | sha1sum | cut -d' ' -f1; }
send_keys() { TM send-keys -t main "$@"; }
send_line() { TM send-keys -t main -l -- "$1"; TM send-keys -t main C-m; }

# A turn is "done" when the agent goes idle. Prefer its BUSY_RE marker (robust
# against spinner animation); fall back to pane-hash stability. Caps at $1 seconds
# and auto-approves any permission prompt seen while waiting.
wait_idle() {
  local cap="$1" last="" idle=0 t=0 cur
  while [ "$t" -lt "$cap" ]; do
    sleep 2; t=$((t+2))
    approve_if_prompted
    if [ -n "$BUSY_RE" ]; then
      if pane | grep -qiE "$BUSY_RE"; then idle=0; else idle=$((idle+2)); fi
    else
      cur="$(pane_sig)"
      if [ "$cur" = "$last" ]; then idle=$((idle+2)); else idle=0; last="$cur"; fi
    fi
    [ "$idle" -ge "$STABLE_SECS" ] && return 0
  done
  return 1
}

# Wait (capped) until the pane contains a regex, ignoring the prompt echo.
wait_for() {
  local re="$1" cap="$2" t=0
  while [ "$t" -lt "$cap" ]; do
    pane | grep -qiE "$re" && return 0
    sleep 2; t=$((t+2)); approve_if_prompted
  done
  return 1
}

# Heuristic approval watcher: if a permission prompt is on screen, accept it.
approve_if_prompted() {
  local p; p="$(pane)"
  if printf '%s' "$p" | grep -qiE 'allow|approve|yes/no|y/n|run (this )?command|edit file|proceed\?|trust|grant|permission'; then
    # Most TUIs default-highlight the affirmative; Enter accepts. Some need 'y'.
    send_keys Enter; sleep 1
  fi
}

# ---- Record ------------------------------------------------------------------
# Run asciinema directly as the session command (tmux executes it via /bin/sh -c,
# inheriting the server's env). tmux's default pane shell would be a LOGIN shell
# that sources /etc/profile and resets PATH, dropping ~/.local/bin where unsloth
# and claude live; running asciinema directly avoids that. asciinema then records
# a clean `bash --norc` shell; nothing secret is typed (the key rides in the env).
TM kill-server 2>/dev/null || true
TM new-session -d -s main -x "$COLS" -y "$ROWS" \
  "asciinema rec --overwrite -i 2 '$CAST' -c 'bash --norc'"
TM set -g status off 2>/dev/null || true
sleep 4

[ -n "$PRE_LAUNCH" ] && { send_line "$PRE_LAUNCH >\$HOME/daemon.log 2>&1 &"; sleep 6; }

# Show and run the real interactive command.
send_line "cd '$WORK'"
send_line "$START_CMD"

rc=0
wait_idle "$START_TIMEOUT" || echo "[rec:$AGENT] TUI did not settle on startup"
sleep 2

# Turn 1: create hello.py
send_line "$T1"; sleep 3
wait_idle "$TURN_TIMEOUT" || echo "[rec:$AGENT] turn 1 did not settle"
sleep 2

# Turn 2: run it
send_line "$T2"; sleep 3
wait_for 'Hello' "$TURN_TIMEOUT" || true
wait_idle "$TURN_TIMEOUT" || echo "[rec:$AGENT] turn 2 did not settle"
sleep 2

# Exit cleanly, then a hard fallback so asciinema always closes.
send_line "$EXIT_SEQ"; sleep 3
send_keys C-c 2>/dev/null || true; sleep 1
send_keys C-d 2>/dev/null || true; sleep 2

# Wait for asciinema to flush the cast (pane dies when the recorder exits).
for _ in $(seq 1 15); do TM has-session -t main 2>/dev/null || break; sleep 2; done
TM kill-server 2>/dev/null || true

# ---- Success assertions ------------------------------------------------------
HELLO="$WORK/hello.py"
[ -f "$HELLO" ] && cp "$HELLO" "$OUT_DIR/${AGENT}-hello.py" 2>/dev/null || true
if [ -f "$HELLO" ] && grep -q 'Hello' "$HELLO"; then
  echo "[rec:$AGENT] hello.py OK:"; sed -n '1,20p' "$HELLO"
else
  echo "[rec:$AGENT] hello.py missing or lacks 'Hello'"; rc=1
fi

# ---- Isolation assertion -----------------------------------------------------
for d in "${REAL_CFG[@]}"; do
  now="ABSENT"; [ -e "$d" ] && now="$(stat -c '%Y:%s' "$d" 2>/dev/null)"
  if [ "${CFG_BEFORE[$d]}" != "$now" ]; then
    echo "::error::[rec:$AGENT] ISOLATION BREACH: $d changed (${CFG_BEFORE[$d]} -> $now)"; rc=1
  fi
done

# ---- Redact + render ---------------------------------------------------------
if [ -f "$CAST" ]; then
  # Strip the key from the cast BEFORE rendering so neither GIF nor PNG can leak it.
  python3 - "$CAST" "$UNSLOTH_API_KEY" <<'PY'
import sys, io
path, key = sys.argv[1], sys.argv[2]
data = io.open(path, encoding="utf-8", errors="replace").read()
if key:
    data = data.replace(key, "<REDACTED>")
io.open(path, "w", encoding="utf-8").write(data)
PY
  agg --theme monokai --font-size 16 "$CAST" "$OUT_DIR/${AGENT}.gif" || echo "[rec:$AGENT] agg failed"
  convert "$OUT_DIR/${AGENT}.gif[0]"  "$OUT_DIR/${AGENT}-start.png" 2>/dev/null || true
  # Composite the final frame cleanly (avoids GIF disposal ghosting).
  python3 - "$OUT_DIR/${AGENT}.gif" "$OUT_DIR/${AGENT}-final.png" <<'PY' || true
import sys
from PIL import Image
gif, out = sys.argv[1], sys.argv[2]
im = Image.open(gif); im.seek(im.n_frames - 1)
im.convert("RGB").save(out)
PY
else
  echo "[rec:$AGENT] no cast produced"; rc=1
fi

echo "$rc" > "$OUT_DIR/${AGENT}.rc"
echo "[rec:$AGENT] done rc=$rc (artifacts in $OUT_DIR)"
exit "$rc"
