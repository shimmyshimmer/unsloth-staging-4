#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Run the file-edit demo (create hello.py, then run it) for ONE agent and echo
# the created file + run transcript, so an asciinema recording of this script
# captures the agent's actual work -- not just the driver's status lines.
#
# Designed to be wrapped by `asciinema rec -c "record-file-edit.sh <agent>"`.
# Writes the underlying drive exit code to recordings/<agent>.rc so the workflow
# can gate on it (asciinema rec itself always exits 0).
set -uo pipefail

AGENT="${1:?usage: record-file-edit.sh <agent>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
mkdir -p "$REPO_ROOT/recordings"

echo "==================================================================="
echo " unsloth start ${AGENT}  --  create hello.py, then run it"
echo " model: unsloth/gemma-4-E4B-it-GGUF  (UD-Q4_K_XL)"
echo "==================================================================="
echo

bash "$SCRIPT_DIR/agent-guides-drive.sh" file-edit "$AGENT"
rc=$?
echo "$rc" > "$REPO_ROOT/recordings/${AGENT}.rc"

echo
echo "----- hello.py created by ${AGENT} -----"
cat "$REPO_ROOT/agent-workdir/${AGENT}/hello.py" 2>/dev/null || echo "(hello.py was not created)"
echo
echo "----- run output (turn 2 transcript, tail) -----"
tail -40 "$REPO_ROOT/logs/${AGENT}-fileedit-turn2.txt" 2>/dev/null || echo "(no transcript captured)"
echo
if [ "$rc" -eq 0 ]; then
  echo "RESULT: ${AGENT} file-edit OK (created hello.py and ran it -> Hello)"
else
  echo "RESULT: ${AGENT} file-edit FAILED (exit ${rc})"
fi
exit "$rc"
