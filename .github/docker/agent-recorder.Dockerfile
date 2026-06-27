# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Throwaway image for recording the INTERACTIVE TUI of `unsloth start <agent>`
# in isolation. The model server runs on the HOST (GPUs); a container started
# from this image with `--network host` reaches it over loopback. No host home
# is mounted, so the maintainer's real ~/.claude, ~/.codex, ~/.config/opencode,
# ~/.pi are physically absent and cannot be read or written.
#
# Build (context = repo root, frontend excluded by the sibling .dockerignore):
#   DOCKER_BUILDKIT=1 docker build -f .github/docker/agent-recorder.Dockerfile \
#     -t unsloth-agent-recorder .
#
# This is PR #6613 validation tooling and is never shipped or merged.
FROM node:22-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/root/.local/bin:${PATH}" \
    # agent-guides-install.sh appends `~/.local/bin` to $GITHUB_PATH; point it at a
    # throwaway file so the append never fails outside Actions.
    GITHUB_PATH=/tmp/github_path \
    PIP_BREAK_SYSTEM_PACKAGES=1

# --- System + recording tools -------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        tmux imagemagick curl ca-certificates git jq procps xz-utils less \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /root/.local/bin /out

# uv (used by install.sh) + asciinema (terminal recorder) + agg (cast -> GIF).
RUN pip3 install --no-cache-dir uv asciinema \
    && curl -fsSL -o /root/.local/bin/agg \
        https://github.com/asciinema/agg/releases/latest/download/agg-x86_64-unknown-linux-gnu \
    && chmod +x /root/.local/bin/agg

# --- Coding-agent CLIs (mirror .github/scripts/agent-guides-install.sh) --------
# Inlined (not the repo script) so this layer stays cached across repo edits.
RUN npm install -g @openai/codex opencode-ai openclaw@latest @earendil-works/pi-coding-agent
RUN curl -fsSL https://claude.ai/install.sh | bash
RUN curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh \
        | bash -s -- --non-interactive --skip-setup --skip-browser --no-skills

# --- Unsloth CLI (our PR code) via install.sh --local --no-torch --------------
# Editable overlay of the copied repo, so `unsloth start` runs exactly this PR's
# start.py (including the new `pi` recipe). Frontend (927M) is excluded by the
# dockerignore; the no-torch backend only needs studio/backend.
ENV UNSLOTH_STUDIO_HOME=/opt/unsloth-studio \
    SKIP_STUDIO_FRONTEND=1
COPY . /opt/unsloth-repo
# install.sh sets up the no-torch venv + editable overlay of our repo (so `unsloth
# start` runs this PR's start.py). SKIP_STUDIO_FRONTEND=1 skips the Tauri frontend
# build (excluded from the context); the venv's `unsloth` launcher is symlinked
# unconditionally since the agent CLI does not need the desktop frontend.
RUN cd /opt/unsloth-repo && (bash install.sh --local --no-torch || true) \
    # install.sh installs unsloth + no-torch-runtime with --no-deps, which on a
    # fresh image leaves the studio backend's web deps (structlog, starlette via
    # fastapi, ...) missing -- `unsloth start` needs them for auth_root. studio.txt
    # is the authoritative, torch-free backend dep set.
    && uv pip install --python /opt/unsloth-studio/unsloth_studio/bin/python --no-cache \
         -r /opt/unsloth-repo/studio/backend/requirements/studio.txt \
    && ln -sf /opt/unsloth-studio/unsloth_studio/bin/unsloth /root/.local/bin/unsloth \
    # Fail the build now (not at record time) if the launcher's import chain breaks.
    && /opt/unsloth-studio/unsloth_studio/bin/python -c \
         "import unsloth_cli._inference as m; m.ensure_studio_backend_path(); import utils.paths; print('utils.paths OK')" \
    && unsloth start --help >/dev/null

# Some agents invoke `python` (not `python3`) to run the file; provide the alias.
RUN ln -sf /usr/bin/python3 /usr/local/bin/python

# Recorder script is bind-mounted at run time (so it can be iterated without a
# rebuild); ENTRYPOINT runs it for one agent.
WORKDIR /work
ENTRYPOINT ["bash", "/opt/record-interactive.sh"]
