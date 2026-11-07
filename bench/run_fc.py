"""CI runner: build the fence-colour page for each arm branch and run bench.py over them."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOB = Path.cwd()
W = Path(os.environ["BENCH_WORK"])
OUT = JOB / "out"
REPO = W / "repo"
OUT.mkdir(parents = True, exist_ok = True)
W.mkdir(parents = True, exist_ok = True)
ARMS = [a.split("=", 1) for a in os.environ["FC_ARMS"].split(",")]


def sh(cmd, cwd = None, env = None, timeout = 3600):
    print("+", cmd, flush = True)
    subprocess.run(cmd, shell = True, cwd = cwd, env = env, check = True, timeout = timeout)


def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output = True, text = True, check = True).stdout.strip()


def node_modules(fe):
    h = hashlib.sha1((fe / "package-lock.json").read_bytes()).hexdigest()[:12]
    nm = W / f"nm_{h}"
    if not (nm / "node_modules").exists():
        nm.mkdir(exist_ok = True)
        shutil.copy(fe / "package.json", nm)
        shutil.copy(fe / "package-lock.json", nm)
        sh("npm ci --no-audit --no-fund --loglevel=error", cwd = nm, timeout = 1800)
    (fe / "node_modules").symlink_to(nm / "node_modules")


def main():
    sh(f"git clone -q --filter=blob:none --no-checkout https://github.com/{os.environ['FC_REPO']} {REPO}")
    refs = " ".join(f"+refs/heads/{branch}:refs/remotes/arm/{name}" for name, branch in ARMS)
    sh(f"git fetch -q origin {refs}", cwd = REPO)
    shas, dists = {}, []
    for name, _ in ARMS:
        wt = W / f"wt_{name}"
        sh(f"git worktree add -q --detach {wt} arm/{name}", cwd = REPO)
        shas[name] = git("rev-parse", f"arm/{name}")
        fe = wt / "studio" / "frontend"
        node_modules(fe)
        for f in ("smoke-fence-colour-main.tsx", "smoke-fence-colour.html", "vite.fence-colour.config.ts"):
            shutil.copy(HERE / f, fe / f)
        dist = W / f"dist_{name}"
        sh("npx vite build -c vite.fence-colour.config.ts --logLevel warn", cwd = fe,
           env = dict(os.environ, FENCE_COLOUR_OUT = str(dist)))
        dists.append(f"{name}={dist}")
    (OUT / "arms.json").write_text(json.dumps(shas, indent = 1))
    dist_of = {d.split("=", 1)[0]: d.split("=", 1)[1] for d in dists}
    # Variants: name=arm@file, the arm's build with a stylesheet or a page script applied.
    for spec in filter(None, os.environ.get("FC_VARIANTS", "").split(",")):
        name, rest = spec.split("=", 1)
        arm, path = rest.split("@", 1)
        dists.append(f"{name}={dist_of[arm]}@{HERE / path}")
    order = [a for a in os.environ.get("FC_ORDER", "").split(",") if a]
    if order:
        by = {d.split("=", 1)[0]: d for d in dists}
        dists = [by[a] for a in order]
    env = dict(os.environ, FC_SHAS = json.dumps(shas))
    sh(f"{sys.executable} -u {HERE / 'bench.py'} --arms {','.join(dists)} --engines {os.environ['FC_ENGINE']} "
       f"--throttle {os.environ['FC_THROTTLE']} --reps 1 --out {OUT / 'result.json'} --record-builds {OUT / 'builds.json'}",
       env = env, timeout = 7200)
    for name in filter(None, os.environ.get("FC_MUTLOG", "").split(",")):
        with open(OUT / f"mutlog_{name}.json", "w") as f:
            subprocess.run([sys.executable, "-u", str(HERE / "fc_mutlog.py"), dist_of[name], os.environ["FC_ENGINE"].split(",")[0], name],
                           cwd = HERE, stdout = f, stderr = subprocess.STDOUT, timeout = 1800)
    for name in filter(None, os.environ.get("FC_LONGTASKS", "").split(",")):
        with open(OUT / f"longtasks_{name}.txt", "w") as f:
            subprocess.run([sys.executable, "-u", str(HERE / "fc_longtasks.py"), dist_of[name], os.environ["FC_ENGINE"].split(",")[0]],
                           cwd = HERE, stdout = f, stderr = subprocess.STDOUT, timeout = 1800)


if __name__ == "__main__":
    main()
