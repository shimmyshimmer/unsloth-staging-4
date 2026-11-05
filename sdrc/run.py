"""sd.cpp open-PR recheck driver (one staging job = one prebuilt leg x one ref group).

Builds every ref of the group exactly like that leg of unsloth-sd-prebuilt.yml (ggml-patches applied in glob order,
same cmake defines, GGML_BACKEND_DL + all CPU variants where the leg ships them), optionally packages it with the
ref's own package_bundle.py and runs from the unzipped bundle, then renders SD1.5 with fixed seeds per arm,
interleaved by rep, and compares PNG pixels between arms. Everything lands in ./art.
"""
import argparse, hashlib, json, os, platform, re, shutil, struct, subprocess, sys, time, zlib, zipfile
from pathlib import Path

H = Path(__file__).resolve().parent
ROOT = Path.cwd()
W = ROOT / "w"; W.mkdir(exist_ok=True)
ART = ROOT / "art"; ART.mkdir(exist_ok=True)
FORK = "https://github.com/unslothai/stable-diffusion.cpp"
IS_WIN = os.name == "nt"
IS_MAC = sys.platform == "darwin"
EXE = ".exe" if IS_WIN else ""
SD15_URL = "https://huggingface.co/Comfy-Org/stable-diffusion-1.5/resolve/main/v1-5-pruned-emaonly-fp16.safetensors"
PROMPT = "a photo of a red fox sitting in fresh snow, golden hour, detailed fur"
LOG = open(ART / "driver.log", "a", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n"); LOG.flush()


def sh(cmd, log=None, cwd=None, env=None, check=True, timeout=None):
    say("$", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)))
    with open(log or os.devnull, "a", encoding="utf-8", errors="replace") as fh:
        p = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, env=env, timeout=timeout,
                           stdout=fh if log else subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, encoding="utf-8", errors="replace")
    out = "" if log else (p.stdout or "")
    if check and p.returncode != 0:
        tail = Path(log).read_text(encoding="utf-8", errors="replace")[-4000:] if log else out[-4000:]
        raise RuntimeError(f"rc={p.returncode}: {cmd}\n{tail}")
    return out


def nproc():
    return os.cpu_count() or 4


def host_info():
    info = {"platform": platform.platform(), "machine": platform.machine(), "nproc": nproc(), "python": sys.version}
    try:
        if IS_MAC:
            info["cpu"] = sh("sysctl -n machdep.cpu.brand_string", check=False).strip()
            info["physical"] = sh("sysctl -n hw.physicalcpu", check=False).strip()
        elif IS_WIN:
            info["cpu"] = sh("powershell -NoProfile -Command \"(Get-CimInstance Win32_Processor | Select-Object -First 1 | "
                             "Format-List Name,NumberOfCores,NumberOfLogicalProcessors | Out-String)\"", check=False).strip()
        else:
            info["cpu"] = sh("lscpu", check=False)
            flags = Path("/proc/cpuinfo").read_text().split("flags", 2)
            info["x86_flags"] = sorted(set(re.findall(r"\b(avx512\w*|avx_vnni|amx_\w+|avx2|f16c|fma)\b", Path("/proc/cpuinfo").read_text())))
            info["arm_features"] = sorted(set(re.findall(r"\b(asimddp|i8mm|sve2?|sme\w*|fphp|asimdhp|bf16)\b", Path("/proc/cpuinfo").read_text())))
    except Exception as e:  # noqa: BLE001
        info["err"] = str(e)
    return info


# ---------------------------------------------------------------- source + build

def fetch_fork(prs):
    fork = W / "fork"
    if not fork.exists():
        sh(["git", "clone", "-q", FORK, str(fork)])
    for n in sorted(set(prs)):
        sh(["git", "-C", str(fork), "fetch", "-q", "origin", f"+refs/pull/{n}/head:refs/pr/{n}"])
    return fork


def make_src(fork, name, spec):
    src = W / f"src_{name}"
    if src.exists():
        return src
    first = spec.get("sha") or (f"refs/pr/{spec['pr']}" if "pr" in spec else spec.get("ref", "origin/master"))
    if "merge" in spec:
        first = f"refs/pr/{spec['merge'][0]}"
    sh(["git", "-C", str(fork), "worktree", "add", "-q", "--detach", str(src), first])
    gi = ["git", "-c", "user.name=sdrc", "-c", "user.email=sdrc@example.com", "-C", str(src)]
    merges = []
    for n in spec.get("merge", [])[1:]:
        r = subprocess.run(gi + ["merge", "-q", "--no-edit", f"refs/pr/{n}"], capture_output=True, text=True)
        if r.returncode != 0:
            conflicted = sh(gi + ["diff", "--name-only", "--diff-filter=U"], check=False).split()
            # only documentation may conflict; anything else is a real conflict and fails the ref
            if conflicted and all(c.startswith("docs/") or c.endswith(".md") for c in conflicted):
                for c in conflicted:
                    sh(gi + ["checkout", "--ours", "--", c]); sh(gi + ["add", c])
                sh(gi + ["commit", "-q", "--no-edit"])
                merges.append({"pr": n, "doc_conflicts": conflicted})
            else:
                raise RuntimeError(f"merge #{n} conflicts in code: {conflicted}\n{r.stdout}{r.stderr}")
        else:
            merges.append({"pr": n})
    sh(["git", "-C", str(src), "submodule", "update", "--init", "--depth", "1", "ggml"])
    patches = sorted((src / "scripts/unsloth/ggml-patches").glob("*.patch"))
    for p in patches:
        sh(["git", "-C", str(src / "ggml"), "apply", "--verbose", f"../scripts/unsloth/ggml-patches/{p.name}"],
           log=str(ART / f"patch_{name}.log"))
    meta = {"name": name, "spec": spec, "head": sh(["git", "-C", str(src), "rev-parse", "HEAD"]).strip(),
            "ggml": sh(["git", "-C", str(src / "ggml"), "rev-parse", "HEAD"]).strip(),
            "patches": [p.name for p in patches], "merges": merges}
    (ART / f"src_{name}.json").write_text(json.dumps(meta, indent=1))
    say("source", name, meta["head"], len(patches), "ggml patches")
    return src


def configure_build(name, src, leg, spec, tbo):
    bd = src / "build"
    log = str(ART / f"build_{name}.log")
    defs = ["-DCMAKE_BUILD_TYPE=Release", "-DSD_BUILD_EXAMPLES=ON", "-DSD_SERVER_BUILD_FRONTEND=OFF",
            "-DSD_WEBP=OFF", "-DSD_WEBM=OFF", "-DGGML_NATIVE=OFF"]
    if spec.get("flags41"):
        # PR #41's unsloth-sd-prebuilt.yml: one shared libstable-diffusion, relative rpath, on every leg
        rp = "@loader_path" if IS_MAC else "$ORIGIN"
        defs += ["-DSD_BUILD_SHARED_LIBS=ON"]
        if not IS_WIN:
            defs += ["-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON", f"-DCMAKE_INSTALL_RPATH={rp}"]
    if leg.get("cpu_floor"):
        defs += ["-DSD_BUILD_SHARED_GGML_LIB=ON", "-DBUILD_SHARED_LIBS=ON", "-DGGML_BACKEND_DL=ON", "-DGGML_CPU_ALL_VARIANTS=ON"]
        if IS_MAC and not spec.get("flags41"):
            defs += ["-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON", "-DCMAKE_INSTALL_RPATH=@loader_path"]
    defs += leg.get("defines", [])
    if IS_WIN:
        defs += ["-DCMAKE_CXX_FLAGS=/bigobj"] if leg.get("msvc", True) else []
    if tbo:
        defs += ["-DGGML_BUILD_TESTS=ON"]
    if shutil.which("ccache"):
        defs += ["-DCMAKE_C_COMPILER_LAUNCHER=ccache", "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"]
    t = time.time()
    sh(["cmake", "-S", str(src), "-B", str(bd), "-G", "Ninja"] + defs, log=log)
    targets = ["sd-cli", "sd-server"] + (["test-backend-ops"] if tbo else [])
    jobs = leg.get("jobs") or nproc()
    rc = 0
    try:
        sh(["cmake", "--build", str(bd), "--config", "Release", "-j", str(jobs), "--target"] + targets, log=log)
    except RuntimeError as e:
        rc = 1
        say("BUILD FAILED", name, str(e)[-3000:])
        if any("LAUNCHER=ccache" in d for d in defs):
            # rule out the harness: one clean retry without ccache
            shutil.rmtree(bd, ignore_errors=True)
            defs = [d for d in defs if "LAUNCHER=ccache" not in d]
            try:
                sh(["cmake", "-S", str(src), "-B", str(bd), "-G", "Ninja"] + defs, log=log + ".noccache")
                sh(["cmake", "--build", str(bd), "--config", "Release", "-j", str(jobs), "--target"] + targets, log=log + ".noccache")
                rc = 0
                say("built without ccache", name)
            except RuntimeError as e2:
                say("BUILD FAILED again without ccache", name, str(e2)[-3000:])
    dt = time.time() - t
    text = Path(log).read_text(encoding="utf-8", errors="replace")
    warn = [l for l in text.splitlines() if re.search(r"warning", l, re.I) and re.search(r"iqk|sgemm|conv|util\.cpp|flash_attn|model_manager|ggml-cpu\.c|cpu-feats", l)]
    (ART / f"build_{name}_warnings.txt").write_text("\n".join(warn[:400]), encoding="utf-8")
    say(f"built {name} rc={rc} in {dt:.0f}s, {len(warn)} relevant warnings")
    return {"rc": rc, "seconds": round(dt), "relevant_warnings": len(warn), "bin": str(bd / "bin")}


def package(name, src, binfo, leg, args):
    """The ref's own release steps for this leg: pin (Linux variants), package_bundle.py, gates, unzip."""
    tool = src / "scripts/unsloth"
    bindir = Path(binfo["bin"])
    log = str(ART / f"package_{name}.log")
    if leg.get("cpu_floor") and sys.platform.startswith("linux"):
        sh(["bash", str(tool / "cpu_variants_gate.sh"), "pin", str(bindir)], log=log)
    if IS_MAC and (tool / "assert_macho_minos.sh").exists():
        dt = next((d.split("=")[1] for d in leg.get("defines", []) if "DEPLOYMENT_TARGET" in d), None)
        if dt:
            sh(["bash", str(tool / "assert_macho_minos.sh"), str(bindir), "arm64" if platform.machine() == "arm64" else "x86_64", dt], log=log)
    out = W / f"dist_{name}"; out.mkdir(exist_ok=True)
    lic = src / "LICENSE"
    env = dict(os.environ, BIN_DIR=str(bindir), OUT_DIR=str(out), TAG=f"sdrc-{name}", LABEL=leg["label"],
               COMMIT="0" * 40, SOURCE_REPO="unslothai/stable-diffusion.cpp", LICENSE_FILE=str(lic))
    sh([sys.executable, str(tool / "package_bundle.py")], log=log, env=env)
    z = next(out.glob("*.zip"))
    if leg.get("cpu_floor") and not IS_WIN:
        sh(["bash", str(tool / "cpu_variants_gate.sh"), "check", str(z), leg["cpu_floor"]], log=log)
    ub = W / f"bundle_{name}"
    shutil.rmtree(ub, ignore_errors=True)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(ub)
        names = zf.namelist()
    cli = next(p for p in ub.rglob(f"sd-cli{EXE}"))
    if not IS_WIN:
        for f in cli.parent.iterdir():
            if f.is_file():
                f.chmod(0o755)
    size = z.stat().st_size
    (ART / f"bundle_{name}.txt").write_text(f"zip {z.name} {size}\n" + "\n".join(names), encoding="utf-8")
    say("packaged", name, z.name, size)
    return {"zip": z.name, "zip_bytes": size, "bin": str(cli.parent), "files": len(names)}


# ---------------------------------------------------------------- PNG compare (no third-party deps)

def png_pixels(path):
    data = Path(path).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, w = 8, b"", None
    while pos < len(data):
        ln, typ = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + ln]
        if typ == b"IHDR":
            w, h, depth, ctype = struct.unpack(">IIBB", body[:10])
            assert depth == 8, depth
            ch = {2: 3, 6: 4, 0: 1, 4: 2}[ctype]
        elif typ == b"IDAT":
            idat += body
        pos += 12 + ln
    raw = zlib.decompress(idat)
    stride = w * ch
    out = bytearray(h * stride)
    prev = bytearray(stride)
    i = 0
    for y in range(h):
        f = raw[i]; i += 1
        line = bytearray(raw[i:i + stride]); i += stride
        for x in range(stride):
            a = line[x - ch] if x >= ch else 0
            b = prev[x]
            c = prev[x - ch] if x >= ch else 0
            if f == 1: line[x] = (line[x] + a) & 255
            elif f == 2: line[x] = (line[x] + b) & 255
            elif f == 3: line[x] = (line[x] + ((a + b) >> 1)) & 255
            elif f == 4:
                p = a + b - c; pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[x] = (line[x] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        out[y * stride:(y + 1) * stride] = line
        prev = line
    return w, h, ch, bytes(out)


def pix_diff(a, b):
    try:
        wa, ha, ca, pa = png_pixels(a); wb, hb, cb, pb = png_pixels(b)
    except Exception as e:  # noqa: BLE001
        return {"err": str(e)}
    if (wa, ha, ca) != (wb, hb, cb):
        return {"err": "shape"}
    n = len(pa); mx = 0; se = 0; neq = 0
    for x, y in zip(pa, pb):
        d = x - y
        if d:
            neq += 1; d = abs(d); mx = max(mx, d); se += d * d
    mse = se / n
    psnr = 99.0 if mse == 0 else 10 * __import__("math").log10(255 * 255 / mse)
    return {"max_abs": mx, "frac_diff": round(neq / n, 5), "psnr": round(psnr, 2)}


# ---------------------------------------------------------------- runs

PAT = {"te": r"get_learned_condition completed, taking ([\d.]+)\s*s", "sampling": r"sampling completed, taking ([\d.]+)\s*s",
       "vae": r"decode_first_stage completed, taking ([\d.]+)\s*s", "generate": r"generate_image completed in ([\d.]+)\s*s",
       "load": r"loading tensors completed, taking ([\d.]+)\s*s"}
DECIDE = r"(conv[23]d direct auto: .*|flash attention auto: .*|loaded \w+ backend from .*|.*[Tt]hreads?[ =:]+\d+.*|.*using device.*|.*ggml_vulkan: .*|.*ggml_metal_device_init: GPU name.*|.*has simdgroup.*|.*repack.*|.*mmap.*)"


def peak_rss_run(cmd, env, log, timeout):
    t = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as fh:
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill(); rc = -999
    wall = time.time() - t
    peak = None
    if IS_WIN:
        try:
            import ctypes
            from ctypes import wintypes
            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
            if ctypes.WinDLL("psapi").GetProcessMemoryInfo(wintypes.HANDLE(int(p._handle)), ctypes.byref(pmc), pmc.cb):
                peak = pmc.PeakWorkingSetSize / 2**20
        except Exception:  # noqa: BLE001
            peak = None
    return rc, wall, peak


def rss_wrapper(cmd):
    """Unix: run through a fresh python so RUSAGE_CHILDREN max-rss is this run's alone."""
    if IS_WIN:
        return cmd
    code = ("import resource,subprocess,sys;r=subprocess.call(sys.argv[1:]);"
            "m=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss;"
            "print('SDRC_MAXRSS_MB=%.1f'%(m/1048576 if sys.platform=='darwin' else m/1024),flush=True);sys.exit(r)")
    return [sys.executable, "-c", code] + cmd


def pinned_bindir(bindir, variant, tag):
    """Copy of bindir holding only one ggml-cpu variant, so ggml cannot pick another one."""
    d = W / f"pin_{tag}_{variant}"
    if not d.exists():
        shutil.copytree(bindir, d, symlinks=True)
        for f in d.iterdir():
            if "ggml-cpu-" in f.name and variant not in f.name:
                f.unlink()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", required=True); ap.add_argument("--group", required=True)
    a = ap.parse_args()
    plan = json.loads((H / "plan.json").read_text())
    leg = dict(plan["legs"][a.leg]); leg["name"] = a.leg
    grp = plan["groups"][a.group]
    res = {"leg": a.leg, "group": a.group, "host": host_info(), "builds": {}, "packages": {}, "runs": [], "compare": []}
    say(json.dumps(res["host"], indent=1)[:3000])
    refs = {k: v for k, v in grp["refs"].items() if a.leg not in v.get("skip_legs", [])}
    prs = [n for v in refs.values() for n in ([v["pr"]] if "pr" in v else v.get("merge", []))]
    fork = fetch_fork(prs)
    tbo_on = bool(leg.get("tbo")) and bool(grp.get("tbo", {}).get(a.leg))
    srcs, bins = {}, {}
    for name, spec in refs.items():
        try:
            srcs[name] = make_src(fork, name, spec)
        except Exception as e:  # noqa: BLE001
            say("SOURCE FAILED", name, e); res["builds"][name] = {"rc": 2, "err": str(e)[-2000:]}; continue
        b = configure_build(name, srcs[name], leg, spec, tbo_on)
        res["builds"][name] = b
        if b["rc"] == 0:
            bins[name] = b["bin"]
            if spec.get("package") and leg.get("package", True):
                try:
                    res["packages"][name] = package(name, srcs[name], b, leg, a)
                    bins[name] = res["packages"][name]["bin"]
                except Exception as e:  # noqa: BLE001
                    say("PACKAGE FAILED", name, e); res["packages"][name] = {"rc": 1, "err": str(e)[-3000:]}
    (ART / "results.json").write_text(json.dumps(res, indent=1))

    # models
    mdir = W / "models"; mdir.mkdir(exist_ok=True)
    sd15 = mdir / "sd15.safetensors"
    arms = [x for x in grp["arms"] if x["ref"] in bins and (not x.get("legs") or a.leg in x["legs"])
            and a.leg not in x.get("skip_legs", []) and (x.get("gpu") is None or x["gpu"] in leg.get("gpu", []))]
    if arms and not sd15.exists():
        for i in range(5):
            r = subprocess.run(["curl", "-sSfL", "--retry", "3", "-o", str(sd15), SD15_URL])
            if r.returncode == 0:
                break
            time.sleep(20)
    models = {"fp16": ["-m", str(sd15)], "q8": ["-m", str(sd15), "--type", "q8_0"], "q3k": ["-m", str(sd15), "--type", "q3_K"]}
    if any(x["model"].startswith("q4kg") for x in arms):
        g = mdir / "sd15_q4k.gguf"
        conv_bin = bins.get(grp.get("convert_with", next(iter(bins))))
        if not g.exists():
            sh([str(Path(conv_bin) / f"sd-cli{EXE}"), "-M", "convert", "-m", str(sd15), "--type", "q4_K", "-o", str(g)],
               log=str(ART / "convert.log"), check=False)
        models["q4kg"] = ["-m", str(g)]
    variants = {}
    t_cores = str(leg.get("threads") or nproc())
    outdir = W / "out"; outdir.mkdir(exist_ok=True)
    reps = max(x.get("reps", 1) for x in arms) if arms else 0
    for rep in range(reps):
        for arm in arms:
            if rep >= arm.get("reps", 1):
                continue
            bindir = Path(bins[arm["ref"]])
            if arm.get("pin") and arm["pin"] in variants and variants[arm["pin"]]:
                bindir = pinned_bindir(bindir, variants[arm["pin"]], arm["ref"])
            cli = bindir / f"sd-cli{EXE}"
            size = str(arm.get("size", grp.get("size", 256)))
            png = outdir / f"{arm['id']}_r{rep}.png"
            cmd = [str(cli)] + models[arm["model"].split(".")[0]] + ["-p", PROMPT, "-W", size, "-H", size,
                   "--steps", str(arm.get("steps", 4)), "--seed", "42", "--cfg-scale", "7", "-o", str(png), "-v"]
            if arm.get("threads", "fixed") == "fixed":
                cmd += ["-t", t_cores]
            if not arm.get("gpu"):
                cmd += ["--backend", "cpu"]
            cmd += arm.get("args", [])
            env = dict(os.environ); env.update(arm.get("env", {}))
            log = ART / "logs" / f"{arm['id']}_r{rep}.log"; log.parent.mkdir(exist_ok=True)
            say(f"run {arm['id']} rep {rep}: {' '.join(cmd[1:])}  env={arm.get('env', {})}")
            rc, wall, peak = peak_rss_run(rss_wrapper(cmd), env, str(log), arm.get("timeout", leg.get("timeout", 2400)))
            text = log.read_text(encoding="utf-8", errors="replace")
            rec = {"id": arm["id"], "ref": arm["ref"], "rep": rep, "rc": rc, "wall": round(wall, 2), "env": arm.get("env", {}),
                   "args": arm.get("args", []), "model": arm["model"], "gpu": arm.get("gpu")}
            for k, p in PAT.items():
                m = re.findall(p, text)
                if m:
                    rec[k] = round(sum(float(x) for x in m), 3) if k == "load" else float(m[-1])
            m = re.search(r"SDRC_MAXRSS_MB=([\d.]+)", text)
            rec["peak_rss_mb"] = float(m.group(1)) if m else (round(peak, 1) if peak else None)
            m = re.search(r"loaded CPU backend from (\S+)", text)
            if m:
                v = re.sub(r"^(lib)?ggml-cpu-|\.(so|dll|dylib)$", "", os.path.basename(m.group(1)))
                rec["variant"] = v
                variants.setdefault(arm["id"], v)
            rec["decisions"] = sorted(set(x.strip()[:220] for x in re.findall(DECIDE, text)))[:40]
            if rc == 0 and png.exists():
                rec["md5"] = hashlib.md5(png.read_bytes()).hexdigest()
            else:
                rec["tail"] = text[-2500:]
            say(f"  -> rc={rc} wall={wall:.1f}s sampling={rec.get('sampling')} vae={rec.get('vae')} variant={rec.get('variant')} "
                f"md5={rec.get('md5')} rss={rec.get('peak_rss_mb')}")
            res["runs"].append(rec)
            (ART / "results.json").write_text(json.dumps(res, indent=1))

    # comparisons: [a, b, expect] with expect in identical / close / differ / any
    byid = {}
    for r in res["runs"]:
        byid.setdefault(r["id"], []).append(r)
    for a_id, b_id, expect in grp.get("compare", []):
        ra, rb = byid.get(a_id), byid.get(b_id)
        if not ra or not rb:
            continue
        pa = outdir / f"{a_id}_r0.png"; pb = outdir / f"{b_id}_r0.png"
        same = ra[0].get("md5") is not None and ra[0].get("md5") == rb[0].get("md5")
        d = {"a": a_id, "b": b_id, "expect": expect, "identical": same}
        if not same and pa.exists() and pb.exists():
            d.update(pix_diff(pa, pb))
        def med(rs, k):
            v = sorted(x[k] for x in rs if x.get(k) is not None)
            return v[len(v) // 2] if v else None
        for k in ("sampling", "vae", "generate", "peak_rss_mb"):
            d[f"{k}_a"], d[f"{k}_b"] = med(ra, k), med(rb, k)
        ok = {"identical": same, "close": same or (d.get("psnr", 0) >= grp.get("close_psnr", 30)), "differ": not same, "any": True}[expect]
        d["verdict"] = "OK" if ok and ra[0]["rc"] == 0 and rb[0]["rc"] == 0 else "FAIL"
        # consistency across reps of the same arm (nondeterminism check)
        d["a_reps_same"] = len({x.get("md5") for x in ra}) == 1
        d["b_reps_same"] = len({x.get("md5") for x in rb}) == 1
        res["compare"].append(d)
        say("compare", json.dumps(d))

    # test-backend-ops
    if tbo_on:
        res["tbo"] = {}
        for name in grp["tbo"][a.leg].get("refs", []):
            if name not in res["builds"] or res["builds"][name]["rc"]:
                continue
            exe = next((p for p in (srcs[name] / "build").rglob(f"test-backend-ops{EXE}")), None)
            if not exe:
                continue
            for spec in grp["tbo"][a.leg]["runs"]:
                mode, be, op = spec.split(":")[:3]
                extra = spec.split(":", 3)[3] if spec.count(":") >= 3 else ""
                log = ART / f"tbo_{name}_{mode}_{be}_{op}.log"
                cmd = [str(exe), mode, "-b", be, "-o", op] + (["-p", extra] if extra else [])
                r = subprocess.run(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, timeout=3600)
                text = log.read_text(errors="replace")
                m = re.search(r"(\d+)/(\d+) tests passed", text)
                res["tbo"][f"{name}:{mode}:{be}:{op}"] = {"rc": r.returncode, "passed": m.group(0) if m else None,
                                                          "fail_lines": [l for l in text.splitlines() if "FAIL" in l][:20]}
                say("tbo", name, spec, r.returncode, m.group(0) if m else "")
    (ART / "results.json").write_text(json.dumps(res, indent=1))
    bad = [c for c in res["compare"] if c["verdict"] != "OK"]
    bfail = [k for k, v in res["builds"].items() if v.get("rc")]
    rfail = [r["id"] for r in res["runs"] if r["rc"] != 0]
    say(f"SUMMARY builds_failed={bfail} runs_failed={sorted(set(rfail))} compare_fail={[(c['a'], c['b']) for c in bad]}")


if __name__ == "__main__":
    main()
