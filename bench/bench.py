"""Chat code-fence colour bench: frames uncoloured during a wheel scroll, settle time after stop,
streaming colour lag, open-to-colour, and main-thread cost, per arm and engine, reps interleaved.

usage: bench.py --arms base=DIR,pr=DIR,head=DIR [--engines chromium,webkit] [--reps 3]
                [--throttle 1,4] [--turns 60] [--out FILE]
"""

import argparse
import functools
import http.server
import json
import statistics
import sys
import threading
import time

from playwright.sync_api import sync_playwright

MAC_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/18.0 Safari/605.1.15"
)
MAC_PLATFORM = "Object.defineProperty(Navigator.prototype, 'platform', { get: () => 'MacIntel' });"


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def serve(directory):
    handler = functools.partial(Quiet, directory = directory)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target = server.serve_forever, daemon = True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def wait_until(page, js, timeout_s = 60.0, poll_ms = 50):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if page.evaluate(js):
            return True
        page.wait_for_timeout(poll_ms)
    return False


SETTLED_JS = """() => { const f = window.__fc.snapshot(); return f && f.inView > 0 && f.uncolouredBlocks === 0; }"""


def settle_time(page, start_js_time, timeout_s = 10.0):
    """ms from start_js_time until a frame shows every in-view block coloured, or None."""
    t = page.evaluate(
        """([start, timeoutMs]) => new Promise((resolve) => {
          const tick = () => {
            const f = window.__fc.snapshot();
            const now = performance.now();
            if (f && f.inView > 0 && f.uncolouredBlocks === 0) { resolve(now - start); return; }
            if (now - start > timeoutMs) { resolve(null); return; }
            requestAnimationFrame(tick);
          };
          requestAnimationFrame(tick);
        })""",
        [start_js_time, timeout_s * 1000],
    )
    return t


PERF_KEYS = ("LayoutDuration", "RecalcStyleDuration", "ScriptDuration", "TaskDuration", "LayoutCount")


def perf(cdp):
    if cdp is None:
        return None
    return {m["name"]: m["value"] for m in cdp.send("Performance.getMetrics")["metrics"] if m["name"] in PERF_KEYS}


def perf_delta(a, b):
    if a is None or b is None:
        return None
    return {k: round((b[k] - a[k]) * (1 if k == "LayoutCount" else 1000), 1) for k in PERF_KEYS}


def run_one(pw, engine, url, throttle, turns, ticks, css = None):
    browser = getattr(pw, engine).launch()
    ctx_args = {"viewport": {"width": 1400, "height": 1000}}
    if engine == "webkit":
        ctx_args["user_agent"] = MAC_UA
    context = browser.new_context(**ctx_args)
    if engine == "webkit":
        context.add_init_script(MAC_PLATFORM)
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(f"{url}/smoke-fence-colour.html")
    page.wait_for_function("() => window.__fc && document.querySelector('.aui-thread-viewport')", timeout = 60000)
    if css:
        page.add_style_tag(content = css)
    cdp = None
    if engine == "chromium":
        cdp = context.new_cdp_session(page)
        cdp.send("Performance.enable")
        if throttle != 1:
            cdp.send("Emulation.setCPUThrottlingRate", {"rate": throttle})
    out = {"engine": engine, "throttle": throttle}
    p0 = perf(cdp)

    # (d) first open: seed and wait until the view is coloured.
    t0 = page.evaluate("() => performance.now()")
    page.evaluate(f"() => window.__fc.seed({turns})")
    out["seed_to_colour_ms"] = settle_time(page, t0, 20)
    page.wait_for_timeout(1500)
    out["fences"] = page.evaluate("() => window.__fc.fenceCount()")

    # (d) reopen: unmount then mount the thread (opening a long chat that was opened before).
    page.evaluate("() => window.__fc.close()")
    page.wait_for_timeout(500)
    t0 = page.evaluate("() => { const t = performance.now(); window.__fc.open(); return t; }")
    out["open_to_colour_ms"] = settle_time(page, t0, 20)
    page.wait_for_timeout(1500)
    t1 = page.evaluate("() => performance.now()")
    out["open_cost"] = page.evaluate(f"() => window.__fc.stats({t0}, {t1})")
    p1 = perf(cdp)
    out["perf_seed_open"] = perf_delta(p0, p1)

    # (a, b, e) fast wheel scroll from the bottom upward.
    page.evaluate("() => window.__fc.toBottom()")
    page.wait_for_timeout(1200)
    box = page.evaluate("() => window.__fc.viewportBox()")
    page.mouse.move(box["x"] + box["w"] * 0.5, box["y"] + box["h"] * 0.5)
    page.evaluate("() => window.__fc.startSampling()")
    s0 = page.evaluate("() => performance.now()")
    for _ in range(ticks):
        page.mouse.wheel(0, -120)
        page.wait_for_timeout(8)
    stop = page.evaluate("() => performance.now()")
    frames = page.evaluate("() => window.__fc.stopSampling()")
    out["scroll_settle_ms"] = settle_time(page, stop, 10)
    page.wait_for_timeout(800)
    s1 = page.evaluate("() => performance.now()")
    scroll_frames = [f for f in frames if f["inView"] > 0]
    out["scroll_frames"] = len(scroll_frames)
    out["scroll_uncoloured_frames"] = sum(1 for f in scroll_frames if f["uncolouredBlocks"] > 0)
    vis = sum(f["visibleLines"] for f in scroll_frames)
    out["scroll_uncoloured_line_share"] = (
        sum(f["uncolouredLines"] for f in scroll_frames) / vis if vis else 0.0
    )
    out["scroll_kind"] = {k: sum(f[k] for f in scroll_frames) for k in ("deferred", "plain", "partial")}
    out["scroll_cost"] = page.evaluate(f"() => window.__fc.stats({s0}, {stop})")
    out["scroll_after_cost"] = page.evaluate(f"() => window.__fc.stats({stop}, {s1})")
    out["scroll_top_end"] = page.evaluate("() => window.__fc.viewportBox().scrollTop")
    p2 = perf(cdp)
    out["perf_scroll"] = perf_delta(p1, p2)

    # width change: the column re-wraps, every laid-out line is laid out again.
    r0 = page.evaluate("() => performance.now()")
    for width in (1000, 1400, 1000, 1400):
        page.set_viewport_size({"width": width, "height": 1000})
        page.wait_for_timeout(600)
    r1 = page.evaluate("() => performance.now()")
    out["resize_cost"] = page.evaluate(f"() => window.__fc.stats({r0}, {r1})")
    p3 = perf(cdp)
    out["perf_resize"] = perf_delta(p2, p3)

    # typing: key to next frame.
    page.click(".aui-composer-input")
    k0 = page.evaluate("""() => {
      window.__keys = [];
      document.addEventListener('keydown', () => { const t = performance.now();
        requestAnimationFrame(() => window.__keys.push(performance.now() - t)); }, { capture: true });
      return performance.now(); }""")
    page.keyboard.type("the quick brown fox jumps over", delay = 40)
    page.wait_for_timeout(300)
    keys = page.evaluate("() => window.__keys")
    k1 = page.evaluate("() => performance.now()")
    out["type_key_to_frame_p50"] = statistics.median(keys) if keys else None
    out["type_key_to_frame_max"] = max(keys) if keys else None
    out["type_cost"] = page.evaluate(f"() => window.__fc.stats({k0}, {k1})")
    page.keyboard.press("Control+A")
    page.keyboard.press("Delete")

    # (c) streaming fence.
    page.evaluate("() => window.__fc.toBottom()")
    page.wait_for_timeout(600)
    p4 = perf(cdp)
    st0 = page.evaluate("() => performance.now()")
    page.evaluate("() => window.__fc.stream({ lines: 70, chunkChars: 24, gapMs: 16 })")
    ok = wait_until(page, "() => window.__fc.streamDone()", 120)
    st1 = page.evaluate("() => performance.now()")
    out["stream_ok"] = ok
    out["stream"] = page.evaluate("() => window.__fc.streamResult()")
    out["stream_cost"] = page.evaluate(f"() => window.__fc.stats({st0}, {st1})")
    out["perf_stream"] = perf_delta(p4, perf(cdp))
    out["errors"] = errors[:5]
    if cdp is not None:
        cdp.send("Emulation.setCPUThrottlingRate", {"rate": 1})
    browser.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required = True)
    ap.add_argument("--engines", default = "chromium,webkit")
    ap.add_argument("--reps", type = int, default = 3)
    ap.add_argument("--throttle", default = "1,4")
    ap.add_argument("--turns", type = int, default = 60)
    ap.add_argument("--ticks", type = int, default = 200)
    ap.add_argument("--out", default = "fence_colour_bench.json")
    ap.add_argument("--record-builds", default = "")
    args = ap.parse_args()
    arms = dict(a.split("=", 1) for a in args.arms.split(","))
    css = {}
    for name, d in list(arms.items()):
        if "@" in d:
            d, path = d.split("@", 1)
            arms[name] = d
            css[name] = open(path).read()
    servers = {name: serve(d) for name, d in arms.items()}
    if args.record_builds:
        import os, re, urllib.request
        shas = json.loads(os.environ.get("FC_SHAS", "{}"))
        assets = lambda html: sorted(re.findall(r'(?:src|href)="(/assets/[^"]+)"', html))
        builds = []
        for name, d in arms.items():
            built = assets(open(os.path.join(d, "smoke-fence-colour.html")).read())
            served = assets(urllib.request.urlopen(servers[name][1] + "/smoke-fence-colour.html", timeout = 10).read().decode())
            builds.append({"arm": name, "sha": shas.get(name), "built": built, "served": served, "match": built == served and bool(built)})
            print(f"build {name} sha={shas.get(name)} assets={len(built)} built==served {builds[-1]['match']}", flush = True)
        with open(args.record_builds, "w") as f:
            json.dump(builds, f, indent = 1)
        if not all(b["match"] for b in builds):
            raise SystemExit("an arm served something other than its own build")
    results = []
    with sync_playwright() as pw:
        for rep in range(args.reps):
            for engine in args.engines.split(","):
                throttles = [float(t) for t in args.throttle.split(",")] if engine == "chromium" else [1.0]
                for throttle in throttles:
                    order = list(arms) if rep % 2 == 0 else list(reversed(list(arms)))
                    for name in order:
                        t = time.monotonic()
                        r = None
                        for attempt in range(2):
                            try:
                                r = run_one(pw, engine, servers[name][1], throttle, args.turns, args.ticks, css.get(name))
                                break
                            except Exception as exc:  # a stalled host times a page out; retry once
                                print(f"rep{rep} {engine} x{throttle:g} {name}: attempt {attempt + 1} failed: {str(exc)[:120]}", flush = True)
                        if r is None:
                            continue
                        r.update(arm = name, rep = rep)
                        results.append(r)
                        print(
                            f"rep{rep} {engine} x{throttle:g} {name}: "
                            f"fences={r['fences']} scroll_unc={r['scroll_uncoloured_frames']}/{r['scroll_frames']} "
                            f"settle={r['scroll_settle_ms']} open={r['open_to_colour_ms']} "
                            f"stream_unc={r['stream']['uncolouredFrames']}/{r['stream']['frames']} "
                            f"lag95={r['stream']['lagP95']:.0f} never={r['stream']['neverColoured']} "
                            f"scrollLong={r['scroll_cost']['longestFrameMs']:.0f} openLT={r['open_cost']['longTaskMs']:.0f} "
                            f"kind={r['scroll_kind']} err={len(r['errors'])} ({time.monotonic() - t:.0f}s)",
                            flush = True,
                        )
                        with open(args.out, "w") as f:
                            json.dump(results, f, indent = 1)
    for _, (server, _) in servers.items():
        server.shutdown()


if __name__ == "__main__":
    sys.exit(main())
