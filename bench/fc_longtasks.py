"""Find what runs in long callbacks (any engine): wraps timers, rAF, rIC, message handlers and
records callbacks over 100 ms with the stack that scheduled them. usage: fc_longtasks.py DIST ENGINE"""

import sys

from bench import MAC_PLATFORM, MAC_UA, serve, settle_time
from playwright.sync_api import sync_playwright

WRAP = r"""
(() => {
  window.__long = [];
  const record = (kind, stack, ms, extra) => { if (ms > 16) window.__long.push({ kind, ms, at: performance.now(), stack, extra }); };
  const wrap = (fn, kind) => {
    const stack = new Error().stack;
    return function (...args) { const t = performance.now(); try { return fn.apply(this, args); } finally { record(kind, stack, performance.now() - t); } };
  };
  for (const name of ['setTimeout', 'requestAnimationFrame', 'requestIdleCallback', 'queueMicrotask']) {
    const real = window[name]; if (typeof real !== 'function') continue;
    window[name] = function (cb, ...rest) { return real.call(window, typeof cb === 'function' ? wrap(cb, name) : cb, ...rest); };
  }
  const desc = Object.getOwnPropertyDescriptor(Worker.prototype, 'onmessage');
  Object.defineProperty(Worker.prototype, 'onmessage', { set(fn) { desc.set.call(this, fn ? wrap(fn, 'worker.onmessage') : fn); }, get() { return desc.get.call(this); } });
  const pdesc = Object.getOwnPropertyDescriptor(MessagePort.prototype, 'onmessage');
  Object.defineProperty(MessagePort.prototype, 'onmessage', { set(fn) { pdesc.set.call(this, fn ? wrap(fn, 'port.onmessage') : fn); }, get() { return pdesc.get.call(this); } });
  const add = EventTarget.prototype.addEventListener;
  EventTarget.prototype.addEventListener = function (type, fn, opts) {
    if (typeof fn === 'function' && ['scroll', 'wheel', 'message', 'keydown', 'input'].includes(type)) {
      const w = fn.__w || (fn.__w = wrap(fn, 'event:' + type));
      return add.call(this, type, w, opts);
    }
    return add.call(this, type, fn, opts);
  };
  const remove = EventTarget.prototype.removeEventListener;
  EventTarget.prototype.removeEventListener = function (type, fn, opts) { return remove.call(this, type, (fn && fn.__w) || fn, opts); };
})();
"""


def main():
    dist, engine = sys.argv[1], sys.argv[2]
    server, url = serve(dist)
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        args = {"viewport": {"width": 1400, "height": 1000}}
        if engine == "webkit":
            args["user_agent"] = MAC_UA
        context = browser.new_context(**args)
        if engine == "webkit":
            context.add_init_script(MAC_PLATFORM)
        context.add_init_script(WRAP)
        page = context.new_page()
        page.goto(f"{url}/smoke-fence-colour.html")
        page.wait_for_function("() => window.__fc && document.querySelector('.aui-thread-viewport')")
        t0 = page.evaluate("() => performance.now()")
        page.evaluate("() => window.__fc.seed(60)")
        print("seed", settle_time(page, t0, 20))
        page.wait_for_timeout(2000)
        page.evaluate("() => window.__fc.toBottom()")
        page.wait_for_timeout(1200)
        s0 = page.evaluate("() => performance.now()")
        box = page.evaluate("() => window.__fc.viewportBox()")
        page.mouse.move(box["x"] + box["w"] * 0.5, box["y"] + box["h"] * 0.5)
        for _ in range(200):
            page.mouse.wheel(0, -120)
            page.wait_for_timeout(8)
        stop = page.evaluate("() => performance.now()")
        print("settle", settle_time(page, stop, 10))
        page.wait_for_timeout(800)
        page.click(".aui-composer-input")
        k0 = page.evaluate("""() => { window.__keys = [];
          document.addEventListener('keydown', () => { const t = performance.now();
            requestAnimationFrame(() => window.__keys.push([t, performance.now() - t])); }, { capture: true });
          return performance.now(); }""")
        page.keyboard.type("the quick brown fox jumps over", delay = 40)
        page.wait_for_timeout(500)
        k1 = page.evaluate("() => performance.now()")
        keys = page.evaluate("() => window.__keys")
        worst = max(keys, key = lambda k: k[1])
        print("worst key lag", round(worst[1]), "at", round(worst[0] - k0), "ms into typing; lags", [round(k[1]) for k in keys])
        import collections
        agg = collections.defaultdict(lambda: [0, 0.0])
        for item in page.evaluate("() => window.__long"):
            phase = "seed" if item["at"] < s0 else "scroll" if item["at"] < stop else "typing" if item["at"] >= k0 else "after"
            frames = [l.strip() for l in (item["stack"] or "").split("\n")[1:8] if "web-inspector" not in l and "bootstrap" not in l]
            site = " < ".join(f.split("@")[0] or "(anon)" for f in frames[:3])
            a = agg[(phase, item["kind"], site)]
            a[0] += 1; a[1] += item["ms"]
        for (phase, kind, site), (n, ms) in sorted(agg.items(), key = lambda kv: -kv[1][1])[:25]:
            print(f"{phase:7s} {ms:7.0f} ms in {n:3d} x {kind:22s} {site}")
        return
        for item in []:
            phase = None
            stack = "\n      ".join([l.strip() for l in (item["stack"] or "").split("\n")[1:7]])
            print(f"{phase} {item['kind']} {item['ms']:.0f}ms\n      {stack}")
        browser.close()
    server.shutdown()


if __name__ == "__main__":
    main()
