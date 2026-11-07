"""Mutation log during a fast wheel scroll: which nodes and attributes change mid-scroll.
usage: fc_mutlog.py DIST ENGINE [label]"""

import json
import sys

from bench import MAC_PLATFORM, MAC_UA, serve, settle_time
from playwright.sync_api import sync_playwright

WRAP_ALL = r"""
(() => {
  window.__js = { on: false, sites: {} };
  const rec = (kind, stack, ms) => {
    if (!window.__js.on) return;
    const frames = (stack || '').split('\n').slice(1).filter((l) => !l.includes('web-inspector') && !l.includes('bootstrap'));
    const site = kind + ' ' + frames.slice(0, 2).map((l) => l.trim().split('@')[0] || '(anon)').join(' < ');
    const e = window.__js.sites[site] || (window.__js.sites[site] = [0, 0]);
    e[0] += 1; e[1] += ms;
  };
  const wrap = (fn, kind) => { const stack = new Error().stack; return function (...a) { const t = performance.now(); try { return fn.apply(this, a); } finally { rec(kind, stack, performance.now() - t); } }; };
  for (const name of ['setTimeout', 'requestAnimationFrame', 'requestIdleCallback']) {
    const real = window[name]; if (typeof real !== 'function') continue;
    window[name] = function (cb, ...rest) { return real.call(window, typeof cb === 'function' ? wrap(cb, name) : cb, ...rest); };
  }
  const add = EventTarget.prototype.addEventListener;
  EventTarget.prototype.addEventListener = function (type, fn, opts) {
    if (typeof fn === 'function') { const w = fn.__w || (fn.__w = wrap(fn, 'event:' + type)); return add.call(this, type, w, opts); }
    return add.call(this, type, fn, opts);
  };
  const remove = EventTarget.prototype.removeEventListener;
  EventTarget.prototype.removeEventListener = function (type, fn, opts) { return remove.call(this, type, (fn && fn.__w) || fn, opts); };
  for (const Ctor of [window.ResizeObserver, window.IntersectionObserver, window.MutationObserver]) {
    if (!Ctor) continue;
    const name = Ctor.name;
    window[name] = class extends Ctor { constructor(cb, o) { super(wrap(cb, name), o); } };
  }
  const pdesc = Object.getOwnPropertyDescriptor(MessagePort.prototype, 'onmessage');
  Object.defineProperty(MessagePort.prototype, 'onmessage', { set(fn) { pdesc.set.call(this, fn ? wrap(fn, 'port.onmessage') : fn); }, get() { return pdesc.get.call(this); } });
})();
"""

INSTALL = r"""() => {
  const counts = {};
  const key = (n) => {
    if (!n || n.nodeType !== 1) return n ? `#${n.nodeName}` : '?';
    const ds = n.getAttribute('data-streamdown') || n.getAttribute('data-slot') || '';
    const block = n.closest('[data-streamdown="code-block"]') ? 'in-block ' : '';
    return `${block}${n.tagName.toLowerCase()}${ds ? '[' + ds + ']' : ''}`;
  };
  const bump = (k, by = 1) => { counts[k] = (counts[k] || 0) + by; };
  window.__mo = new MutationObserver((list) => {
    for (const m of list) {
      if (m.type === 'attributes') bump(`attr ${m.attributeName} on ${key(m.target)}`);
      else if (m.type === 'characterData') bump(`text in ${key(m.target.parentElement)}`);
      else {
        if (m.addedNodes.length) bump(`add under ${key(m.target)}`, m.addedNodes.length);
        if (m.removedNodes.length) bump(`remove under ${key(m.target)}`, m.removedNodes.length);
      }
    }
  });
  window.__mo.observe(document.querySelector('.aui-thread-root') || document.body,
    { subtree: true, attributes: true, childList: true, characterData: true });
  window.__counts = counts;
}"""


def main():
    dist, engine = sys.argv[1], sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else dist.rsplit("/", 1)[-1]
    server, url = serve(dist)
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        args = {"viewport": {"width": 1400, "height": 1000}}
        if engine == "webkit":
            args["user_agent"] = MAC_UA
        context = browser.new_context(**args)
        if engine == "webkit":
            context.add_init_script(MAC_PLATFORM)
        context.add_init_script(WRAP_ALL)
        page = context.new_page()
        page.goto(f"{url}/smoke-fence-colour.html")
        page.wait_for_function("() => window.__fc && document.querySelector('.aui-thread-viewport')")
        t0 = page.evaluate("() => performance.now()")
        page.evaluate("() => window.__fc.seed(60)")
        settle_time(page, t0, 20)
        page.wait_for_timeout(3000)
        page.evaluate("() => window.__fc.toBottom()")
        page.wait_for_timeout(1500)
        box = page.evaluate("() => window.__fc.viewportBox()")
        page.mouse.move(box["x"] + box["w"] * 0.5, box["y"] + box["h"] * 0.5)
        page.evaluate(INSTALL)
        page.evaluate("() => { window.__js.on = true; window.__js.sites = {}; }")
        s0 = page.evaluate("() => performance.now()")
        for _ in range(200):
            page.mouse.wheel(0, -120)
            page.wait_for_timeout(8)
        s1 = page.evaluate("() => performance.now()")
        page.evaluate("() => { window.__js.on = false; }")
        counts = page.evaluate("() => { window.__mo.takeRecords(); return window.__counts; }")
        frames = page.evaluate(f"() => window.__fc.stats({s0}, {s1})")
        sites = page.evaluate("() => window.__js.sites")
        js_total = sum(v[1] for v in sites.values())
        js_top = sorted(((k, v[0], round(v[1])) for k, v in sites.items()), key = lambda x: -x[2])[:15]
        top = sorted(counts.items(), key = lambda kv: -kv[1])[:25]
        print(json.dumps({"label": label, "engine": engine, "scroll_ms": round(s1 - s0), "frames": frames,
                          "js_total_ms": round(js_total), "js_top": js_top,
                          "mutations": sum(counts.values()), "mutation_top": top}, indent = 1))
        browser.close()
    server.shutdown()


if __name__ == "__main__":
    main()
