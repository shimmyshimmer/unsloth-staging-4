"""Median (min-max) per arm/engine/throttle from bench JSON files. usage: summarise.py FILE..."""
import json, statistics, sys, collections

METRICS = [
    ("scroll uncoloured frames", lambda r: r["scroll_uncoloured_frames"] / max(1, r["scroll_frames"]) * 100, "%"),
    ("scroll stop -> all coloured", lambda r: r["scroll_settle_ms"] if r["scroll_settle_ms"] is not None else 10000, "ms"),
    ("stream frames w/ uncoloured line", lambda r: r["stream"]["uncolouredFrames"] / max(1, r["stream"]["frames"]) * 100, "%"),
    ("stream line colour lag p95", lambda r: r["stream"]["lagP95"], "ms"),
    ("seed -> coloured", lambda r: r["seed_to_colour_ms"] if r["seed_to_colour_ms"] is not None else 20000, "ms"),
    ("reopen -> coloured", lambda r: r["open_to_colour_ms"] if r["open_to_colour_ms"] is not None else 20000, "ms"),
    ("open longest frame", lambda r: r["open_cost"]["longestFrameMs"], "ms"),
    ("open long-task total", lambda r: r["open_cost"]["longTaskMs"] if r["open_cost"]["longTaskSupported"] else r["open_cost"]["frameTimeOver50Ms"], "ms"),
    ("scroll longest frame", lambda r: r["scroll_cost"]["longestFrameMs"], "ms"),
    ("scroll long-task total", lambda r: r["scroll_cost"]["longTaskMs"] if r["scroll_cost"]["longTaskSupported"] else r["scroll_cost"]["frameTimeOver50Ms"], "ms"),
    ("scroll+after long-task total", lambda r: sum((c["longTaskMs"] if c["longTaskSupported"] else c["frameTimeOver50Ms"]) for c in (r["scroll_cost"], r["scroll_after_cost"])), "ms"),
    ("after-stop longest frame", lambda r: r["scroll_after_cost"]["longestFrameMs"], "ms"),
    ("stream longest frame", lambda r: r["stream_cost"]["longestFrameMs"], "ms"),
    ("stream long-task total", lambda r: r["stream_cost"]["longTaskMs"] if r["stream_cost"]["longTaskSupported"] else r["stream_cost"]["frameTimeOver50Ms"], "ms"),
    ("typing key->frame p50", lambda r: r["type_key_to_frame_p50"] or 0, "ms"),
    ("typing key->frame max", lambda r: r["type_key_to_frame_max"] or 0, "ms"),
]
rows = []
for f in sys.argv[1:]:
    rows += json.load(open(f))
groups = collections.defaultdict(list)
for r in rows:
    groups[(r["engine"], r["throttle"], r["arm"])].append(r)
configs = sorted({(e, t) for e, t, _ in groups})
arms = []
for _, _, a in groups:
    if a not in arms: arms.append(a)
for e, t in configs:
    present = [a for a in arms if (e, t, a) in groups]
    print(f"\n### {e} cpu x{t:g} (n = {', '.join(str(len(groups[(e, t, a)])) for a in present)})")
    print("| metric | " + " | ".join(present) + " |")
    print("| --- | " + " | ".join("---" for _ in present) + " |")
    for name, fn, unit in METRICS:
        cells = []
        for a in present:
            vals = [fn(r) for r in groups[(e, t, a)]]
            cells.append(f"{statistics.median(vals):.0f}{unit} ({min(vals):.0f}-{max(vals):.0f})")
        print(f"| {name} | " + " | ".join(cells) + " |")

# Chromium layout and style time per phase (CDP Performance metrics), when recorded.
for e, t in configs:
    present = [a for a in arms if (e, t, a) in groups and groups[(e, t, a)][0].get("perf_scroll")]
    if not present:
        continue
    print(f"\n### {e} cpu x{t:g}: main-thread breakdown (CDP Performance metrics, ms)")
    print("| phase / metric | " + " | ".join(present) + " |")
    print("| --- | " + " | ".join("---" for _ in present) + " |")
    for phase in ("perf_seed_open", "perf_scroll", "perf_resize", "perf_stream"):
        for key in ("LayoutDuration", "RecalcStyleDuration", "ScriptDuration", "TaskDuration"):
            cells = []
            for a in present:
                vals = [r[phase][key] for r in groups[(e, t, a)] if r.get(phase)]
                cells.append(f"{statistics.median(vals):.0f} ({min(vals):.0f}-{max(vals):.0f})" if vals else "-")
            print(f"| {phase[5:]} {key} | " + " | ".join(cells) + " |")
    for a in present:
        pass
print_resize = True
for e, t in configs:
    present = [a for a in arms if (e, t, a) in groups and "resize_cost" in groups[(e, t, a)][0]]
    if not present:
        continue
    print(f"\n### {e} cpu x{t:g}: width change x4")
    print("| metric | " + " | ".join(present) + " |")
    print("| --- | " + " | ".join("---" for _ in present) + " |")
    for name, fn in (("longest frame", lambda r: r["resize_cost"]["longestFrameMs"]),
                     ("frames over 50 ms, total", lambda r: r["resize_cost"]["frameTimeOver50Ms"])):
        cells = []
        for a in present:
            vals = [fn(r) for r in groups[(e, t, a)]]
            cells.append(f"{statistics.median(vals):.0f}ms ({min(vals):.0f}-{max(vals):.0f})")
        print(f"| {name} | " + " | ".join(cells) + " |")
