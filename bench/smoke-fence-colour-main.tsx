// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Measurement entry for chat code-fence colour: the real Thread, a seeded long chat with many
// fences of mixed lengths and languages, a streaming adapter, and a per-frame colour sampler.

/* eslint-disable no-restricted-imports -- a measurement entry point, not app code. */
import "@/features/chat/stores/sidebar-organization-store";
/* eslint-enable no-restricted-imports */

import { Thread } from "@/components/assistant-ui/thread";
import { TooltipProvider } from "@/components/ui/tooltip";
import {
  AssistantRuntimeProvider,
  type ChatModelAdapter,
  ExportedMessageRepository,
  type ThreadMessageLike,
  useAui,
  useLocalRuntime,
} from "@assistant-ui/react";
import {
  RouterProvider,
  createMemoryHistory,
  createRootRoute,
  createRouter,
} from "@tanstack/react-router";
import { type ReactElement, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./src/index.css";

const realFetch = window.fetch.bind(window);
window.fetch = (input, init) => {
  const url = typeof input === "string" ? input : ((input as Request).url ?? String(input));
  if (url.includes("/api/")) {
    return Promise.resolve(
      new Response("{}", { status: 200, headers: { "content-type": "application/json" } }),
    );
  }
  return realFetch(input, init);
};

// ---- long-task and frame recorders, always on so every window can be cut out of them ----
type Span = { start: number; duration: number };
const longTasks: Span[] = [];
let longTaskSupported = false;
try {
  const observer = new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
      longTasks.push({ start: entry.startTime, duration: entry.duration });
    }
  });
  observer.observe({ type: "longtask", buffered: true });
  longTaskSupported = PerformanceObserver.supportedEntryTypes?.includes("longtask") ?? false;
} catch {
  longTaskSupported = false;
}
const frameTimes: number[] = [];

// ---- fixture ----
const PROSE = [
  "The reception of a long thread is decided by what the renderer does on every interaction.",
  "A reply that arrives quickly can still leave a thread that answers a keystroke slowly.",
  "Layout that is not contained propagates upward, so a change inside one message is felt everywhere.",
  "None of this is visible on a short thread, which is why the fixture here is long.",
];

const prose = (index: number, paragraphs: number): string =>
  Array.from({ length: paragraphs }, (_, i) => `${PROSE[(index + i) % PROSE.length]} (paragraph ${i + 1} of reply ${index})`).join("\n\n");

type Lang = { tag: string; line: (i: number, n: number) => string };
const LANGS: Lang[] = [
  { tag: "python", line: (i, n) => `def step_${i}_${n}(rows: list[dict], scale: float = ${(i + n) / 7}) -> float:\n    total = sum(row.get("w_${n}", 0.0) for row in rows)  # weight ${n}\n    return total * scale + ${n}.0` },
  { tag: "typescript", line: (i, n) => `export function score${i}_${n}(rows: readonly Row[]): number {\n  return rows.reduce((t, r) => t + r.weight * ${n + 1}, 0); // step ${n}\n}` },
  { tag: "rust", line: (i, n) => `fn step_${i}_${n}(rows: &[Row]) -> f64 {\n    rows.iter().map(|r| r.weight * ${n}.5).sum::<f64>() // step\n}` },
  { tag: "go", line: (i, n) => `func Step${i}x${n}(rows []Row) float64 {\n\ttotal := 0.0\n\tfor _, r := range rows { total += r.Weight * ${n}.0 }\n\treturn total\n}` },
  { tag: "json", line: (i, n) => `  {"id": "row-${i}-${n}", "weight": ${(n % 17) + 0.5}, "count": ${n * 3 + i}, "ok": true},` },
  { tag: "bash", line: (i, n) => `for f in ./data/${i}/*.csv; do echo "step ${n}: $f"; wc -l "$f" | awk '{print $1}'; done` },
  { tag: "sql", line: (i, n) => `SELECT id, SUM(weight * ${n}) AS total FROM rows_${i} WHERE label <> 'skip-${n}' GROUP BY id;` },
  { tag: "cpp", line: (i, n) => `double step_${i}_${n}(const std::vector<Row>& rows) {\n  double t = 0; for (auto& r : rows) t += r.weight * ${n}; return t; // step\n}` },
  { tag: "yaml", line: (i, n) => `step_${i}_${n}:\n  weight: ${n}.5\n  labels: [a, b, "skip-${n}"]` },
  { tag: "java", line: (i, n) => `public static double step${i}x${n}(List<Row> rows) { return rows.stream().mapToDouble(r -> r.weight * ${n}).sum(); }` },
  { tag: "css", line: (i, n) => `.row-${i}-${n} { color: hsl(${(i * 31 + n * 7) % 360}, 70%, 55%); margin: ${n % 8}px 0; }` },
  { tag: "html", line: (i, n) => `<div class="row-${i}-${n}" data-step="${n}"><span>${n}</span> <a href="/rows/${i}">open</a></div>` },
];
// Units per fence (each unit is 1 to 5 lines): short, medium and long fences.
const SIZES = [2, 6, 14, 40, 3, 90, 8, 24];

const fence = (index: number, slot: number): string => {
  const lang = LANGS[(index * 3 + slot) % LANGS.length];
  const units = SIZES[(index + slot * 5) % SIZES.length];
  const body = [lang.tag === "json" ? "[" : `${lang.tag === "python" || lang.tag === "bash" || lang.tag === "yaml" ? "#" : "//"} reply ${index} fence ${slot}`];
  for (let n = 0; n < units; n += 1) body.push(lang.line(index * 10 + slot, n));
  if (lang.tag === "json") body.push(`  {"end": ${index}}`, "]");
  return ["```" + lang.tag, ...body, "```"].join("\n");
};

const replyText = (index: number): string => {
  const fences = 2 + (index % 3);
  const parts = [prose(index, 2)];
  for (let slot = 0; slot < fences; slot += 1) {
    parts.push(fence(index, slot));
    parts.push(prose(index + slot + 1, 1));
  }
  return parts.join("\n\n");
};

const history = (turns: number): ThreadMessageLike[] => {
  const out: ThreadMessageLike[] = [];
  for (let i = 0; i < turns; i += 1) {
    out.push({ role: "user", content: [{ type: "text", text: `Question ${i}: show me the code.` }] });
    out.push({ role: "assistant", content: [{ type: "text", text: replyText(i) }] });
  }
  return out;
};

const streamText = (lines: number): string => {
  const body = ["# streamed reply: a scorer written line by line", "from dataclasses import dataclass", ""];
  for (let n = 0; n < lines; n += 1) {
    body.push(`def live_${n}(rows: list[dict], scale: float = ${n / 3}) -> float:  # step ${n}`);
    body.push(`    return sum(row.get("w_${n}", 0.0) for row in rows) * scale`);
  }
  return [prose(900, 1), ["```python", ...body, "```"].join("\n"), prose(901, 1)].join("\n\n");
};

// ---- colour sampler ----
const viewport = (): HTMLElement | null => document.querySelector<HTMLElement>(".aui-thread-viewport");
const blocks = (): HTMLElement[] =>
  Array.from(document.querySelectorAll<HTMLElement>('.aui-thread-root [data-streamdown="code-block"]'));

const lineColoured = (line: Element): boolean =>
  (line as HTMLElement).style?.getPropertyValue("--sdm-c") !== ""
  || line.querySelector('[style*="--sdm-c"]') !== null;

type BlockView = { inView: boolean; visibleLines: number; uncolouredLines: number; deferred: boolean; plain: boolean };

const viewOf = (block: HTMLElement, top: number, bottom: number): BlockView => {
  const rect = block.getBoundingClientRect();
  const none = { inView: false, visibleLines: 0, uncolouredLines: 0, deferred: false, plain: false };
  if (rect.bottom <= top || rect.top >= bottom || rect.height === 0) return none;
  const body = block.querySelector<HTMLElement>('[data-streamdown="code-block-body"]');
  const code = block.querySelector<HTMLElement>("pre code");
  if (!code || !body) return { ...none, inView: true };
  const bodyRect = body.getBoundingClientRect();
  const lo = Math.max(top, bodyRect.top);
  const hi = Math.min(bottom, bodyRect.bottom);
  if (hi <= lo) return { ...none, inView: false };
  const deferred = block.hasAttribute("data-unsloth-fence-deferred");
  const lines = code.children;
  if (deferred || lines.length === 0) {
    const text = (code.textContent ?? "").trim();
    return { inView: true, visibleLines: text ? 1 : 0, uncolouredLines: text ? 1 : 0, deferred, plain: !deferred };
  }
  let low = 0;
  let high = lines.length - 1;
  while (low < high) {
    const mid = (low + high) >> 1;
    if (lines[mid].getBoundingClientRect().bottom <= lo) low = mid + 1;
    else high = mid;
  }
  let visible = 0;
  let uncoloured = 0;
  for (let i = low; i < lines.length; i += 1) {
    const r = lines[i].getBoundingClientRect();
    if (r.top >= hi) break;
    if ((lines[i].textContent ?? "").trim() === "") continue;
    visible += 1;
    if (!lineColoured(lines[i])) uncoloured += 1;
  }
  return { inView: true, visibleLines: visible, uncolouredLines: uncoloured, deferred: false, plain: false };
};

type Frame = { t: number; inView: number; uncolouredBlocks: number; uncolouredLines: number; visibleLines: number; deferred: number; plain: number; partial: number };
const state = {
  sampling: false,
  frames: [] as Frame[],
  streamStartedAt: null as number | null,
  streamEndedAt: null as number | null,
  streamLines: 80,
  chunkChars: 24,
  gapMs: 16,
  lineSeen: new Map<number, number>(),
  lineColoured: new Map<number, number>(),
  streamFrames: 0,
  streamUncolouredFrames: 0,
};

const sampleAll = (now: number): void => {
  const view = viewport();
  if (!view) return;
  const vr = view.getBoundingClientRect();
  let inView = 0;
  let uncolouredBlocks = 0;
  let uncolouredLines = 0;
  let visibleLines = 0;
  let deferred = 0;
  let plain = 0;
  let partial = 0;
  for (const block of blocks()) {
    const v = viewOf(block, vr.top, vr.bottom);
    if (!v.inView) continue;
    inView += 1;
    visibleLines += v.visibleLines;
    uncolouredLines += v.uncolouredLines;
    if (v.uncolouredLines > 0) {
      uncolouredBlocks += 1;
      if (v.deferred) deferred += 1;
      else if (v.plain) plain += 1;
      else partial += 1;
    }
  }
  state.frames.push({ t: now, inView, uncolouredBlocks, uncolouredLines, visibleLines, deferred, plain, partial });
};

const sampleStream = (now: number): void => {
  const all = blocks();
  const block = all[all.length - 1];
  const view = viewport();
  if (!block || !view) return;
  const vr = view.getBoundingClientRect();
  const body = block.querySelector<HTMLElement>('[data-streamdown="code-block-body"]');
  const code = block.querySelector<HTMLElement>("pre code");
  if (!body || !code) return;
  const br = body.getBoundingClientRect();
  const lo = Math.max(vr.top, br.top);
  const hi = Math.min(vr.bottom, br.bottom);
  state.streamFrames += 1;
  const lines = code.children;
  let uncoloured = 0;
  if (lines.length === 0 && (code.textContent ?? "").trim() !== "") uncoloured = 1;
  for (let i = 0; i < lines.length; i += 1) {
    const r = lines[i].getBoundingClientRect();
    if (r.bottom <= lo) continue;
    if (r.top >= hi) break;
    if ((lines[i].textContent ?? "").trim() === "") continue;
    const coloured = lineColoured(lines[i]);
    if (!state.lineSeen.has(i)) state.lineSeen.set(i, now);
    if (coloured && !state.lineColoured.has(i)) state.lineColoured.set(i, now);
    if (!coloured) uncoloured += 1;
  }
  if (uncoloured > 0) state.streamUncolouredFrames += 1;
};

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const adapter: ChatModelAdapter = {
  async *run() {
    const text = streamText(state.streamLines);
    state.streamStartedAt = performance.now();
    let cursor = 0;
    while (cursor < text.length) {
      cursor = Math.min(text.length, cursor + state.chunkChars);
      yield { content: [{ type: "text" as const, text: text.slice(0, cursor) }] };
      await sleep(state.gapMs);
    }
    state.streamEndedAt = performance.now();
  },
};

const windowStats = (start: number, end: number) => {
  const gaps: number[] = [];
  for (let i = 1; i < frameTimes.length; i += 1) {
    if (frameTimes[i] < start || frameTimes[i - 1] > end) continue;
    gaps.push(frameTimes[i] - frameTimes[i - 1]);
  }
  const tasks = longTasks.filter((task) => task.start + task.duration >= start && task.start <= end);
  return {
    frames: gaps.length,
    longestFrameMs: gaps.length ? Math.max(...gaps) : 0,
    framesOver50: gaps.filter((g) => g > 50).length,
    frameTimeOver50Ms: gaps.filter((g) => g > 50).reduce((a, b) => a + b, 0),
    longTaskSupported,
    longTasks: tasks.length,
    longTaskMs: tasks.reduce((a, task) => a + task.duration, 0),
    longestTaskMs: tasks.length ? Math.max(...tasks.map((task) => task.duration)) : 0,
  };
};

function FenceColourApi({ mounted, setMounted }: { mounted: boolean; setMounted: (v: boolean) => void }): null {
  const aui = useAui();
  useEffect(() => {
    let handle = 0;
    const loop = (now: number) => {
      frameTimes.push(now);
      if (frameTimes.length > 20000) frameTimes.splice(0, 10000);
      if (state.sampling) sampleAll(now);
      if (state.streamStartedAt !== null && state.streamEndedAt === null) sampleStream(now);
      handle = requestAnimationFrame(loop);
    };
    handle = requestAnimationFrame(loop);
    const api = {
      seed(turns: number): number {
        aui.thread().import(ExportedMessageRepository.fromArray(history(turns)));
        return turns;
      },
      seedBig(fences: number, lines: number): number {
        const text = Array.from({ length: fences }, (_, f) => {
          const body = Array.from({ length: lines }, (_, n) => `total_${f}_${n} = sum(row.get("w_${n}", 0.0) for row in rows) * ${n}.5  # step ${n}`);
          return ["```python", ...body, "```"].join("\n");
        }).join("\n\nAnd the next one.\n\n");
        aui.thread().import(ExportedMessageRepository.fromArray([
          { role: "user", content: [{ type: "text", text: "big" }] },
          { role: "assistant", content: [{ type: "text", text }] },
        ]));
        return fences * lines;
      },
      fenceCount: () => blocks().length,
      now: () => performance.now(),
      close: () => setMounted(false),
      open: () => setMounted(true),
      isOpen: () => mounted,
      startSampling(): void {
        state.frames = [];
        state.sampling = true;
      },
      stopSampling(): Frame[] {
        state.sampling = false;
        return state.frames;
      },
      snapshot(): Frame {
        const before = state.frames;
        state.frames = [];
        sampleAll(performance.now());
        const frame = state.frames[0];
        state.frames = before;
        return frame;
      },
      viewportBox(): { x: number; y: number; w: number; h: number; scrollTop: number; scrollHeight: number } {
        const v = viewport();
        const r = v?.getBoundingClientRect();
        return { x: r?.left ?? 0, y: r?.top ?? 0, w: r?.width ?? 0, h: r?.height ?? 0, scrollTop: v?.scrollTop ?? -1, scrollHeight: v?.scrollHeight ?? -1 };
      },
      toBottom(): number {
        const v = viewport();
        if (!v) return -1;
        v.style.scrollBehavior = "auto";
        v.scrollTop = v.scrollHeight;
        return v.scrollTop;
      },
      stats: windowStats,
      stream(options: { lines?: number; chunkChars?: number; gapMs?: number } = {}): void {
        state.streamLines = options.lines ?? state.streamLines;
        state.chunkChars = options.chunkChars ?? state.chunkChars;
        state.gapMs = options.gapMs ?? state.gapMs;
        state.streamStartedAt = null;
        state.streamEndedAt = null;
        state.lineSeen = new Map();
        state.lineColoured = new Map();
        state.streamFrames = 0;
        state.streamUncolouredFrames = 0;
        setTimeout(() => {
          aui.thread().append({ role: "user", content: [{ type: "text", text: "stream it" }] });
        }, 0);
      },
      streamDone: () => state.streamEndedAt !== null && !aui.thread().getState().isRunning,
      streamResult() {
        const lags: number[] = [];
        let never = 0;
        for (const [line, seen] of state.lineSeen) {
          const coloured = state.lineColoured.get(line);
          if (coloured === undefined) never += 1;
          else lags.push(coloured - seen);
        }
        lags.sort((a, b) => a - b);
        const q = (p: number) => (lags.length ? lags[Math.min(lags.length - 1, Math.floor(p * lags.length))] : 0);
        return {
          startedAt: state.streamStartedAt,
          endedAt: state.streamEndedAt,
          frames: state.streamFrames,
          uncolouredFrames: state.streamUncolouredFrames,
          linesSeen: state.lineSeen.size,
          neverColoured: never,
          lagP50: q(0.5),
          lagP95: q(0.95),
          lagMax: lags.length ? lags[lags.length - 1] : 0,
        };
      },
    };
    (window as unknown as { __fc: typeof api }).__fc = api;
    return () => cancelAnimationFrame(handle);
  }, [aui, mounted, setMounted]);
  return null;
}

function Harness(): ReactElement {
  const runtime = useLocalRuntime(adapter);
  const [mounted, setMounted] = useState(true);
  return (
    <TooltipProvider>
      <AssistantRuntimeProvider runtime={runtime}>
        <FenceColourApi mounted={mounted} setMounted={setMounted} />
        <div data-smoke="fence-colour" style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
          {mounted ? <Thread hideWelcome={true} /> : null}
        </div>
      </AssistantRuntimeProvider>
    </TooltipProvider>
  );
}

const rootRoute = createRootRoute({ component: Harness });
const router = createRouter({ routeTree: rootRoute, history: createMemoryHistory({ initialEntries: ["/"] }) });
const root = document.getElementById("root");
if (!root) throw new Error("missing #root");
createRoot(root).render(<RouterProvider router={router as unknown as never} />);
