// Layout harness for PR 7927. Renders the SHIPPING components (via the `@`
// alias into the PR worktree) or their pre-PR copies (src/before/), inside a
// faithful reproduction of the settings-dialog shell and the custom titlebar
// band, and exposes a measurement hook for Playwright.
//
// ?variant=before|after   which copy of the four changed files to mount
// ?case=settings|chrome   which surface to render
// ?platform=Windows|macOS drives shouldUseCustomWindowTitlebar (see index.html)

import { StrictMode, type ReactNode } from "react";
import { createRoot } from "react-dom/client";

import { SidebarProvider } from "@/components/ui/sidebar";
import { useSidebarPin } from "@/hooks/use-sidebar-pin";

import { SettingsRow as SettingsRowAfter } from "@/features/settings/components/settings-row";
import { Navbar as NavbarAfter } from "@/components/navbar";
import { WindowTitlebar as WindowTitlebarAfter } from "@/components/tauri/window-titlebar";

import { SettingsRow as SettingsRowBefore } from "./before/settings-row";
import { Navbar as NavbarBefore } from "./before/navbar";
import { WindowTitlebar as WindowTitlebarBefore } from "./before/window-titlebar";

import "./harness.css";

const params = new URLSearchParams(location.search);
const VARIANT = params.get("variant") === "before" ? "before" : "after";
const CASE = params.get("case") || "settings";

const SettingsRow = VARIANT === "before" ? SettingsRowBefore : SettingsRowAfter;
const Navbar = VARIANT === "before" ? NavbarBefore : NavbarAfter;
const WindowTitlebar =
  VARIANT === "before" ? WindowTitlebarBefore : WindowTitlebarAfter;

// The same custom properties provider.tsx applies on the Tauri path.
const CUSTOM_CHROME_STYLE = {
  "--studio-titlebar-height": "0px",
  "--studio-custom-titlebar-height": "34px",
  "--studio-desktop-titlebar-height": "34px",
  "--studio-sidebar-expanded-width": "17.5rem",
  "--studio-sidebar-collapsed-width": "3rem",
  "--studio-window-control-inset": "112px",
} as React.CSSProperties;

// ---------------------------------------------------------------- settings case
// Controls are width-faithful reproductions of the real ones (the row's geometry
// depends only on the control's width and shrink behaviour, not its internals).

function HfTokenControl() {
  return (
    <div className="flex flex-col items-end gap-1.5">
      <div className="flex items-center gap-2">
        <div className="relative w-[260px]">
          <input className="h-8 w-full rounded-md border px-2 font-mono text-xs" />
        </div>
        <button type="button" className="h-8 rounded-md border px-3 text-sm">
          Save
        </button>
      </div>
    </div>
  );
}

function EmbeddingModelControl() {
  return (
    <div className="flex flex-col items-end gap-1 max-[360px]:w-full">
      <div className="flex items-center gap-2 max-[360px]:w-full">
        <button
          type="button"
          className="h-8 w-[220px] rounded-md border px-2 text-sm max-[360px]:min-w-0 max-[360px]:flex-1"
        >
          BAAI/bge-base-en-v1.5
        </button>
        <button type="button" className="h-8 rounded-md border px-3 text-sm">
          Save
        </button>
      </div>
    </div>
  );
}

function ModelsFolderControl() {
  return (
    <div className="grid w-[392px] min-w-0 grid-cols-[minmax(0,1fr)_auto] gap-x-2 gap-y-1.5 max-[840px]:w-full">
      <div className="relative min-w-0">
        <input className="h-8 w-full rounded-md border pr-7 font-mono text-xs" />
      </div>
      <button type="button" className="h-8 rounded-md border px-3 text-sm">
        Change
      </button>
    </div>
  );
}

function PaletteControl() {
  return (
    <div
      data-probe="stretch"
      className="grid grid-cols-4 gap-2 rounded-md border p-2"
    >
      {["a", "b", "c", "d"].map((k) => (
        <div key={k} className="h-10 rounded bg-muted" />
      ))}
    </div>
  );
}

const ROWS: {
  id: string;
  label: string;
  description: string;
  className?: string;
  control: ReactNode;
}[] = [
  {
    id: "hf-token",
    label: "Hugging Face token",
    description: "Used to load gated models and push artifacts.",
    control: <HfTokenControl />,
  },
  {
    id: "embedding-model",
    label: "Embedding model",
    description:
      "Hugging Face model or local path used to index and search your documents. Default is BAAI/bge-base-en-v1.5.",
    className:
      "max-[360px]:flex-col max-[360px]:items-stretch max-[360px]:gap-3",
    control: <EmbeddingModelControl />,
  },
  {
    id: "models-folder",
    label: "Models folder",
    description: "Where downloaded models are stored.",
    className:
      "max-[840px]:flex-col max-[840px]:items-stretch max-[840px]:gap-2",
    control: <ModelsFolderControl />,
  },
  {
    id: "palette",
    label: "Palette",
    description: "Pick the accent colours used across the app.",
    className: "flex-col items-stretch gap-3",
    control: <PaletteControl />,
  },
];

function SettingsCase() {
  // Mirrors settings-dialog.tsx: content capped at min(960px, 100vw-2rem),
  // a 248px nav aside, and a p-6 main column.
  return (
    <div className="flex h-dvh items-center justify-center bg-background">
      <div
        data-probe="dialog"
        className="flex h-[min(680px,calc(100dvh-2rem))] w-[min(960px,calc(100vw-2rem))] overflow-hidden rounded-xl border bg-background"
      >
        <aside className="w-[248px] shrink-0 border-r" />
        <main data-probe="main" className="min-w-0 flex-1 overflow-y-auto p-6">
          {ROWS.map((row) => (
            <div key={row.id} data-row={row.id}>
              <SettingsRow
                label={row.label}
                description={row.description}
                className={row.className}
              >
                {row.control}
              </SettingsRow>
            </div>
          ))}
        </main>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ chrome case

function ChromeCase() {
  const { pinned, setPinned, togglePinned } = useSidebarPin();
  return (
    <div
      className="relative h-dvh min-h-0 overflow-hidden bg-background"
      style={CUSTOM_CHROME_STYLE}
    >
      <WindowTitlebar showSidebarSurface={true} />
      <SidebarProvider
        pinned={pinned}
        setPinned={setPinned}
        togglePinned={togglePinned}
        className="!min-h-0 h-full"
      >
        {/* Stand-in for SidebarInset: Navbar positions absolutely inside it. */}
        <div
          data-probe="inset"
          className="relative flex min-h-0 w-full flex-1 flex-col"
        >
          <Navbar />
          <div className="flex-1" />
        </div>
      </SidebarProvider>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>{CASE === "chrome" ? <ChromeCase /> : <SettingsCase />}</StrictMode>,
);
