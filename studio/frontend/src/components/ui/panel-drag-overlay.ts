// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Z_LAYER } from "@/lib/z-layers"

// A drag needs one cursor over the whole viewport, because the pointer travels across buttons and
// text that would otherwise claim their own. That used to be `html[data-panel-resizing] *` plus
// `cursor`/`user-select` on <body>. Both reach every element in the document: the universal
// selector matches all of them, and `cursor` and `user-select` are inherited, so writing them on
// <body> marks inherited style dirty for everything below it. The cost is therefore proportional to
// the STANDING DOM rather than to the one thing that changed, which is the same shape as the
// sidebar-width writes scoped in #9400/#9441. The same is true of the rule that blanked pointer
// events on the sidebar and on [data-slot="sidebar-inset"], the <main> holding the whole app
// including the thread: `pointer-events` is inherited too, so that write dirtied the thread's
// subtree on both flips as well. A single fixed element on top of the viewport does both jobs with
// an invalidation set of one element. It carries the cursor, and by being the hit test target for
// the whole viewport it keeps hover and click off the content underneath. It is transparent and it
// is removed the instant the drag ends, so nothing about what the user sees changes.
const DRAG_OVERLAY_SLOT = "panel-resize-drag-overlay"
const PANEL_RESIZING_ATTRIBUTE = "data-panel-resizing"
/** Nested drags cannot happen through pointer capture, but a stuck overlay would
 *  swallow the whole UI, so ownership is explicit rather than assumed. */
let dragOverlayOwners = 0
let resizingHolders = 0
const DRAG_SLOP = 4

function holdResizing(): void {
  resizingHolders += 1
  if (resizingHolders > 1) return
  document.documentElement.setAttribute(PANEL_RESIZING_ATTRIBUTE, "true")
}

function releaseResizing(): void {
  if (resizingHolders === 0) return
  resizingHolders -= 1
  if (resizingHolders > 0) return
  document.documentElement.removeAttribute(PANEL_RESIZING_ATTRIBUTE)
}

export function acquireDragOverlay(): void {
  dragOverlayOwners += 1
  if (dragOverlayOwners > 1) return
  holdResizing()
  const el = document.createElement("div")
  el.setAttribute("data-slot", DRAG_OVERLAY_SLOT)
  // Decorative and non-interactive as far as assistive tech is concerned: it
  // exists only to own the cursor while the pointer is already captured.
  el.setAttribute("aria-hidden", "true")
  const s = el.style
  s.position = "fixed"
  s.inset = "0"
  // Top of the named scale, not a hand-picked large number: the rules it replaces were `!important`
  // and blanked whole subtrees, so anything it did not out-rank it would only partly stand in for.
  s.zIndex = String(Z_LAYER.DRAG_CURSOR_OVERLAY)
  s.background = "transparent"
  // Explicit, because it is load-bearing rather than incidental. Being the hit
  // test target for the whole viewport is what keeps hover and click off the
  // content underneath, which is the job `pointer-events: none` on
  // [data-slot="sidebar-inset"] used to do by dirtying the thread's subtree.
  s.pointerEvents = "auto"
  // col-resize unconditionally, which is what the replaced rule did even when a
  // collapsed edge was being dragged open.
  s.cursor = "col-resize"
  s.userSelect = "none"
  s.touchAction = "none"
  document.body.appendChild(el)
}

export function releaseDragOverlay(): void {
  if (dragOverlayOwners === 0) return
  dragOverlayOwners -= 1
  if (dragOverlayOwners > 0) return
  document
    .querySelector(`[data-slot="${DRAG_OVERLAY_SLOT}"]`)
    ?.remove()
  releaseResizing()
}

export function armPanelDrag(startX: number): {
  move: (x: number) => void
  release: () => void
} {
  holdResizing()
  let overlaid = false
  let released = false
  return {
    move: (x) => {
      if (overlaid || released || Math.abs(x - startX) < DRAG_SLOP) return
      overlaid = true
      acquireDragOverlay()
    },
    release: () => {
      if (released) return
      released = true
      if (overlaid) releaseDragOverlay()
      releaseResizing()
    },
  }
}

export function panelDragInProgress(): boolean {
  return dragOverlayOwners > 0
}
