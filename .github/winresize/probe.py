"""Launch the Desktop exe, resize its window fast, and record whether the WebView
(wry container + WebView2 widget) ends at the window's client size.

Observes only; verdict.py judges. Windows only, stdlib ctypes (+ Pillow if present).
"""

import argparse
import ctypes
import ctypes.wintypes as wt
import json
import os
import subprocess
import sys
import time

user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.SetProcessDpiAwarenessContext.restype = wt.BOOL
user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # PER_MONITOR_AWARE_V2

WNDENUMPROC = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
SWP_NOSIZE, SWP_NOMOVE, SWP_NOZORDER, SWP_NOACTIVATE, SWP_ASYNCWINDOWPOS = (
    0x1, 0x2, 0x4, 0x10, 0x4000)
GWL_STYLE = -16
WS_THICKFRAME = 0x00040000
MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP = 0x2, 0x4
user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t


def pid_of(hwnd):
    pid = wt.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


def class_name(hwnd):
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)
    return buf.value


def top_windows(pid):
    out = []

    def cb(hwnd, _):
        if pid_of(hwnd) == pid and not user32.GetWindow(hwnd, 4):  # GW_OWNER
            out.append(hwnd)
        return True

    user32.EnumWindows(WNDENUMPROC(cb), 0)
    return out


def descendants(hwnd):
    out = []

    def cb(child, _):
        out.append(child)
        return True

    user32.EnumChildWindows(hwnd, WNDENUMPROC(cb), 0)
    return out


def client_size(hwnd):
    r = wt.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(r))
    return [r.right - r.left, r.bottom - r.top]


def rect_in_client(top, hwnd):
    r = wt.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    pts = (wt.POINT * 2)(wt.POINT(r.left, r.top), wt.POINT(r.right, r.bottom))
    user32.MapWindowPoints(None, top, pts, 2)
    return [pts[0].x, pts[0].y, pts[1].x - pts[0].x, pts[1].y - pts[0].y]


def geometry(top):
    client = client_size(top)
    kids = []
    for h in descendants(top):
        if not user32.IsWindowVisible(h):
            continue
        kids.append({"class": class_name(h), "pid_is_app": None,
                     "parent_is_top": user32.GetParent(h) == top,
                     "rect": rect_in_client(top, h)})
    # The wry container is the top's direct child; WebView2's hosted window is the
    # Chrome_WidgetWin_* descendant (owned by msedgewebview2.exe).
    container = next((k for k in kids if k["parent_is_top"]), None)
    widget = next((k for k in kids if k["class"].startswith("Chrome_WidgetWin")), None)

    def off(k):
        if k is None:
            return None
        x, y, w, h = k["rect"]
        return [w - client[0], h - client[1]]

    return {"client": client, "container": container and container["rect"],
            "container_class": container and container["class"],
            "widget": widget and widget["rect"], "widget_class": widget and widget["class"],
            "container_delta": off(container), "widget_delta": off(widget),
            "n_children": len(kids)}


def mismatch(g):
    for key in ("container_delta", "widget_delta"):
        d = g[key]
        if d is None or d[0] != 0 or d[1] != 0:
            return True
    return False


def wait_window(proc, timeout):
    deadline = time.time() + timeout
    top = None
    while time.time() < deadline:
        if proc.poll() is not None:
            return None, "exited %s" % proc.returncode
        tops = top_windows(proc.pid)
        if tops:
            top = tops[0]
            g = geometry(top)
            if g["widget"] is not None and user32.IsWindowVisible(top):
                return top, "ready"
        time.sleep(0.5)
    if top is not None:
        user32.ShowWindow(top, 5)  # SW_SHOW: the setup page may never reveal it
        time.sleep(5)
        if geometry(top)["widget"] is not None:
            return top, "forced_show"
    return top, "timeout"


def shot(top, path):
    try:
        from PIL import ImageGrab
    except ImportError:
        return None
    r = wt.RECT()
    user32.GetWindowRect(top, ctypes.byref(r))
    ImageGrab.grab(bbox=(r.left, r.top, r.right, r.bottom), all_screens=True).save(path)
    return path


def burst(top, sizes, flags):
    for w, h in sizes:
        user32.SetWindowPos(top, None, 0, 0, w, h, flags)


def drag(top, dx, dy, steps, pause):
    r = wt.RECT()
    user32.GetWindowRect(top, ctypes.byref(r))
    x, y = r.right - 3, r.bottom - 3
    user32.SetCursorPos(x, y)
    time.sleep(0.2)
    user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    for i in range(1, steps + 1):
        user32.SetCursorPos(x + dx * i // steps, y + dy * i // steps)
        time.sleep(pause)
    user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def ramp(a, b, n):
    return [(a[0] + (b[0] - a[0]) * i // n, a[1] + (b[1] - a[1]) * i // n) for i in range(1, n + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--launch", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tag = "%s_l%d" % (args.label, args.launch)
    rec = {"label": args.label, "launch": args.launch, "exe": args.exe, "trials": []}

    # Interactive-session probe: SendInput drags mean nothing without one.
    user32.SetCursorPos(123, 77)
    p = wt.POINT()
    user32.GetCursorPos(ctypes.byref(p))
    rec["cursor_ok"] = (p.x, p.y) == (123, 77)
    rec["input_desktop"] = bool(user32.OpenInputDesktop(0, False, 0x0100))
    rec["screen"] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]

    proc = subprocess.Popen([args.exe], stdout=open(os.path.join(args.out, tag + ".log"), "w"),
                            stderr=subprocess.STDOUT)
    try:
        top, how = wait_window(proc, 180)
        rec["ready"] = how
        if top is None or how == "timeout":
            rec["error"] = "no webview window: " + how
            return rec
        time.sleep(3)
        style = user32.GetWindowLongPtrW(top, GWL_STYLE)
        rec["thickframe"] = bool(style & WS_THICKFRAME)
        user32.SetWindowPos(top, None, 40, 40, 900, 650, SWP_NOZORDER | SWP_NOACTIVATE)
        time.sleep(1.5)
        rec["initial"] = geometry(top)
        shot(top, os.path.join(args.out, tag + "_initial.png"))

        plan = []
        for i, (a, b) in enumerate([((900, 650), (1500, 1000)), ((1500, 1000), (800, 600))] * 3):
            plan.append(("burst_async", a, b, 80, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_ASYNCWINDOWPOS))
        for i, (a, b) in enumerate([((900, 650), (1500, 1000)), ((1500, 1000), (800, 600))] * 2):
            plan.append(("burst_sync", a, b, 80, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE))
        if rec["cursor_ok"] and rec["thickframe"]:
            for dx, dy in [(600, 350), (-600, -350)] * 3:
                plan.append(("drag", dx, dy, 30, 0.002))

        for n, step in enumerate(plan):
            kind = step[0]
            if kind == "drag":
                _, dx, dy, steps, pause = step
                user32.SetWindowPos(top, None, 40, 40, 900 if dx > 0 else 1500, 650 if dy > 0 else 1000,
                                    SWP_NOZORDER | SWP_NOACTIVATE)
                time.sleep(1.0)
                drag(top, dx, dy, steps, pause)
            else:
                _, a, b, count, flags = step
                user32.SetWindowPos(top, None, 0, 0, a[0], a[1], SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE)
                time.sleep(1.0)
                burst(top, ramp(a, b, count), flags)
            time.sleep(0.15)
            early = geometry(top)
            time.sleep(1.5)
            settled = geometry(top)
            png = shot(top, os.path.join(args.out, "%s_t%02d_%s.png" % (tag, n, kind)))
            rec["trials"].append({"n": n, "kind": kind, "early": early, "settled": settled,
                                  "early_mismatch": mismatch(early),
                                  "settled_mismatch": mismatch(settled), "png": png})
            print(tag, n, kind, "settled", settled["client"], settled["container_delta"],
                  settled["widget_delta"], flush=True)
        rec["alive_at_end"] = proc.poll() is None
        return rec
    finally:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
        time.sleep(3)


if __name__ == "__main__":
    a = sys.argv
    rec = main()
    out = a[a.index("--out") + 1]
    with open(os.path.join(out, "%s_l%s.json" % (a[a.index("--label") + 1], a[a.index("--launch") + 1])),
              "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    print("PROBE", json.dumps({k: rec.get(k) for k in ("label", "launch", "ready", "cursor_ok",
                                                         "input_desktop", "thickframe", "error")}))
