"""Load repro/probe.html in a real WebKitGTK WebView and print the result JSON.

This is the engine the Tauri desktop build uses on Linux, and the one the bug
was reported against (WebKitGTK 2.52.3). Run under xvfb-run.

  xvfb-run -a python3 repro/gtk_probe.py [--no-overlay]

`--no-overlay` sets GTK_OVERLAY_SCROLLING=0 for the classic-scrollbar control.
"""

import json
import os
import sys
from pathlib import Path

import gi

for api in ("4.1", "4.0", "6.0"):
    try:
        if api == "6.0":
            gi.require_version("WebKit", "6.0")
            from gi.repository import WebKit as WebKit2  # type: ignore
        else:
            gi.require_version("WebKit2", api)
            from gi.repository import WebKit2  # type: ignore
        WEBKIT_API = api
        break
    except (ValueError, ImportError):
        continue
else:  # pragma: no cover
    print(json.dumps({"error": "no WebKit2/WebKit typelib found"}))
    sys.exit(2)

gi.require_version("Gtk", "3.0" if WEBKIT_API != "6.0" else "4.0")
from gi.repository import GLib, Gtk  # noqa: E402

HERE = Path(__file__).resolve().parent
URL = (HERE / "probe.html").as_uri()

result = {"webkit_api": WEBKIT_API}
done = False


def finish(payload):
    global done
    result.update(payload)
    done = True
    Gtk.main_quit() if WEBKIT_API != "6.0" else None


def on_js_done(view, task):
    try:
        value = view.evaluate_javascript_finish(task)
        raw = value.to_string()
    except Exception:
        try:
            js_result = view.run_javascript_finish(task)
            raw = js_result.get_js_value().to_string()
        except Exception as exc:  # pragma: no cover
            finish({"error": f"evaluate failed: {exc!r}"})
            return
    try:
        finish(json.loads(raw))
    except Exception:
        finish({"error": "unparseable result", "raw": raw[:2000]})


def on_load_changed(view, event):
    if event != WebKit2.LoadEvent.FINISHED:
        return

    def poll():
        script = "window.__RESULT ? JSON.stringify(window.__RESULT) : null"
        try:
            view.evaluate_javascript(script, -1, None, None, None, on_js_done)
        except AttributeError:
            view.run_javascript(script, None, on_js_done)
        return False

    # The page runs its measurement 250ms after load; give it room plus a
    # couple of frames so the scrollbar has settled into its steady state.
    GLib.timeout_add(1200, poll)


def main():
    if "--no-overlay" in sys.argv:
        os.environ["GTK_OVERLAY_SCROLLING"] = "0"
        result["gtk_overlay_scrolling"] = "0"
    else:
        result["gtk_overlay_scrolling"] = os.environ.get("GTK_OVERLAY_SCROLLING", "default(1)")

    result["webkit_version"] = "%d.%d.%d" % (
        WebKit2.get_major_version(),
        WebKit2.get_minor_version(),
        WebKit2.get_micro_version(),
    )

    win = Gtk.Window()
    win.set_default_size(900, 700)
    view = WebKit2.WebView()
    settings = view.get_settings()
    settings.set_enable_developer_extras(True)
    win.add(view)
    win.show_all()
    view.connect("load-changed", on_load_changed)
    view.load_uri(URL)

    GLib.timeout_add_seconds(45, lambda: (finish({"error": "timeout"}), False)[1])
    Gtk.main()

    print(json.dumps(result, indent=2))
    out = os.environ.get("REPRO_OUT")
    if out:
        Path(out).write_text(json.dumps(result, indent=2))
    # Non-zero only on harness failure, never on "did not reproduce".
    sys.exit(0 if "error" not in result else 1)


if __name__ == "__main__":
    main()
