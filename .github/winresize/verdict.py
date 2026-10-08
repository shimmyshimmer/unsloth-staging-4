"""Judge probe.py records: base must leave the WebView short of the client area and
head must not. Prints a markdown table + one VERDICT line; writes verdict.json."""

import glob
import json
import os
import sys

out = sys.argv[1]
recs = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(out, "*_l*.json")))]
arms = {}
for r in recs:
    a = arms.setdefault(r["label"], {"launches": 0, "errors": [], "kinds": {}})
    a["launches"] += 1
    if r.get("error"):
        a["errors"].append(r["error"])
    for t in r["trials"]:
        k = a["kinds"].setdefault(t["kind"], {"trials": 0, "settled_bad": 0, "early_bad": 0, "worst": [0, 0]})
        k["trials"] += 1
        k["settled_bad"] += t["settled_mismatch"]
        k["early_bad"] += t["early_mismatch"]
        for key in ("container_delta", "widget_delta"):
            d = t["settled"][key] or [0, 0]
            if abs(d[0]) + abs(d[1]) > abs(k["worst"][0]) + abs(k["worst"][1]):
                k["worst"] = d

lines = ["| arm | trial kind | trials | WebView short of window after 1.5 s | short at 150 ms | worst settled delta (w,h px) |",
         "|---|---|---|---|---|---|"]
for arm in sorted(arms):
    for kind, k in sorted(arms[arm]["kinds"].items()):
        lines.append("| %s | %s | %d | %d | %d | %s |" % (arm, kind, k["trials"], k["settled_bad"], k["early_bad"], k["worst"]))
print("\n".join(lines))
probe = {r["label"] + str(r["launch"]): {k: r.get(k) for k in ("ready", "cursor_ok", "input_desktop", "thickframe", "error")} for r in recs}
print("probes:", json.dumps(probe))


def total(arm, field):
    return sum(k[field] for k in arms.get(arm, {"kinds": {}})["kinds"].values())


def trials(arm):
    return sum(k["trials"] for k in arms.get(arm, {"kinds": {}})["kinds"].values())


if not trials("base") or not trials("head"):
    verdict = "VOID (an arm ran no trials)"
elif total("base", "settled_bad") == 0:
    verdict = "NOT_REPRODUCED (base WebView always matched the window after settling)"
elif total("head", "settled_bad") == 0:
    verdict = "CONFIRMED (base %d/%d settled mismatches, head 0/%d)" % (
        total("base", "settled_bad"), trials("base"), trials("head"))
else:
    verdict = "FIX_INCOMPLETE (base %d/%d, head %d/%d)" % (
        total("base", "settled_bad"), trials("base"), total("head", "settled_bad"), trials("head"))
print("VERDICT", verdict)
json.dump({"verdict": verdict, "arms": arms, "probes": probe}, open(os.path.join(out, "verdict.json"), "w"), indent=1)
if os.environ.get("GITHUB_STEP_SUMMARY"):
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\nVERDICT " + verdict + "\n")
