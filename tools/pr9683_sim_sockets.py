"""SIM 1 -- address-family reachability.

Which spellings of "loopback" actually reach a listener, and what does the ASGI
`scope["client"]` look like when they do? This is the leg of hypothesis H1 that decides
whether a non-trustworthy authority is even a threat: if the connection never lands, the
Host rule never matters.

Runs three listener shapes (IPv4 loopback, IPv6 loopback, dual-stack wildcard) against
every connect target. Pure stdlib, so it runs identically on Linux, macOS and Windows.

    python sim_sockets.py [--json out.json]
"""

import argparse
import json
import platform
import socket
import sys
import threading
import time

TIMEOUT = 3.0


def _serve_once(family, bind_addr, results, key):
    srv = socket.socket(family, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if family == socket.AF_INET6:
        # dual-stack when we bind ::, single-stack when we bind ::1
        try:
            srv.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        except OSError:
            pass
    srv.bind(bind_addr)
    srv.listen(16)
    results[key] = srv
    return srv


LISTENERS = [
    ("ipv4_loopback", socket.AF_INET, ("127.0.0.1", 0)),
    ("ipv6_loopback", socket.AF_INET6, ("::1", 0)),
    ("dual_wildcard", socket.AF_INET6, ("::", 0)),
]

# (label, connect family, connect host) -- the literal a browser or client would dial
TARGETS = [
    ("127.0.0.1", socket.AF_INET, "127.0.0.1"),
    ("127.0.0.2", socket.AF_INET, "127.0.0.2"),
    ("0.0.0.0", socket.AF_INET, "0.0.0.0"),
    ("::1", socket.AF_INET6, "::1"),
    ("::ffff:127.0.0.1", socket.AF_INET6, "::ffff:127.0.0.1"),
    ("::ffff:7f00:1", socket.AF_INET6, "::ffff:7f00:1"),
    ("::", socket.AF_INET6, "::"),
    ("::127.0.0.1 (v4-compat)", socket.AF_INET6, "::127.0.0.1"),
    ("localhost", None, "localhost"),
]


def probe(listener_label, family, bind_addr):
    srv = socket.socket(family, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if family == socket.AF_INET6:
        try:
            srv.setsockopt(
                socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0 if bind_addr[0] == "::" else 1
            )
        except OSError:
            pass
    try:
        srv.bind(bind_addr)
    except OSError as exc:
        return {"listener": listener_label, "error": f"bind failed: {exc}", "targets": {}}
    srv.listen(16)
    port = srv.getsockname()[1]

    seen = {}
    stop = threading.Event()

    def acceptor():
        srv.settimeout(0.25)
        while not stop.is_set():
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                conn.recv(256)
                conn.sendall(b"HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n")
            except OSError:
                pass
            seen.setdefault(addr, 0)
            seen[addr] += 1
            conn.close()

    t = threading.Thread(target=acceptor, daemon=True)
    t.start()

    out = {}
    for label, cfamily, host in TARGETS:
        before = sum(seen.values())
        try:
            if cfamily is None:
                s = socket.create_connection((host, port), timeout=TIMEOUT)
            else:
                s = socket.socket(cfamily, socket.SOCK_STREAM)
                s.settimeout(TIMEOUT)
                s.connect((host, port))
            s.sendall(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
            time.sleep(0.15)
            s.close()
            after = sum(seen.values())
            peer = None
            if after > before:
                peer = list(seen.keys())[-1]
            out[label] = {
                "reached": after > before,
                "peer": str(peer[0]) if peer else None,
            }
        except Exception as exc:
            out[label] = {"reached": False, "error": type(exc).__name__ + ": " + str(exc)}

    stop.set()
    time.sleep(0.4)
    srv.close()
    return {"listener": listener_label, "bind": str(bind_addr), "port": port, "targets": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()

    report = {
        "platform": platform.platform(),
        "system": platform.system(),
        "python": sys.version.split()[0],
        "has_ipv6": socket.has_ipv6,
        "listeners": [],
    }
    for label, family, bind_addr in LISTENERS:
        report["listeners"].append(probe(label, family, bind_addr))

    print(f"# SIM 1 -- address-family reachability on {report['system']} ({report['platform']})")
    print()
    for entry in report["listeners"]:
        print(f"## listener bound {entry['listener']} ({entry.get('bind')})")
        if "error" in entry:
            print(f"   {entry['error']}")
            print()
            continue
        print(f"   {'connect target':28} {'reached':>8}  peer as the server sees it")
        for label, res in entry["targets"].items():
            mark = "YES" if res["reached"] else "no"
            extra = res.get("peer") or res.get("error", "")
            print(f"   {label:28} {mark:>8}  {extra}")
        print()

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"json -> {args.json}")


if __name__ == "__main__":
    main()
