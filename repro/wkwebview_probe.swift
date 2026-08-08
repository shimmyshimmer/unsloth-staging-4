// Load repro/probe.html in a real WKWebView and print the result JSON.
//
// WKWebView is the engine the Tauri desktop build uses on macOS, where overlay
// scrollbars are the system default. Build and run:
//
//   swiftc -O repro/wkwebview_probe.swift -o /tmp/wkprobe
//   /tmp/wkprobe repro/probe.html
//
// Exits 0 with the JSON on stdout; exits 1 on harness failure only, never on
// "did not reproduce".

import Cocoa
import WebKit

final class Probe: NSObject, WKNavigationDelegate {
  let webView: WKWebView
  let window: NSWindow
  var finished = false

  init(width: CGFloat = 900, height: CGFloat = 700) {
    let config = WKWebViewConfiguration()
    let frame = NSRect(x: 0, y: 0, width: width, height: height)
    webView = WKWebView(frame: frame, configuration: config)
    window = NSWindow(
      contentRect: frame,
      styleMask: [.titled, .closable, .resizable],
      backing: .buffered,
      defer: false)
    super.init()
    window.contentView = webView
    window.makeKeyAndOrderFront(nil)
    webView.navigationDelegate = self
  }

  func load(_ path: String) {
    let url = URL(fileURLWithPath: path)
    webView.loadFileURL(url, allowingReadAccessTo: url.deletingLastPathComponent())
  }

  func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
    // The page measures 250ms after load; wait past that plus a few frames.
    DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { self.readResult(attempt: 0) }
  }

  func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
    self.emit("{\"error\": \"navigation failed: \(error.localizedDescription)\"}", code: 1)
  }

  func readResult(attempt: Int) {
    let js = "window.__RESULT ? JSON.stringify(window.__RESULT) : null"
    webView.evaluateJavaScript(js) { value, error in
      if let error = error {
        self.emit("{\"error\": \"evaluate failed: \(error.localizedDescription)\"}", code: 1)
        return
      }
      if let str = value as? String {
        self.emit(str, code: 0)
      } else if attempt < 20 {
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
          self.readResult(attempt: attempt + 1)
        }
      } else {
        self.emit("{\"error\": \"window.__RESULT never appeared\"}", code: 1)
      }
    }
  }

  func emit(_ json: String, code: Int32) {
    guard !finished else { return }
    finished = true
    // Annotate with the scroller style the OS is actually using.
    let style = NSScroller.preferredScrollerStyle == .overlay ? "overlay" : "legacy"
    var out = json
    if json.hasPrefix("{"), let brace = json.firstIndex(of: "{") {
      out = json.replacingCharacters(
        in: brace...brace, with: "{\"nsScrollerStyle\": \"\(style)\", ")
    }
    print(out)
    if let path = ProcessInfo.processInfo.environment["REPRO_OUT"] {
      try? out.write(toFile: path, atomically: true, encoding: .utf8)
    }
    exit(code)
  }
}

let args = CommandLine.arguments
guard args.count > 1 else {
  FileHandle.standardError.write("usage: wkprobe <path to probe.html>\n".data(using: .utf8)!)
  exit(2)
}

let app = NSApplication.shared
app.setActivationPolicy(.accessory)
let probe = Probe()
probe.load(args[1])

DispatchQueue.main.asyncAfter(deadline: .now() + 45) {
  probe.emit("{\"error\": \"timeout\"}", code: 1)
}

app.run()
