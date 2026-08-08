// Load repro/probe.html in a real WebView2 control and print the result JSON.
//
// WebView2 is the engine the Tauri desktop build embeds on Windows, so this is
// the faithful Windows counterpart to the WebKitGTK and WKWebView probes.
//
//   dotnet run --project repro/webview2 -- <abs path to probe.html>
//
// Exits 0 with JSON on stdout; exits 1 on harness failure only, never on
// "did not reproduce".

using System;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Forms;
using Microsoft.Web.WebView2.Core;
using Microsoft.Web.WebView2.WinForms;

internal static class Program
{
    private static string? _outPath;

    [STAThread]
    private static int Main(string[] args)
    {
        if (args.Length < 1)
        {
            Console.Error.WriteLine("usage: webview2probe <path to probe.html>");
            return 2;
        }

        _outPath = Environment.GetEnvironmentVariable("REPRO_OUT");
        Application.EnableVisualStyles();

        var form = new Form { Width = 900, Height = 700, ShowInTaskbar = false };
        var view = new WebView2 { Dock = DockStyle.Fill };
        form.Controls.Add(view);

        int exitCode = 1;
        string payload = "{\"error\": \"never ran\"}";

        form.Shown += async (_, _) =>
        {
            try
            {
                await view.EnsureCoreWebView2Async(null);
                var tcs = new TaskCompletionSource<bool>();
                view.CoreWebView2.NavigationCompleted += (_, e) => tcs.TrySetResult(e.IsSuccess);
                view.CoreWebView2.Navigate(new Uri(Path.GetFullPath(args[0])).AbsoluteUri);

                var ok = await tcs.Task;
                if (!ok)
                {
                    payload = "{\"error\": \"navigation failed\"}";
                }
                else
                {
                    // The page measures 250ms after load; poll past that.
                    for (var attempt = 0; attempt < 25; attempt++)
                    {
                        await Task.Delay(500);
                        var raw = await view.CoreWebView2.ExecuteScriptAsync(
                            "window.__RESULT ? JSON.stringify(window.__RESULT) : null");
                        if (raw is not null && raw != "null")
                        {
                            // ExecuteScriptAsync returns a JSON-encoded string.
                            payload = System.Text.Json.JsonSerializer.Deserialize<string>(raw)
                                      ?? "{\"error\": \"undecodable result\"}";
                            exitCode = 0;
                            break;
                        }
                    }

                    if (exitCode != 0)
                    {
                        payload = "{\"error\": \"window.__RESULT never appeared\"}";
                    }
                }
            }
            catch (Exception ex)
            {
                payload = "{\"error\": " + System.Text.Json.JsonSerializer.Serialize(ex.Message) + "}";
            }
            finally
            {
                form.Close();
            }
        };

        Application.Run(form);

        Console.WriteLine(payload);
        if (!string.IsNullOrEmpty(_outPath))
        {
            File.WriteAllText(_outPath, payload);
        }
        return exitCode;
    }
}
