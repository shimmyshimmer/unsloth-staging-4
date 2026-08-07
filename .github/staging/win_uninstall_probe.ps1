# Windows probe for scripts/uninstall.ps1 (PR #7360).
#
# cross-platform-parity-ci.yml's paths filter covers install.ps1 and studio/setup.ps1
# but NOT scripts/uninstall.ps1, so the PowerShell half of this PR has no Windows job
# anywhere. This runs it on a real windows-latest runner.
#
# Everything is redirected into a sandbox directory first: USERPROFILE, LOCALAPPDATA and
# APPDATA are repointed, so the runner's own profile is never a target.

$ErrorActionPreference = "Stop"
$script:Pass = 0
$script:Fail = 0
function Ok  { param($m) Write-Host "  PASS: $m"; $script:Pass++ }
function Bad { param($m) Write-Host "  FAIL: $m" -ForegroundColor Red; $script:Fail++ }
function AssertGone    { param($l,$p) if (Test-Path -LiteralPath $p) { Bad "$l (still present: $p)" } else { Ok $l } }
function AssertPresent { param($l,$p) if (Test-Path -LiteralPath $p) { Ok $l } else { Bad "$l (missing: $p)" } }

$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
$uninstall = Join-Path $repo "scripts\uninstall.ps1"
$bid = "ai.unsloth.studio"

# ---------------------------------------------------------------- 0. parse ----
$err = $null
$null = [System.Management.Automation.Language.Parser]::ParseFile($uninstall, [ref]$null, [ref]$err)
if ($err) { $err | ForEach-Object { $_.Message }; Bad "uninstall.ps1 does not parse"; exit 1 }
Ok "uninstall.ps1 parses on Windows PowerShell $($PSVersionTable.PSVersion)"

# Build a sandboxed profile and run the uninstaller against it.
# Returns the captured output.
function Invoke-Sandboxed {
    # StudioHomeIsUserProfile rather than an $ExtraEnv hashtable: arguments are evaluated
    # at the call site, where $script:UserProf still holds the PREVIOUS sandbox's path.
    param([scriptblock]$Setup, [switch]$StudioHomeIsUserProfile)
    $sb = Join-Path ([System.IO.Path]::GetTempPath()) ("unsl_" + [guid]::NewGuid().ToString("N").Substring(0,8))
    New-Item -ItemType Directory -Force -Path $sb | Out-Null
    $script:Sandbox   = $sb
    $script:UserProf  = Join-Path $sb "user"
    $script:LocalApp  = Join-Path $sb "user\AppData\Local"
    $script:RoamApp   = Join-Path $sb "user\AppData\Roaming"
    New-Item -ItemType Directory -Force -Path $script:UserProf,$script:LocalApp,$script:RoamApp | Out-Null

    & $Setup

    $old = @{}
    foreach ($k in @("USERPROFILE","LOCALAPPDATA","APPDATA","UNSLOTH_STUDIO_HOME","STUDIO_HOME")) {
        $old[$k] = [Environment]::GetEnvironmentVariable($k)
    }
    try {
        $env:USERPROFILE  = $script:UserProf
        $env:LOCALAPPDATA = $script:LocalApp
        $env:APPDATA      = $script:RoamApp
        $env:UNSLOTH_STUDIO_HOME = $null
        $env:STUDIO_HOME         = $null
        if ($StudioHomeIsUserProfile) { $env:UNSLOTH_STUDIO_HOME = $script:UserProf }
        # -File, not dot-source: uninstall.ps1 ends with `Uninstall-UnslothStudio @args`,
        # so dot-sourcing runs it immediately and a second explicit call would make each
        # case run the uninstaller twice.
        $out = & pwsh -NoProfile -File $uninstall 2>&1 | Out-String
        return $out
    } finally {
        foreach ($k in $old.Keys) {
            if ($null -eq $old[$k]) { Remove-Item "env:$k" -ErrorAction SilentlyContinue }
            else { Set-Item -Path "env:$k" -Value $old[$k] }
        }
    }
}

# ------------------------------------------- 1. WebView data + blast radius ----
Write-Host "--- 1. WebView data removed, look-alike bundle ids untouched ---"
$out = Invoke-Sandboxed {
    New-Item -ItemType Directory -Force -Path `
        (Join-Path $script:LocalApp "$bid\EBWebView\Default\Cache"),
        (Join-Path $script:LocalApp "$bid\EBWebView\Default\Local Storage"),
        (Join-Path $script:LocalApp "${bid}2"),
        (Join-Path $script:LocalApp "$bid.other"),
        (Join-Path $script:LocalApp "com.other.app"),
        (Join-Path $script:RoamApp  "$bid"),
        (Join-Path $script:UserProf "Documents") | Out-Null
    "x" | Set-Content (Join-Path $script:LocalApp "$bid\EBWebView\Default\Cache\stale.js")
    "x" | Set-Content (Join-Path $script:LocalApp "${bid}2\keepme")
    "x" | Set-Content (Join-Path $script:LocalApp "$bid.other\keepme")
    "x" | Set-Content (Join-Path $script:LocalApp "com.other.app\keepme")
    "x" | Set-Content (Join-Path $script:UserProf "Documents\notes.txt")
}
AssertGone    "LOCALAPPDATA\$bid removed"           (Join-Path $script:LocalApp $bid)
AssertGone    "APPDATA\$bid removed"                (Join-Path $script:RoamApp  $bid)
AssertPresent "look-alike ${bid}2 survived"         (Join-Path $script:LocalApp "${bid}2\keepme")
AssertPresent "look-alike $bid.other survived"      (Join-Path $script:LocalApp "$bid.other\keepme")
AssertPresent "unrelated app survived"              (Join-Path $script:LocalApp "com.other.app\keepme")
AssertPresent "user documents survived"             (Join-Path $script:UserProf "Documents\notes.txt")

# --------------------------------------------- 2. studio.db summary wording ----
# The scoped claim ("the studio.db it found") must only appear when a database was
# actually deleted. This is the Windows mirror of the shell-side _DB_REMOVED_FLAG.
Write-Host "--- 2. summary only claims the history is gone when studio.db was removed ---"
$out = Invoke-Sandboxed {
    New-Item -ItemType Directory -Force -Path (Join-Path $script:UserProf ".unsloth\studio\share") | Out-Null
    "x" | Set-Content (Join-Path $script:UserProf ".unsloth\studio\share\studio.conf")
    "x" | Set-Content (Join-Path $script:UserProf ".unsloth\studio\studio.db")
}
AssertGone "default-mode studio.db removed" (Join-Path $script:UserProf ".unsloth\studio\studio.db")
if ($out -match "studio.db it found") { Ok "studio.db removed -> summary states the history is gone" }
else { Bad "studio.db was removed but the summary never says so" }

$out = Invoke-Sandboxed {
    New-Item -ItemType Directory -Force -Path (Join-Path $script:LocalApp "$bid") | Out-Null
}
if ($out -match "studio.db it found") { Bad "claimed the history is gone with no studio.db removed" }
else { Ok "no studio.db removed -> no claim that the history is gone" }

# ------------------------------------------------------ 3. env-mode blindness --
# An env-mode install keeps studio.db inside the custom root and leaves no breadcrumb
# in the user profile, so a bare run cannot find it and must not claim it is gone.
Write-Host "--- 3. undiscovered env-mode root ---"
$envRootHolder = $null
$out = Invoke-Sandboxed {
    $script:EnvRoot = Join-Path $script:Sandbox "envroot"
    New-Item -ItemType Directory -Force -Path (Join-Path $script:EnvRoot "share") | Out-Null
    "x" | Set-Content (Join-Path $script:EnvRoot "share\studio.conf")
    "x" | Set-Content (Join-Path $script:EnvRoot "studio.db")
    $envRootHolder = $script:EnvRoot
}
AssertPresent "undiscovered env-mode studio.db survives" (Join-Path $script:EnvRoot "studio.db")
if ($out -match "studio.db it found") { Bad "claimed history gone with an undiscovered studio.db on disk" }
else { Ok "undiscovered env-mode root -> no claim that the history is gone" }

# ------------------------------------- 3b. deny-listed root is incomplete cleanup --
# install.ps1 accepts any writable root, so a deny-listed path can hold a real install.
# Skipping it must count as incomplete cleanup, exactly as uninstall.sh does, or the
# summary claims the keys and history are gone while that root still holds them.
Write-Host "--- 3b. deny-listed custom root ---"
$out = Invoke-Sandboxed {
    # USERPROFILE itself is on the deny list, and studio.conf makes it a valid Studio root.
    New-Item -ItemType Directory -Force -Path (Join-Path $script:UserProf "share") | Out-Null
    "x" | Set-Content (Join-Path $script:UserProf "share\studio.conf")
    "x" | Set-Content (Join-Path $script:UserProf "studio.db")
    New-Item -ItemType Directory -Force -Path (Join-Path $script:UserProf ".unsloth\studio\share") | Out-Null
    "x" | Set-Content (Join-Path $script:UserProf ".unsloth\studio\share\studio.conf")
    "x" | Set-Content (Join-Path $script:UserProf ".unsloth\studio\studio.db")
} -StudioHomeIsUserProfile
AssertPresent "deny-listed root left alone" (Join-Path $script:UserProf "studio.db")
($out -split "`n") | Where-Object { $_ -match "refusing|Note:" } | ForEach-Object { Write-Host "    | $($_.Trim())" }
if ($out -match "studio.db it found") {
    Bad "deny-listed root still claimed the keys and history are gone"
} else {
    Ok "deny-listed root counts as incomplete cleanup"
}

# ------------------------------------- 3c. WebView2 helper tree, ancestor walk ----
# Extracts the real $isOurs scriptblock from uninstall.ps1 and drives it against a
# synthetic process table, since a live WebView2 tree cannot be produced on a runner.
# WebView2 is Chromium's process model: helpers are children of the BROWSER process,
# not of unsloth-studio.exe, so an immediate-parent test matches only the browser.
Write-Host "--- 3c. WebView2 ancestor walk ---"
$src = Get-Content -Raw -LiteralPath $uninstall
$m = [regex]::Match($src, '(?s)\$isOurs = (\{.*?\n            \})')
if (-not $m.Success) { Bad "could not extract the \$isOurs scriptblock"; }
else {
    $studioPids = @(1000)
    #  1000 unsloth-studio.exe
    #   2000 msedgewebview2.exe  (browser)   parent 1000
    #     3000 renderer                      parent 2000
    #     3001 gpu                           parent 2000
    #   9000 msedgewebview2.exe of ANOTHER app, parent 8000
    $parentOf = @{ 2000 = 1000; 3000 = 2000; 3001 = 2000; 9000 = 8000 }
    # The captured text still has its outer braces, so Create() alone yields a scriptblock
    # whose body is a scriptblock LITERAL: invoking it just emits the inner block, which is
    # always truthy. Evaluate the literal once to get the block itself.
    $isOurs = & ([scriptblock]::Create($m.Groups[1].Value))
    if ($isOurs -isnot [scriptblock]) { Bad "extraction did not yield a scriptblock" }
    if (& $isOurs 2000) { Ok "browser process recognised" } else { Bad "browser process missed" }
    if (& $isOurs 3000) { Ok "renderer helper recognised (grandchild)" } else { Bad "renderer helper missed" }
    if (& $isOurs 3001) { Ok "gpu helper recognised (grandchild)" } else { Bad "gpu helper missed" }
    if (& $isOurs 9000) { Bad "another app's WebView2 wrongly claimed" } else { Ok "another app's WebView2 left alone" }
    # A parent cycle must terminate rather than spin.
    $parentOf = @{ 5000 = 5001; 5001 = 5000 }
    if (& $isOurs 5000) { Bad "cycle wrongly matched" } else { Ok "parent cycle terminates without matching" }
}

# ------------------------------------------------- 4. arguments are rejected ---
Write-Host "--- 4. unknown arguments never trigger removal ---"
$sb = Join-Path ([System.IO.Path]::GetTempPath()) ("unsl_" + [guid]::NewGuid().ToString("N").Substring(0,8))
New-Item -ItemType Directory -Force -Path "$sb\user\AppData\Local\$bid" | Out-Null
$oldUP = $env:USERPROFILE; $oldLA = $env:LOCALAPPDATA
try {
    $env:USERPROFILE  = "$sb\user"
    $env:LOCALAPPDATA = "$sb\user\AppData\Local"
    # Pass the argument the way a user would, through the script's own `@args`.
    & pwsh -NoProfile -File $uninstall --nope 2>&1 | Out-Null
    $rc = $LASTEXITCODE
} finally { $env:USERPROFILE = $oldUP; $env:LOCALAPPDATA = $oldLA }
if ($rc -ne 0) { Ok "unknown argument fails instead of removing (rc=$rc)" } else { Bad "unknown argument exited 0" }
AssertPresent "unknown argument left the data alone" "$sb\user\AppData\Local\$bid"

Write-Host ""
Write-Host "Results: $script:Pass passed, $script:Fail failed"
if ($script:Fail -gt 0) { exit 1 }
