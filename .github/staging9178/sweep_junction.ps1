$ErrorActionPreference = 'Stop'
. "$env:RUNNER_TEMP\sweep.ps1"
Remove-StudioStalePrivateTempDirectories -Root $env:SWEEP_ROOT
