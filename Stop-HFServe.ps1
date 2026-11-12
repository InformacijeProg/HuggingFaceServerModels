<#
.SYNOPSIS
    Stop hfserve: release the loaded model's memory first, then stop the server.

.DESCRIPTION
    Freeing the GPU is the point of this script, so the model is unloaded through
    POST /api/unload *before* the server is taken down, instead of trusting a dying
    process to give the VRAM back. The memory is reported before and after, so the
    result is visible rather than assumed.

    Host and port come from hfserve.config - that is, from the environment or from
    .env (HFSERVE_HOST / HFSERVE_PORT, or a whole SERVER_URL) - so this script and
    the server always agree on the address. -Port / -ServerHost override them.

.PARAMETER All
    Stop every hfserve server on this machine, not only the one on -Port.

.EXAMPLE
    .\Stop-HFServe.ps1
    Clear the model from memory and stop the server on the configured port.

.EXAMPLE
    .\Stop-HFServe.ps1 -All
    Same, then stop every other hfserve server too.
#>
param(
    [int]$Port = 0,
    [string]$ServerHost = "",
    [switch]$All
)

$ErrorActionPreference = "Continue"
$Root = $PSScriptRoot
if (-not $Root) { $Root = Split-Path -Parent $MyInvocation.MyCommand.Path }

function Find-Python {
    foreach ($candidate in @(
        (Join-Path $Root ".vemv\Scripts\python.exe"),
        (Join-Path $Root ".venv\Scripts\python.exe"))) {
        if (Test-Path $candidate) { return $candidate }
    }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    throw "Python not found. Create the venv first (see README.md)."
}
$Python = Find-Python

# Ask hfserve where its server is, so the environment and .env are read in exactly one
# place (hfserve/config.py) instead of being parsed a second time here.
if ($Port -le 0 -or -not $ServerHost) {
    Push-Location $Root
    try {
        $resolved = @(& $Python -c "from hfserve import config; print(config.DEFAULT_HOST); print(config.DEFAULT_PORT)")
    } finally {
        Pop-Location
    }
    if ($resolved.Count -ge 2) {
        if (-not $ServerHost) { $ServerHost = $resolved[0].Trim() }
        if ($Port -le 0) { $Port = [int]$resolved[1].Trim() }
    }
}
if (-not $ServerHost) { $ServerHost = "127.0.0.1" }
if ($Port -le 0) { $Port = 11435 }

$ConnectHost = $ServerHost
if ($ConnectHost -eq "0.0.0.0" -or $ConnectHost -eq "::") { $ConnectHost = "127.0.0.1" }
$Url = "http://${ConnectHost}:${Port}"

function Get-ServerStatus {
    try {
        $h = Invoke-RestMethod -Uri "$Url/api/health" -TimeoutSec 3
        if ($h.app -eq "hfserve") {
            return [pscustomobject]@{ Running = $true; Model = $h.model; Pid = $h.pid; Device = $h.device }
        }
    } catch { }
    return [pscustomobject]@{ Running = $false; Model = $null; Pid = $null; Device = $null }
}

function Get-GpuLine {
    # One "used / total" line for the first NVIDIA GPU, or $null when there is none.
    try {
        $csv = & nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits
    } catch {
        return $null
    }
    if ($LASTEXITCODE -ne 0 -or -not $csv) { return $null }
    $fields = @(([string]@($csv)[0]).Split(",") | ForEach-Object { $_.Trim() })
    if ($fields.Count -lt 3) { return $null }
    return "{0}: VRAM {1:N1} / {2:N1} GB in use" -f $fields[0], ([double]$fields[1] / 1024), ([double]$fields[2] / 1024)
}

function Clear-ModelMemory {
    # Release the weights a loaded model holds. Safe with no server and safe twice: the
    # model comes back on demand, because every /api/chat goes through ensure_loaded.
    $status = Get-ServerStatus
    if (-not $status.Running) {
        Write-Host "No server answering on $Url - nothing loaded to clear." -ForegroundColor DarkGray
        return
    }
    if (-not $status.Model) {
        Write-Host "Server is running with no model loaded - memory is already clear." -ForegroundColor DarkGray
        return
    }
    Write-Host "Clearing model memory ($($status.Model) on $($status.Device)) ..." -NoNewline
    try {
        $null = Invoke-RestMethod -Method Post -Uri "$Url/api/unload" -TimeoutSec 120
        Write-Host " done." -ForegroundColor Green
    } catch {
        Write-Host " failed: $($_.Exception.Message)" -ForegroundColor Yellow
        Write-Host "Stopping the server anyway - that releases the memory with the process." -ForegroundColor Yellow
    }
}

# ----------------------------------------------------------------------------- main
Write-Host ""
Write-Host "hfserve stop  -  $Url" -ForegroundColor Cyan
$before = Get-GpuLine
if ($before) { Write-Host "Before : $before" }

$status = Get-ServerStatus
Clear-ModelMemory

if ($status.Running -or $All) {
    $stopArgs = @("--host", $ServerHost, "--port", "$Port", "stop")
    if ($All) { $stopArgs += "--all" }
    Push-Location $Root
    try {
        & $Python -m hfserve @stopArgs
    } finally {
        Pop-Location
    }
} else {
    Write-Host "No server to stop." -ForegroundColor DarkGray
}

$after = Get-GpuLine
if ($after) { Write-Host "After  : $after" }

if ((Get-ServerStatus).Running) {
    Write-Host "A server is still answering on $Url." -ForegroundColor Yellow
    exit 1
}
Write-Host "Model memory cleared and no server running on $Url." -ForegroundColor Green
exit 0
