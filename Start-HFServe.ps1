<#
.SYNOPSIS
    Starter script for hfserve, the local HuggingFace model server.

.DESCRIPTION
    Without parameters an interactive menu is shown: start / stop the server, chat,
    switch the loaded model, download a model, list installed models.

    Installed models are read from models\models.json (written by hfserve itself on
    every pull / rm / list). Model names in that file are what -Model accepts.

    Parameters skip the menu:
      -Start                 serve the first installed model (or -Model) in this window
      -Start -Background     start the server minimized in its own window and return
      -Chat                  open the console chat (hfserve run); \bye stops the server
      -Stop                  stop the server on -Port (and everything it spawned)
      -List                  print the installed models and exit
      -Pull <url>            download a model from huggingface.co, then exit
      -Model <name or index> model name or list number from models.json (default: first)
      -Port <n>              server port (default: $env:HFSERVE_PORT or 11435)
      -Refresh               rebuild models.json before doing anything else

.EXAMPLE
    .\Start-HFServe.ps1
    .\Start-HFServe.ps1 -Start
    .\Start-HFServe.ps1 -Start -Model K2-Horizon-0.9B -Background
    .\Start-HFServe.ps1 -Chat
    .\Start-HFServe.ps1 -Pull https://huggingface.co/IFM/K2-Horizon-0.9B
    .\Start-HFServe.ps1 -Stop

.NOTES
    Works with Windows PowerShell 5.1 and PowerShell 7. If scripts are blocked, run once:
      Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#>
[CmdletBinding()]
param(
    [switch]$Start,
    [switch]$Background,
    [switch]$Chat,
    [switch]$Stop,
    [switch]$List,
    [string]$Pull = "",
    [string]$Model = "",
    [int]$Port = 0,
    [string]$ServerHost = "",
    [switch]$Refresh
)

# "Continue": native tools (tqdm progress, uvicorn logs) write to stderr; that must not abort us.
$ErrorActionPreference = "Continue"
$Root = $PSScriptRoot
if (-not $Root) { $Root = Split-Path -Parent $MyInvocation.MyCommand.Path }

# ----------------------------------------------------------------------------- settings
if ($Port -le 0) {
    if ($env:HFSERVE_PORT) { $Port = [int]$env:HFSERVE_PORT } else { $Port = 11435 }
}
if (-not $ServerHost) {
    if ($env:HFSERVE_HOST) { $ServerHost = $env:HFSERVE_HOST } else { $ServerHost = "127.0.0.1" }
}
$ConnectHost = $ServerHost
if ($ConnectHost -eq "0.0.0.0" -or $ConnectHost -eq "::") { $ConnectHost = "127.0.0.1" }
$Url = "http://${ConnectHost}:${Port}"

function Get-DotEnvValue {
    # First non-empty value of $Names in .env next to this script ("" when absent).
    # Same keys and precedence as hfserve/config.py, so the script and the server
    # always agree on which models folder is in use.
    param([string[]]$Names)
    $envFile = Join-Path $Root ".env"
    if (-not (Test-Path $envFile)) { return "" }
    $lines = @(Get-Content $envFile -ErrorAction SilentlyContinue)
    foreach ($name in $Names) {
        foreach ($line in $lines) {
            $text = $line.Trim()
            if (-not $text -or $text.StartsWith("#")) { continue }
            $i = $text.IndexOf("=")
            if ($i -lt 1) { continue }
            if ($text.Substring(0, $i).Trim() -ine $name) { continue }
            $value = $text.Substring($i + 1).Trim().Trim('"').Trim("'")
            if ($value) { return $value }
        }
    }
    return ""
}

# Models folder: a real HFSERVE_MODELS wins, then HFSERVE_MODELS / MODELS_DIR from
# .env (a relative value is relative to this script), else <Root>\models.
if ($env:HFSERVE_MODELS) {
    $ModelsDir = $env:HFSERVE_MODELS
} else {
    $FromDotEnv = Get-DotEnvValue @("HFSERVE_MODELS", "MODELS_DIR")
    if ($FromDotEnv) {
        if ([System.IO.Path]::IsPathRooted($FromDotEnv)) { $ModelsDir = $FromDotEnv }
        else { $ModelsDir = Join-Path $Root $FromDotEnv }
    } else {
        $ModelsDir = Join-Path $Root "models"
    }
}
try { $ModelsDir = [System.IO.Path]::GetFullPath($ModelsDir) } catch { }
if (-not (Test-Path $ModelsDir)) {
    Write-Host "The models folder does not exist: $ModelsDir" -ForegroundColor Yellow
    Write-Host "Fix HFSERVE_MODELS / MODELS_DIR (environment or .env), or download a model with -Pull <url>." -ForegroundColor Yellow
}
$IndexFile = Join-Path $ModelsDir "models.json"

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

# ----------------------------------------------------------------------------- helpers
function Invoke-HFServe {
    # Run `python -m hfserve <args>` in the project folder. Call it as a statement (never
    # assign its result): that way the child inherits the console, so streamed chat
    # output and progress bars show up live. Read $LASTEXITCODE afterwards.
    param([string[]]$Arguments)
    Push-Location $Root
    try {
        & $Python -m hfserve --host $ServerHost --port $Port @Arguments
    } finally {
        Pop-Location
    }
}

function Get-ModelIndex {
    # Installed models from models.json; (re)builds the file through hfserve when needed.
    param([switch]$Rebuild)
    if ($Rebuild -or -not (Test-Path $IndexFile)) {
        Invoke-HFServe @("list", "--json") | Out-Null   # quiet: only the file matters here
    }
    if (-not (Test-Path $IndexFile)) { return @() }
    $raw = Get-Content -Path $IndexFile -Raw -Encoding UTF8
    if (-not $raw.Trim()) { return @() }
    $index = $raw | ConvertFrom-Json
    return @($index.models)
}

function Show-Models {
    param($Models)
    if (-not $Models -or $Models.Count -eq 0) {
        Write-Host "  (no models installed - use the download option or -Pull <url>)" -ForegroundColor Yellow
        return
    }
    $width = ($Models | ForEach-Object { $_.name.Length } | Measure-Object -Maximum).Maximum
    $i = 0
    foreach ($m in $Models) {
        $i++
        $flag = ""
        if ($m.incomplete) { $flag = "  (incomplete download)" }
        $line = "  {0,2}  {1,-$width}  {2,8}  {3}{4}" -f $i, $m.name, $m.size, $m.model_type, $flag
        Write-Host $line
    }
}

function Resolve-ModelRef {
    # $Ref: list index, full name (org/name), short name, or unique substring.
    param($Models, [string]$Ref)
    if (-not $Models -or $Models.Count -eq 0) { return $null }
    $Ref = $Ref.Trim()
    if (-not $Ref) { return $Models[0] }
    $n = 0
    if ([int]::TryParse($Ref, [ref]$n)) {
        if ($n -ge 1 -and $n -le $Models.Count) { return $Models[$n - 1] }
        return $null
    }
    foreach ($m in $Models) { if ($m.name -ieq $Ref -or $m.short_name -ieq $Ref) { return $m } }
    $subs = @($Models | Where-Object { $_.name -like "*$Ref*" })
    if ($subs.Count -eq 1) { return $subs[0] }
    return $null
}

function Read-Answer {
    # Read-Host that yields "q" when no interactive console is available (piped / -NonInteractive),
    # so the menu can never spin forever.
    param([string]$Prompt)
    try {
        $answer = Read-Host $Prompt
    } catch {
        return "q"
    }
    if ($null -eq $answer) { return "q" }
    return $answer
}

function Select-ModelInteractive {
    param($Models, [string]$Prompt = "Model")
    if (-not $Models -or $Models.Count -eq 0) { return $null }
    Show-Models $Models
    $answer = Read-Answer "$Prompt (number or name, Enter = 1, Q = cancel)"
    if ($answer -ieq "q") { return $null }
    $m = Resolve-ModelRef $Models $answer
    if (-not $m) { Write-Host "No installed model matches '$answer'." -ForegroundColor Red }
    return $m
}

function Get-ServerStatus {
    try {
        $h = Invoke-RestMethod -Uri "$Url/api/health" -TimeoutSec 3
        if ($h.app -eq "hfserve") {
            return [pscustomobject]@{ Running = $true; Model = $h.model; Pid = $h.pid; Device = $h.device }
        }
    } catch { }
    return [pscustomobject]@{ Running = $false; Model = $null; Pid = $null; Device = $null }
}

function Format-Status {
    param($Status)
    if (-not $Status.Running) { return "$Url  (not running)" }
    if ($Status.Model) { return "$Url  (running, pid $($Status.Pid), model $($Status.Model) on $($Status.Device))" }
    return "$Url  (running, pid $($Status.Pid), no model loaded)"
}

function Clear-ModelMemory {
    # Release the VRAM (and pinned RAM) a loaded model holds, through POST /api/unload.
    # Always safe to call: with no server running there is nothing to clear, and the
    # model comes back on demand because every /api/chat goes through ensure_loaded.
    param([switch]$Quiet)
    $status = Get-ServerStatus
    if (-not $status.Running) {
        if (-not $Quiet) { Write-Host "No server running - no model memory to clear." -ForegroundColor DarkGray }
        return
    }
    if (-not $status.Model) {
        if (-not $Quiet) { Write-Host "No model loaded - memory is already clear." -ForegroundColor DarkGray }
        return
    }
    Write-Host "Clearing model memory ($($status.Model)) ..." -NoNewline
    try {
        $null = Invoke-RestMethod -Method Post -Uri "$Url/api/unload" -TimeoutSec 120
        Write-Host " done." -ForegroundColor Green
    } catch {
        Write-Host " failed: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

function Wait-ForServer {
    param([int]$TimeoutSec = 300)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        $s = Get-ServerStatus
        if ($s.Running) { return $s }
        Start-Sleep -Milliseconds 500
    }
    return $null
}

function Start-Server {
    param($ModelObj, [bool]$Detached)
    $status = Get-ServerStatus
    if ($status.Running) {
        Write-Host "Server is already running: $(Format-Status $status)" -ForegroundColor Yellow
        if ($ModelObj -and $status.Model -ne $ModelObj.name) { Switch-Model $ModelObj }
        return
    }
    $serveArgs = @("--host", $ServerHost, "--port", "$Port", "serve")
    if ($ModelObj) { $serveArgs += @("--model", $ModelObj.name) }
    if ($Detached) {
        Write-Host "Starting server in its own window (model: $($ModelObj.name)) ..."
        Start-Process -FilePath $Python -ArgumentList (@("-m", "hfserve") + $serveArgs) `
            -WorkingDirectory $Root -WindowStyle Minimized | Out-Null
        $s = Wait-ForServer
        if ($s) {
            Write-Host "Server running: $(Format-Status $s)" -ForegroundColor Green
            Write-Host "  API: POST $Url/api/chat   POST $Url/v1/chat/completions   GET $Url/api/tags"
        } else {
            Write-Host "Server did not answer within the timeout; check the minimized window." -ForegroundColor Red
        }
    } else {
        Write-Host "Starting server in this window (Ctrl+C stops it and ends this script) ..."
        Push-Location $Root
        try { & $Python -m hfserve @serveArgs } finally { Pop-Location }
    }
}

function Stop-Server {
    # Unload first, so the weights are released before the process tree is taken down
    # rather than relying on the dying process to free the GPU.
    Clear-ModelMemory -Quiet
    Invoke-HFServe @("stop")
}

function Switch-Model {
    param($ModelObj)
    $status = Get-ServerStatus
    if (-not $status.Running) { Write-Host "Server is not running." -ForegroundColor Yellow; return }
    Write-Host "Loading $($ModelObj.name) ..." -NoNewline
    try {
        $body = @{ model = $ModelObj.name } | ConvertTo-Json -Compress
        $r = Invoke-RestMethod -Method Post -Uri "$Url/api/load" -Body $body -ContentType "application/json" -TimeoutSec 900
        $vram = ""
        if ($r.vram_bytes) { $vram = " (VRAM {0:N1} GB)" -f ($r.vram_bytes / 1GB) }
        Write-Host (" done: {0} / {1} in {2:N1}s{3}" -f $r.device, $r.dtype, $r.load_seconds, $vram) -ForegroundColor Green
    } catch {
        Write-Host ""
        Write-Host "Load failed: $($_.Exception.Message)" -ForegroundColor Red
    }
}

function Start-Chat {
    param($ModelObj)
    Write-Host "Opening chat with $($ModelObj.name). Type \bye to stop the server and return here." -ForegroundColor Cyan
    Invoke-HFServe @("run", $ModelObj.name)
}

function Invoke-Pull {
    param([string]$Ref)
    if (-not $Ref) { $Ref = Read-Answer "huggingface.co URL or org/name" }
    if (-not $Ref -or $Ref -ieq "q") { return 1 }
    Invoke-HFServe @("pull", $Ref)
    $rc = $LASTEXITCODE
    if ($rc -eq 0) { Write-Host "Model list refreshed ($IndexFile)." -ForegroundColor Green }
    return $rc
}

# ----------------------------------------------------------------------------- non-interactive
# Always begin on a clean GPU: whatever an earlier session left loaded is released now,
# and released again in the finally at the bottom, however this script ends (including
# `exit` - PowerShell still runs finally - and Ctrl+C out of a foreground server).
Clear-ModelMemory -Quiet
try {

if ($Refresh) { $null = Get-ModelIndex -Rebuild }

if ($Pull) {
    $rc = Invoke-Pull $Pull
    exit $rc
}

if ($List) {
    $models = @(Get-ModelIndex)
    Write-Host "Installed models ($IndexFile):"
    Show-Models $models
    exit 0
}

if ($Stop) {
    Stop-Server
    exit 0
}

if ($Start -or $Chat) {
    $models = @(Get-ModelIndex)
    if (-not $models -or $models.Count -eq 0) {
        Write-Host "No models installed. Download one first:  .\Start-HFServe.ps1 -Pull https://huggingface.co/IFM/K2-Horizon-0.9B" -ForegroundColor Red
        exit 1
    }
    $m = Resolve-ModelRef $models $Model
    if (-not $m) {
        Write-Host "No installed model matches '$Model'. Installed:" -ForegroundColor Red
        Show-Models $models
        exit 1
    }
    if ($Chat) {
        Start-Chat $m
    } else {
        Start-Server $m $Background.IsPresent
    }
    exit 0
}

# ----------------------------------------------------------------------------- menu
$models = @(Get-ModelIndex)
while ($true) {
    $status = Get-ServerStatus
    Write-Host ""
    Write-Host "hfserve starter  -  $Root" -ForegroundColor Cyan
    Write-Host "Server : $(Format-Status $status)"
    Write-Host "Models : $IndexFile"
    Show-Models $models
    Write-Host ""
    if ($status.Running) {
        Write-Host "  [1] Switch model          [4] Download a model"
        Write-Host "  [2] Chat (\bye stops the server)"
        Write-Host "  [3] Stop server           [5] Refresh model list"
    } else {
        Write-Host "  [1] Start server (background window)   [4] Download a model"
        Write-Host "  [2] Chat with a model (starts the server)"
        Write-Host "  [3] Start server in this window (Ctrl+C ends it)   [5] Refresh model list"
    }
    Write-Host "  [Q] Quit"
    $choice = (Read-Answer "Choice").Trim().ToLower()
    switch ($choice) {
        "1" {
            $m = Select-ModelInteractive $models
            if ($m) {
                if ($status.Running) { Switch-Model $m } else { Start-Server $m $true }
            }
        }
        "2" {
            $m = $null
            if ($status.Running -and $status.Model) { $m = Resolve-ModelRef $models $status.Model }
            if (-not $m) { $m = Select-ModelInteractive $models }
            if ($m) { Start-Chat $m }
        }
        "3" {
            if ($status.Running) {
                Stop-Server
            } else {
                $m = Select-ModelInteractive $models
                if ($m) { Start-Server $m $false }
            }
        }
        "4" { $null = Invoke-Pull ""; $models = @(Get-ModelIndex) }
        "5" { $models = @(Get-ModelIndex -Rebuild) }
        "q" { }
        "quit" { }
        "exit" { }
        "" { }
        default { Write-Host "Unknown choice '$choice'." -ForegroundColor Red }
    }
    # `break` inside `switch` only leaves the switch, so the loop exit is checked here.
    if ($choice -eq "q" -or $choice -eq "quit" -or $choice -eq "exit") { break }
}

} finally {
    # Never leave a model sitting in VRAM behind us. A server started with -Background
    # keeps running and reloads the model by itself on the next request.
    Clear-ModelMemory -Quiet
}
