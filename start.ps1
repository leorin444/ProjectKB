<#
.SYNOPSIS
    Starts the Local Multimodal Knowledge Base (Ollama + FastAPI backend + Streamlit UI).
.DESCRIPTION
    1. Ensures Ollama is running (starts it if not).
    2. Checks required Ollama models are pulled (offers to pull missing ones).
    3. Boots the FastAPI backend on port 8000.
    4. Waits for /health to respond, then launches the Streamlit UI on port 8501.
    5. Opens your browser automatically.
    6. Cleanly shuts down all services on exit.
#>

[CmdletBinding()]
param (
    [string]$ApiHost   = "127.0.0.1",
    [int]$ApiPort      = 8000,
    [int]$UiPort       = 8501,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

# Navigate to project root (works whether run from project dir or scripts dir)
$ProjectRoot = $PSScriptRoot
if (-not (Test-Path "$ProjectRoot\app")) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}
if (-not (Test-Path "$ProjectRoot\app")) {
    $ProjectRoot = Get-Location
}
Set-Location $ProjectRoot

# Read models from .env if present
$EnvFile = "$ProjectRoot\.env"
$RequiredLlmModel    = "llama3.2:1b"
$RequiredVisionModel = "moondream"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match "^LLM_MODEL=(.+)$")    { $RequiredLlmModel    = $Matches[1].Trim() }
        if ($_ -match "^VISION_MODEL=(.+)$") { $RequiredVisionModel = $Matches[1].Trim() }
    }
}

$VenvPython = "$ProjectRoot\.venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host ""
    Write-Host "  ❌ Virtual environment not found at $VenvPython" -ForegroundColor Red
    Write-Host "  Please run setup.ps1 first." -ForegroundColor Yellow
    exit 1
}

# Track what we started so we can stop it on exit
$ollamaProcess = $null
$apiProcess    = $null
$uiProcess     = $null

# ──────────────────────────────────────────────────────────────────────────────
# Visual Banner
# ──────────────────────────────────────────────────────────────────────────────
Clear-Host
Write-Host ""
Write-Host "  =================================================================" -ForegroundColor Cyan
Write-Host "       🧠  LOCAL MULTIMODAL KNOWLEDGE BASE  (Project KB)          " -ForegroundColor Cyan
Write-Host "  =================================================================" -ForegroundColor Cyan
Write-Host "   Privacy : 100% Offline — Zero Cloud Egress" -ForegroundColor DarkGray
Write-Host "   Backend : FastAPI   →  http://$ApiHost`:$ApiPort" -ForegroundColor DarkGray
Write-Host "   Frontend: Streamlit →  http://localhost:$UiPort" -ForegroundColor DarkGray
Write-Host "  =================================================================" -ForegroundColor Cyan
Write-Host ""

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: Ensure Ollama is running
# ──────────────────────────────────────────────────────────────────────────────
Write-Host "  [1/4] Ollama Service..." -ForegroundColor Yellow

$ollamaRunning = $false
try {
    $null = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 3
    $ollamaRunning = $true
    Write-Host "        ✅ Ollama already running." -ForegroundColor Green
} catch {
    Write-Host "        ⚡ Ollama not detected — starting it now..." -ForegroundColor DarkGray
    $ollamaExe = Get-Command "ollama" -ErrorAction SilentlyContinue
    if (-not $ollamaExe) {
        Write-Host "        ❌ 'ollama' not found on PATH." -ForegroundColor Red
        Write-Host "           Download from https://ollama.com and re-run." -ForegroundColor Yellow
        exit 1
    }

    $ollamaProcess = Start-Process -FilePath "ollama" -ArgumentList "serve" -PassThru -WindowStyle Hidden

    # Wait up to 15s for Ollama to become ready
    Write-Host -NoNewline "        Waiting for Ollama to start" -ForegroundColor DarkGray
    for ($i = 0; $i -lt 25; $i++) {
        Start-Sleep -Milliseconds 600
        try {
            $null = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 2
            $ollamaRunning = $true
            break
        } catch {
            Write-Host -NoNewline "." -ForegroundColor DarkGray
        }
    }

    if ($ollamaRunning) {
        Write-Host " ✅ Ready!" -ForegroundColor Green
    } else {
        Write-Host " ⚠️ Ollama did not start in time — Q&A may not work." -ForegroundColor Yellow
    }
}

# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: Check / pull required models
# ──────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  [2/4] Checking required Ollama models..." -ForegroundColor Yellow

if ($ollamaRunning) {
    try {
        $tagResp       = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 5
        $availableNames = $tagResp.models | ForEach-Object { $_.name }

        foreach ($model in @($RequiredLlmModel, $RequiredVisionModel)) {
            $short = $model.Split(":")[0]
            $present = $availableNames | Where-Object { $_ -like "$short*" -or $_ -eq $model }
            if ($present) {
                Write-Host "        ✅ $model — found" -ForegroundColor Green
            } else {
                Write-Host "        ⬇️  $model — not found, pulling now..." -ForegroundColor Yellow
                & ollama pull $model
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "        ✅ $model — pulled successfully" -ForegroundColor Green
                } else {
                    Write-Host "        ⚠️ Pull failed for $model (check internet connection)" -ForegroundColor Red
                }
            }
        }
    } catch {
        Write-Host "        ⚠️ Could not query Ollama model list: $_" -ForegroundColor Yellow
    }
} else {
    Write-Host "        ⚠️ Skipping model check (Ollama not available)." -ForegroundColor DarkGray
}

# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: Launch FastAPI Backend
# ──────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  [3/4] Starting API Backend..." -ForegroundColor Yellow
$apiProcess = Start-Process -FilePath $VenvPython `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--host", $ApiHost, "--port", "$ApiPort", "--log-level", "warning" `
    -WorkingDirectory $ProjectRoot `
    -PassThru

$healthUrl  = "http://$ApiHost`:$ApiPort/health"
$maxRetries = 25
$healthy    = $false
Write-Host -NoNewline "        Waiting for API server" -ForegroundColor DarkGray

for ($i = 0; $i -lt $maxRetries; $i++) {
    Start-Sleep -Milliseconds 600
    try {
        $resp = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($resp -and $resp.status) { $healthy = $true; break }
    } catch {
        Write-Host -NoNewline "." -ForegroundColor DarkGray
    }
}

if ($healthy) {
    Write-Host " ✅ Ready!" -ForegroundColor Green
} else {
    Write-Host " ⚠️ Timeout (API may still be starting up)" -ForegroundColor Yellow
}

# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: Launch Streamlit Frontend
# ──────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  [4/4] Starting Streamlit UI..." -ForegroundColor Yellow
$uiArgs = @("-m", "streamlit", "run", "$ProjectRoot\ui\streamlit_app.py",
            "--server.port", "$UiPort", "--server.headless", "true")

$uiProcess = Start-Process -FilePath $VenvPython `
    -ArgumentList $uiArgs `
    -WorkingDirectory $ProjectRoot `
    -PassThru

Start-Sleep -Seconds 2

# ──────────────────────────────────────────────────────────────────────────────
# Ready banner
# ──────────────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  =================================================================" -ForegroundColor Green
Write-Host "   🎉  Project KB is ready!" -ForegroundColor Green
Write-Host ""
Write-Host "   🌐  Web Interface  →  http://localhost:$UiPort" -ForegroundColor White
Write-Host "   📖  API Docs       →  http://$ApiHost`:$ApiPort/docs" -ForegroundColor White
Write-Host "   💚  Health Status  →  http://$ApiHost`:$ApiPort/health" -ForegroundColor White
Write-Host ""
Write-Host "   Press [Enter] in this window to stop all services." -ForegroundColor Yellow
Write-Host "  =================================================================" -ForegroundColor Green
Write-Host ""

if (-not $NoBrowser) {
    Start-Process "http://localhost:$UiPort"
}

# ──────────────────────────────────────────────────────────────────────────────
# Wait for user — then graceful shutdown
# ──────────────────────────────────────────────────────────────────────────────
try {
    $null = Read-Host "  Running (Press Enter to stop all services)"
} finally {
    Write-Host ""
    Write-Host "  Shutting down all Project KB services..." -ForegroundColor Cyan

    if ($uiProcess -and -not $uiProcess.HasExited) {
        Write-Host "    Stopping Streamlit UI  (PID $($uiProcess.Id))..." -ForegroundColor DarkGray
        Stop-Process -Id $uiProcess.Id -Force -ErrorAction SilentlyContinue
    }
    if ($apiProcess -and -not $apiProcess.HasExited) {
        Write-Host "    Stopping FastAPI Backend (PID $($apiProcess.Id))..." -ForegroundColor DarkGray
        Stop-Process -Id $apiProcess.Id -Force -ErrorAction SilentlyContinue
    }
    if ($ollamaProcess -and -not $ollamaProcess.HasExited) {
        Write-Host "    Stopping Ollama         (PID $($ollamaProcess.Id))..." -ForegroundColor DarkGray
        Stop-Process -Id $ollamaProcess.Id -Force -ErrorAction SilentlyContinue
    }

    Write-Host ""
    Write-Host "  ✅ All services stopped cleanly." -ForegroundColor Green
    Write-Host ""
}
