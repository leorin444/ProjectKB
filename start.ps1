<#
.SYNOPSIS
    Starts the Local Multimodal Knowledge Base (FastAPI backend + Streamlit UI).
.DESCRIPTION
    Checks prerequisite models in Ollama, boots the FastAPI backend on port 8000,
    waits for the /health check to respond OK, then launches the Streamlit UI on port 8501
    and automatically opens your browser. Gracefully shuts down both processes on exit.
#>

[CmdletBinding()]
param (
    [string]$ApiHost = "127.0.0.1",
    [int]$ApiPort = 8000,
    [int]$UiPort = 8501,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

# Navigate to project root
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path "$ProjectRoot\app")) {
    $ProjectRoot = $PSScriptRoot
}
Set-Location $ProjectRoot

$VenvPython = "$ProjectRoot\.venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "❌ Error: Virtual environment python not found at $VenvPython" -ForegroundColor Red
    Write-Host "Please run setup.ps1 first." -ForegroundColor Yellow
    exit 1
}

# Visual Banner
Clear-Host
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "       🧠 LOCAL MULTIMODAL KNOWLEDGE BASE (Project KB)          " -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  • Privacy:    100% Offline / Local Execution (Zero Cloud Egress)" -ForegroundColor DarkGray
Write-Host "  • Backend:    FastAPI on http://$ApiHost`:$ApiPort" -ForegroundColor DarkGray
Write-Host "  • Frontend:   Streamlit on http://localhost:$UiPort" -ForegroundColor DarkGray
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check Ollama Liveness
Write-Host "[1/3] Checking Ollama service..." -ForegroundColor Yellow
try {
    $ollamaCheck = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 3 -ErrorAction SilentlyContinue
    $availableModels = ($ollamaCheck.models | ForEach-Object { $_.name }) -join ", "
    Write-Host "  ✅ Ollama is running ($availableModels)" -ForegroundColor Green
} catch {
    Write-Host "  ⚠️ Warning: Ollama could not be reached at http://localhost:11434" -ForegroundColor Yellow
    Write-Host "    Make sure Ollama is started if you want Q&A or image parsing." -ForegroundColor DarkGray
}

# 2. Launch FastAPI Backend
Write-Host "[2/3] Starting API Backend..." -ForegroundColor Yellow
$apiProcess = Start-Process -FilePath $VenvPython `
    -ArgumentList "-m", "uvicorn", "app.main:app", "--host", $ApiHost, "--port", "$ApiPort", "--log-level", "info" `
    -WorkingDirectory $ProjectRoot `
    -PassThru

# Wait for backend health check
$healthUrl = "http://$ApiHost`:$ApiPort/health"
$maxRetries = 20
$healthy = $false
Write-Host -NoNewline "  Waiting for API server to become ready..." -ForegroundColor DarkGray

for ($i = 0; $i -lt $maxRetries; $i++) {
    Start-Sleep -Milliseconds 600
    try {
        $resp = Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($resp -and $resp.status) {
            $healthy = $true
            break
        }
    } catch {
        Write-Host -NoNewline "." -ForegroundColor DarkGray
    }
}

if ($healthy) {
    Write-Host " ✅ Ready!" -ForegroundColor Green
} else {
    Write-Host " ⚠️ Timeout waiting for /health, continuing anyway..." -ForegroundColor Yellow
}

# 3. Launch Streamlit Frontend
Write-Host "[3/3] Starting Streamlit Frontend..." -ForegroundColor Yellow
$uiArgs = @("-m", "streamlit", "run", "$ProjectRoot\ui\streamlit_app.py", "--server.port", "$UiPort")
if ($NoBrowser) {
    $uiArgs += "--server.headless", "true"
}

$uiProcess = Start-Process -FilePath $VenvPython `
    -ArgumentList $uiArgs `
    -WorkingDirectory $ProjectRoot `
    -PassThru

Write-Host ""
Write-Host "=================================================================" -ForegroundColor Green
Write-Host "  🎉 Project KB is running!" -ForegroundColor Green
Write-Host "     • Web Interface: http://localhost:$UiPort" -ForegroundColor White
Write-Host "     • API Docs:      http://$ApiHost`:$ApiPort/docs" -ForegroundColor White
Write-Host "     • Health Status: http://$ApiHost`:$ApiPort/health" -ForegroundColor White
Write-Host "=================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Press [Ctrl+C] or [Enter] in this window anytime to stop both services." -ForegroundColor Yellow
Write-Host ""

# Automatically open browser if not headless
if (-not $NoBrowser) {
    Start-Sleep -Seconds 1
    Start-Process "http://localhost:$UiPort"
}

# Keep running and trap exit
try {
    $null = Read-Host "Running... (Press Enter to stop)"
} finally {
    Write-Host ""
    Write-Host "Shutting down Project KB services..." -ForegroundColor Cyan

    if ($uiProcess -and -not $uiProcess.HasExited) {
        Write-Host "  Stopping UI (PID: $($uiProcess.Id))..." -ForegroundColor DarkGray
        Stop-Process -Id $uiProcess.Id -Force -ErrorAction SilentlyContinue
    }

    if ($apiProcess -and -not $apiProcess.HasExited) {
        Write-Host "  Stopping Backend (PID: $($apiProcess.Id))..." -ForegroundColor DarkGray
        Stop-Process -Id $apiProcess.Id -Force -ErrorAction SilentlyContinue
    }

    Write-Host "✅ All services stopped safely." -ForegroundColor Green
}
