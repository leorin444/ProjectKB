<#
.SYNOPSIS
  Pull Ollama models sized for a low-RAM laptop and point .env at them.

.DESCRIPTION
  Profiles:
    Recommended  llama3.2:3b + qwen2.5vl:3b + whisper base   (~5.5 GB download, 8 GB RAM)
    Light        llama3.2:1b + moondream      + whisper tiny  (~3.5 GB download, tight RAM)
    Default      llama3.1:8b + qwen2.5vl:7b   + whisper small (needs 16 GB RAM)

.EXAMPLE
  .\scripts\pull_models.ps1
  .\scripts\pull_models.ps1 -Profile Light
  .\scripts\pull_models.ps1 -SkipEnvUpdate
#>
[CmdletBinding()]
param(
    [ValidateSet("Recommended", "Light", "Default")]
    [string]$Profile = "Recommended",
    [switch]$SkipEnvUpdate,
    [switch]$SetOllamaEnv
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

$profiles = @{
    Recommended = @{ Llm = "llama3.2:3b"; Vision = "qwen2.5vl:3b"; Whisper = "base" }
    Light       = @{ Llm = "llama3.2:1b"; Vision = "moondream";     Whisper = "tiny" }
    Default     = @{ Llm = "llama3.1:8b"; Vision = "qwen2.5vl:7b";  Whisper = "small" }
}
$sel = $profiles[$Profile]
Write-Host "Profile: $Profile -> LLM=$($sel.Llm) VISION=$($sel.Vision) WHISPER=$($sel.Whisper)" -ForegroundColor Cyan

# --- Ollama present / running ---
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user"
}
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "ollama not found. Install it (winget install Ollama.Ollama) and open a NEW terminal."
}

function Test-OllamaUp {
    try { Invoke-RestMethod "http://localhost:11434/api/tags" -TimeoutSec 3 | Out-Null; $true }
    catch { $false }
}

if ($SetOllamaEnv) {
    # Keep at most one model in RAM; free it quickly after use. Restart Ollama to apply.
    [Environment]::SetEnvironmentVariable("OLLAMA_MAX_LOADED_MODELS", "1", "User")
    [Environment]::SetEnvironmentVariable("OLLAMA_KEEP_ALIVE", "2m", "User")
    [Environment]::SetEnvironmentVariable("OLLAMA_NUM_PARALLEL", "1", "User")
    Write-Host "Set OLLAMA_MAX_LOADED_MODELS=1, OLLAMA_KEEP_ALIVE=2m, OLLAMA_NUM_PARALLEL=1 (quit and restart Ollama to apply)." -ForegroundColor Yellow
}

if (-not (Test-OllamaUp)) {
    Write-Host "Starting Ollama server..."
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    for ($i = 0; $i -lt 30 -and -not (Test-OllamaUp); $i++) { Start-Sleep -Seconds 1 }
}
if (-not (Test-OllamaUp)) { throw "Ollama server is not responding at http://localhost:11434." }

# --- pull ---
$failed = @()
foreach ($m in @($sel.Llm, $sel.Vision)) {
    Write-Host "Pulling $m ..." -ForegroundColor Cyan
    & ollama pull $m
    if ($LASTEXITCODE -ne 0) { $failed += $m; Write-Warning "Failed to pull $m" }
}

# --- update .env (UTF-8 without BOM so the BOM can't corrupt the first key) ---
if (-not $SkipEnvUpdate) {
    $envFile = Join-Path $root ".env"
    if (Test-Path $envFile) {
        $text = [IO.File]::ReadAllText($envFile).TrimStart([char]0xFEFF)
    } else {
        $text = ""
    }
    $values = [ordered]@{ LLM_MODEL = $sel.Llm; VISION_MODEL = $sel.Vision; WHISPER_MODEL = $sel.Whisper }
    foreach ($k in $values.Keys) {
        $line = "$k=$($values[$k])"
        if ($text -match "(?m)^$k=.*$") { $text = [regex]::Replace($text, "(?m)^$k=.*$", $line) }
        else { $text = $text.TrimEnd() + "`r`n$line`r`n" }
    }
    [IO.File]::WriteAllText($envFile, $text, (New-Object Text.UTF8Encoding($false)))
    Write-Host ".env updated: LLM_MODEL, VISION_MODEL, WHISPER_MODEL" -ForegroundColor Green
}

Write-Host ""
& ollama list
if ($failed.Count -gt 0) {
    Write-Host "Some pulls failed: $($failed -join ', '). Re-run this script or 'ollama pull <name>'." -ForegroundColor Red
    exit 1
}
Write-Host "Done. Restart the API, then check http://127.0.0.1:8000/health" -ForegroundColor Green
