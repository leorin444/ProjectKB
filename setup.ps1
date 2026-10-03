<#
.SYNOPSIS
  One-shot setup for the local multimodal knowledge base (Windows, native).

.DESCRIPTION
  Installs/checks: Python 3.11, Git, ffmpeg, Ollama (via winget), creates a venv,
  installs Python packages, creates the folder structure and .env, pre-downloads
  the embedding + Whisper models, and pulls the Ollama models.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\setup.ps1

.EXAMPLE
  .\setup.ps1 -ProjectPath D:\kb -DataPath E:\kb-data -SkipOllamaModels
#>

[CmdletBinding()]
param(
    [string]$ProjectPath   = $PSScriptRoot,
    [string]$DataPath      = "",
    [string]$LlmModel      = "llama3.1:8b",
    [string]$VisionModel   = "qwen2.5vl:7b",
    [string]$WhisperModel  = "small",
    [string]$EmbedModel    = "sentence-transformers/all-MiniLM-L6-v2",
    [switch]$SkipOllamaModels,
    [switch]$SkipModelDownloads,
    [switch]$InstallTesseract
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------- helpers ----------
function Write-Step($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "    [ok] $msg" -ForegroundColor Green }
function Write-Warn2($msg){ Write-Host "    [!!] $msg" -ForegroundColor Yellow }

function Test-Cmd($name) { return [bool](Get-Command $name -ErrorAction SilentlyContinue) }

function Update-SessionPath {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user    = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user"
}

function Install-WingetPackage($id, $label) {
    Write-Host "    Installing $label via winget ($id)..."
    winget install --id $id -e --silent --accept-package-agreements --accept-source-agreements
    # winget returns non-zero when already installed/up to date; verify afterwards instead
    Update-SessionPath
}

function Get-PythonCommand {
    # Prefer the py launcher with 3.11, then python on PATH if version is 3.10-3.12
    if (Test-Cmd "py") {
        & py -3.11 --version 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { return @("py", "-3.11") }
    }
    if (Test-Cmd "python") {
        $v = & python -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $v -match "^3\.(10|11|12)$") { return ,@("python") }
    }
    return $null
}

function Invoke-Checked {
    param([string]$What, [scriptblock]$Block)
    & $Block
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit code $LASTEXITCODE)." }
}

# ---------- preflight ----------
Write-Step "Preflight checks"
if (-not (Test-Cmd "winget")) {
    throw "winget was not found. Install 'App Installer' from the Microsoft Store, then re-run this script."
}
Write-Ok "winget found"

$free = (Get-PSDrive -Name ((Resolve-Path $PWD).Drive.Name)).Free / 1GB
Write-Host ("    Free space on current drive: {0:N1} GB (models + video data need a lot; 30+ GB recommended)" -f $free)

# ---------- system tools ----------
Write-Step "System tools"

$pyCmd = Get-PythonCommand
if (-not $pyCmd) {
    Install-WingetPackage "Python.Python.3.11" "Python 3.11"
    $pyCmd = Get-PythonCommand
    if (-not $pyCmd) { throw "Python 3.11 not detected after install. Open a NEW PowerShell window and re-run." }
}
$pyExe = $pyCmd[0]
$pyPre = @()
if ($pyCmd.Count -gt 1) { $pyPre = $pyCmd[1..($pyCmd.Count - 1)] }
Write-Ok ("Python: " + (& $pyExe @pyPre --version 2>&1))

if (-not (Test-Cmd "git")) { Install-WingetPackage "Git.Git" "Git" }
if (Test-Cmd "git") { Write-Ok (git --version) } else { Write-Warn2 "Git not detected yet (open a new terminal if needed)" }

if (-not (Test-Cmd "ffmpeg")) { Install-WingetPackage "Gyan.FFmpeg" "ffmpeg" }
if (Test-Cmd "ffmpeg") { Write-Ok "ffmpeg found" } else { Write-Warn2 "ffmpeg not on PATH yet - open a NEW terminal after setup finishes" }

if (-not (Test-Cmd "ollama")) { Install-WingetPackage "Ollama.Ollama" "Ollama" }
if (Test-Cmd "ollama") { Write-Ok "Ollama found" } else { Write-Warn2 "Ollama not on PATH yet - open a NEW terminal after setup finishes" }

if ($InstallTesseract) {
    Install-WingetPackage "UB-Mannheim.TesseractOCR" "Tesseract OCR (optional)"
}

# ---------- project folders ----------
Write-Step "Project folders"
$ProjectPath = [System.IO.Path]::GetFullPath($ProjectPath)
if ([string]::IsNullOrWhiteSpace($DataPath)) { $DataPath = Join-Path $ProjectPath "data" }
$DataPath = [System.IO.Path]::GetFullPath($DataPath)

$dirs = @(
    "app\api", "app\ingestion", "app\retrieval", "app\llm", "app\storage", "app\jobs",
    "ui", "tests", "docs"
) | ForEach-Object { Join-Path $ProjectPath $_ }

$dataDirs = @("originals", "markdown", "frames", "index") | ForEach-Object { Join-Path $DataPath $_ }

foreach ($d in ($dirs + $dataDirs)) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
}
Write-Ok "Project: $ProjectPath"
Write-Ok "Data:    $DataPath"

# ---------- virtual environment ----------
Write-Step "Python virtual environment"
$venvPath = Join-Path $ProjectPath ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    $pyArgs = @()
    if ($pyCmd.Count -gt 1) { $pyArgs += $pyCmd[1..($pyCmd.Count - 1)] }
    $pyArgs += @("-m", "venv", $venvPath)
    & $pyCmd[0] @pyArgs
    if ($LASTEXITCODE -ne 0) { throw "Could not create virtual environment." }
}
Write-Ok "venv at $venvPath"

Invoke-Checked "pip upgrade" { & $venvPython -m pip install --upgrade pip setuptools wheel --quiet }

# ---------- requirements ----------
Write-Step "Python packages"
$requirements = @'
# API
fastapi>=0.115
uvicorn[standard]>=0.30
python-multipart>=0.0.9
pydantic>=2.7
pydantic-settings>=2.3
python-dotenv>=1.0
httpx>=0.27

# UI
streamlit>=1.37

# LLM / embeddings / retrieval
ollama>=0.3
sentence-transformers>=3.0
qdrant-client>=1.10

# Media
faster-whisper>=1.0
opencv-python-headless>=4.9
imagehash>=4.3
Pillow>=10.3
PyMuPDF>=1.24
pytesseract>=0.3.10

# Dev
pytest>=8.2
ruff>=0.5
'@
$reqFile = Join-Path $ProjectPath "requirements.txt"
if (-not (Test-Path $reqFile)) {
    Set-Content -Path $reqFile -Value $requirements -Encoding UTF8
    Write-Ok "requirements.txt created"
} else {
    Write-Warn2 "requirements.txt already exists - using it as is (repo owns this file)"
}

Write-Host "    Installing packages (first run can take several minutes)..."
Invoke-Checked "pip install" { & $venvPython -m pip install -r $reqFile }
Write-Ok "Packages installed"

# ---------- .env ----------
Write-Step "Configuration (.env)"
$envFile    = Join-Path $ProjectPath ".env"
$envExample = Join-Path $ProjectPath ".env.example"

$apiKey = -join ((48..57) + (97..122) | Get-Random -Count 32 | ForEach-Object { [char]$_ })
$dataPosix = $DataPath.Replace("\", "/")

$envText = @"
KB_DATA_DIR=$dataPosix
KB_ORIGINALS_DIR=
KB_FRAMES_DIR=
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=$LlmModel
VISION_MODEL=$VisionModel
EMBED_MODEL=$EmbedModel
WHISPER_MODEL=$WhisperModel
API_KEY=$apiKey
API_HOST=127.0.0.1
API_PORT=8000
MAX_UPLOAD_MB=2048
CHUNK_TOKENS=500
CHUNK_OVERLAP=50
MIN_RELEVANCE=0.25
USE_RERANKER=false
MAX_FRAMES_PER_VIDEO=300
FRAME_DEDUP_THRESHOLD=6
SECTION_SECONDS=120
"@

if (-not (Test-Path $envFile)) {
    Set-Content -Path $envFile -Value $envText -Encoding UTF8
    Write-Ok ".env created (random API_KEY generated)"
} else {
    Write-Warn2 ".env already exists - left untouched"
}
$exampleText = ($envText -replace "API_KEY=.*", "API_KEY=change-me")
Set-Content -Path $envExample -Value $exampleText -Encoding UTF8

$gitignore = Join-Path $ProjectPath ".gitignore"
if (-not (Test-Path $gitignore)) {
    Set-Content -Path $gitignore -Encoding UTF8 -Value @"
.venv/
__pycache__/
*.pyc
.env
data/
.pytest_cache/
.ruff_cache/
"@
}

# ---------- Ollama ----------
Write-Step "Ollama"
function Test-OllamaUp {
    try { Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 3 | Out-Null; return $true }
    catch { return $false }
}

if (Test-Cmd "ollama") {
    if (-not (Test-OllamaUp)) {
        Write-Host "    Starting Ollama server..."
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
        for ($i = 0; $i -lt 30 -and -not (Test-OllamaUp); $i++) { Start-Sleep -Seconds 1 }
    }
    if (Test-OllamaUp) {
        Write-Ok "Ollama server is running"
        if (-not $SkipOllamaModels) {
            foreach ($m in @($LlmModel, $VisionModel)) {
                Write-Host "    Pulling $m (large download, please wait)..."
                & ollama pull $m
                if ($LASTEXITCODE -ne 0) { Write-Warn2 "Could not pull $m - run 'ollama pull $m' manually later" }
                else { Write-Ok "$m ready" }
            }
        } else {
            Write-Warn2 "Skipping Ollama model pulls (-SkipOllamaModels)"
        }
    } else {
        Write-Warn2 "Ollama server did not respond. Start it (Ollama app or 'ollama serve') and run: ollama pull $LlmModel ; ollama pull $VisionModel"
    }
} else {
    Write-Warn2 "Ollama command not available in this session. Open a NEW terminal and run: ollama pull $LlmModel ; ollama pull $VisionModel"
}

# ---------- pre-download local models ----------
if (-not $SkipModelDownloads) {
    Write-Step "Pre-downloading embedding and Whisper models"
    $pyScript = @"
from sentence_transformers import SentenceTransformer
from faster_whisper import WhisperModel
print('Downloading embedding model: $EmbedModel')
SentenceTransformer('$EmbedModel')
print('Downloading Whisper model: $WhisperModel')
WhisperModel('$WhisperModel', device='cpu', compute_type='int8')
print('done')
"@
    $tmpPy = Join-Path $env:TEMP "kb_predownload.py"
    Set-Content -Path $tmpPy -Value $pyScript -Encoding UTF8
    & $venvPython $tmpPy
    if ($LASTEXITCODE -ne 0) { Write-Warn2 "Model pre-download failed (they will download on first use instead)" }
    else { Write-Ok "Local models cached" }
    Remove-Item $tmpPy -Force -ErrorAction SilentlyContinue
}

# ---------- save prompt reminder ----------
$promptSrc = Join-Path $PSScriptRoot "ANTIGRAVITY_PROMPT.md"
if (Test-Path $promptSrc) {
    Copy-Item $promptSrc (Join-Path $ProjectPath "docs\PROJECT_BRIEF.md") -Force
    Write-Ok "Copied ANTIGRAVITY_PROMPT.md to docs\PROJECT_BRIEF.md"
}

# ---------- summary ----------
Write-Step "Summary"
$summary = [ordered]@{
    "Python (venv)" = (& $venvPython --version 2>&1)
    "ffmpeg"        = $(if (Test-Cmd "ffmpeg") { "found" } else { "NOT on PATH (open a new terminal)" })
    "Ollama"        = $(if (Test-OllamaUp) { "running" } else { "not running" })
    "Project"       = $ProjectPath
    "Data"          = $DataPath
}
$summary.GetEnumerator() | ForEach-Object { Write-Host ("    {0,-14} {1}" -f $_.Key, $_.Value) }

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Next steps:"
Write-Host "  1. Open this folder in Antigravity:  $ProjectPath"
Write-Host "  2. Paste the contents of ANTIGRAVITY_PROMPT.md as your first message."
Write-Host "  3. Activate the environment in a terminal:  $venvPath\Scripts\Activate.ps1"
Write-Host "  If ffmpeg/ollama were reported missing, close this window and open a new one first."
