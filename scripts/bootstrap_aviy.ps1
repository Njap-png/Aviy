param(
    [switch]$ResumeExisting
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

if ((Test-Path $Root) -and !$ResumeExisting) {
    throw "Project directory already exists. Use -ResumeExisting to verify and complete missing structure."
}

New-Item -ItemType Directory -Force -Path $Root | Out-Null
Set-Location $Root

$Directories = @(
    "configs/model", "configs/training", "configs/tokenizer",
    "configs/agents", "configs/memory", "configs/autonomy",
    "configs/security", "configs/system",
    "aviy/core", "aviy/models/architectures",
    "aviy/tokenizer", "aviy/data", "aviy/learning",
    "aviy/training", "aviy/cognition", "aviy/memory",
    "aviy/knowledge", "aviy/agents", "aviy/tools",
    "aviy/multimodal", "aviy/autonomous", "aviy/evaluation",
    "aviy/infrastructure", "aviy/security", "aviy/api", "aviy/cli",
    "scripts", "notebooks", "experiments", "datasets",
    "outputs", "checkpoints", "reports", "docs", "tests"
)

foreach ($Directory in $Directories) {
    New-Item -ItemType Directory -Force -Path $Directory | Out-Null
}

$Packages = @(
    "aviy", "aviy/core", "aviy/models",
    "aviy/models/architectures", "aviy/tokenizer",
    "aviy/data", "aviy/learning", "aviy/training",
    "aviy/cognition", "aviy/memory", "aviy/knowledge",
    "aviy/agents", "aviy/tools", "aviy/multimodal",
    "aviy/autonomous", "aviy/evaluation",
    "aviy/infrastructure", "aviy/security", "aviy/api", "aviy/cli"
)

$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

foreach ($Package in $Packages) {
    $InitFile = Join-Path $Package "__init__.py"
    if (!(Test-Path $InitFile)) {
        [System.IO.File]::WriteAllText(
            (Join-Path $Root $InitFile),
            '"""Aviy package."""' + [Environment]::NewLine,
            $Utf8NoBom
        )
    }
}

if (!(Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
}

$Python = ".\.venv\Scripts\python.exe"
& $Python --version
if ($LASTEXITCODE -ne 0) {
    throw "Python environment verification failed."
}

Write-Host "Aviy structure verification completed." -ForegroundColor Green
Write-Host "No existing files were overwritten by the bootstrap."
