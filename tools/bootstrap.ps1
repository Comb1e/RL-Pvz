param(
    [string]$GameRepo = 'E:\Projects\pvz',
    [switch]$Cuda
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$gameRoot = (Resolve-Path -LiteralPath $GameRepo).Path
$expectedCommit = 'b3cfbd886ab378313a1fdb57ee43a9a1b36a0793'
$actualCommit = & git -C $gameRoot rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $actualCommit.Trim() -ne $expectedCommit) {
    throw "Game checkout must be at $expectedCommit. The installer does not change it."
}
$gameChanges = & git -C $gameRoot status --porcelain --untracked-files=no
if ($LASTEXITCODE -ne 0 -or $gameChanges) { throw 'Game has tracked changes; use a clean pinned checkout.' }

function Invoke-Checked {
    param([string]$Program, [string[]]$CommandArgs)
    & $Program @CommandArgs
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}

$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    Invoke-Checked -Program 'py' -CommandArgs @('-3.12', '-m', 'venv', (Join-Path $projectRoot '.venv'))
}
$torchIndex = if ($Cuda) { 'https://download.pytorch.org/whl/cu128' } else { 'https://download.pytorch.org/whl/cpu' }
$torchVersion = if ($Cuda) { 'torch==2.8.0+cu128' } else { 'torch==2.8.0+cpu' }
Invoke-Checked -Program $venvPython -CommandArgs @('-m', 'pip', 'install', '--no-cache-dir', $torchVersion, '--index-url', $torchIndex)
Invoke-Checked -Program $venvPython -CommandArgs @('-m', 'pip', 'install', '--no-cache-dir', '-r', (Join-Path $projectRoot 'requirements-lock.txt'))
Invoke-Checked -Program $venvPython -CommandArgs @('-m', 'pip', 'install', '--no-deps', $gameRoot, '-e', "${projectRoot}[dev,ui]")
Invoke-Checked -Program $venvPython -CommandArgs @('-m', 'pip', 'check')
Invoke-Checked -Program $venvPython -CommandArgs @('-m', 'pvz_rl', 'doctor')
