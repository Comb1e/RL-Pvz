param(
    [string]$GameRepo = 'E:\Projects\pvz',
    [switch]$Cuda
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$gameRoot = (Resolve-Path -LiteralPath $GameRepo).Path
$enginePin = Get-Content -LiteralPath (Join-Path $projectRoot 'src\pvz_rl\data\engine-lock.json') -Raw | ConvertFrom-Json
$expectedCommit = $enginePin.commit
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
$stageRoot = Join-Path $projectRoot ('build\game-' + [guid]::NewGuid().ToString('N'))
Invoke-Checked -Program $venvPython -CommandArgs @('-B', (Join-Path $projectRoot 'tools\stage_game.py'), '--repo', $gameRoot, '--output', $stageRoot)
Invoke-Checked -Program $venvPython -CommandArgs @('-m', 'pip', 'install', '--no-deps', '--force-reinstall', $stageRoot)
Invoke-Checked -Program $venvPython -CommandArgs @('-m', 'pip', 'install', '--no-deps', '-e', "${projectRoot}[dev,ui]")
Invoke-Checked -Program $venvPython -CommandArgs @('-m', 'pip', 'check')
Invoke-Checked -Program $venvPython -CommandArgs @('-m', 'pvz_rl', 'doctor')
