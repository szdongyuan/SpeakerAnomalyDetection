$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot ".venv2\Scripts\python.exe"
$specFile = Join-Path $projectRoot "main_window_Launcher.spec"
$distPath = Join-Path $projectRoot "dist_onnx"
$workPath = Join-Path $projectRoot "build_onnx"

if (-not (Test-Path $pythonExe)) {
    throw "Missing virtual environment python: $pythonExe"
}

if (-not (Test-Path $specFile)) {
    throw "Missing spec file: $specFile"
}

& $pythonExe -m PyInstaller --noconfirm --clean --distpath $distPath --workpath $workPath $specFile
