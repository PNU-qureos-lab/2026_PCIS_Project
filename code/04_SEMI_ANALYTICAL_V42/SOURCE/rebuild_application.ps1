# 개발자용 재빌드. 일반 사용자는 상위 폴더의 수심분석.exe만 실행하면 됩니다.
# Python 3.12(64비트, Tcl/Tk 포함)가 필요합니다. 라이브러리는 별도 가상환경에 설치합니다.
# Example: powershell -ExecutionPolicy Bypass -File .\rebuild_application.ps1 -Python C:\Python312\python.exe
# 최초 의존성 설치에는 인터넷이 필요하며, 완성된 배포본은 인터넷 없이 실행됩니다.
# 재빌드하기 전에 실행 중인 수심분석.exe를 닫으세요. 프로그램 폴더는 새 런타임으로 교체합니다.
param([string]$Python = 'python', [string]$BuildDirectory = '')
$ErrorActionPreference = 'Stop'
$sourceRoot = $PSScriptRoot
$packageRoot = Split-Path -Parent $sourceRoot
# 동봉 라이브러리와 같은 Python 계열·아키텍처로 빌드하여 실행환경 차이를 줄입니다.
& $Python -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3,12) and sys.maxsize > 2**32 else 1)'
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 64비트를 지정하세요.' }
if (-not $BuildDirectory) {
    $BuildDirectory = Join-Path ([System.IO.Path]::GetTempPath()) ('bathymetry-build-' + [Guid]::NewGuid().ToString('N'))
}
$BuildDirectory = [System.IO.Path]::GetFullPath($BuildDirectory)
if ($BuildDirectory -eq $packageRoot -or $BuildDirectory.StartsWith($packageRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw '빌드 작업 폴더는 배포 폴더 밖에 지정하세요.'
}
New-Item -ItemType Directory -Force -Path $BuildDirectory | Out-Null
$venvDirectory = Join-Path $BuildDirectory 'venv'
& $Python -m venv $venvDirectory
if ($LASTEXITCODE -ne 0) { throw '가상환경 생성 실패' }
$buildPython = Join-Path $venvDirectory 'Scripts\python.exe'
& $buildPython -m pip install -r (Join-Path $sourceRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw '의존성 설치 실패' }
# Tk의 스크립트와 DLL을 실제 초기화하여 누락을 빌드 전에 발견합니다.
& $buildPython -c 'import tkinter; t=tkinter.Tk(); t.withdraw(); t.update_idletasks(); t.destroy()'
if ($LASTEXITCODE -ne 0) { throw 'Python의 Tcl/Tk 설치를 확인하세요.' }
$env:PYTHONDONTWRITEBYTECODE = '1'
Push-Location $sourceRoot
try {
    & $buildPython -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw '회귀검증 실패. 배포를 중단합니다.' }
    & $buildPython -m PyInstaller --noconfirm --clean --windowed --onedir --name 수심분석 --contents-directory 프로그램 --collect-all rasterio --distpath (Join-Path $BuildDirectory 'dist') --workpath (Join-Path $BuildDirectory 'work') --specpath $BuildDirectory (Join-Path $sourceRoot 'main.py')
    if ($LASTEXITCODE -ne 0) { throw '실행파일 빌드 실패' }
    # 계산 결과/자료/소스는 보존하며 실행파일과 런타임 파일만 갱신합니다.
    $builtPackage = Join-Path $BuildDirectory 'dist\수심분석'
    Copy-Item -LiteralPath (Join-Path $builtPackage '수심분석.exe') -Destination $packageRoot -Force
    # 이전 빌드에만 있던 DLL이 남지 않도록 동봉 런타임 폴더만 정확히 교체합니다.
    $runtimeTarget = [System.IO.Path]::GetFullPath((Join-Path $packageRoot '프로그램'))
    if ((Split-Path -Parent $runtimeTarget) -ne $packageRoot -or (Split-Path -Leaf $runtimeTarget) -ne '프로그램') {
        throw '런타임 교체 경로가 배포 폴더의 프로그램 하위 폴더가 아닙니다.'
    }
    if (Test-Path -LiteralPath $runtimeTarget) {
        if ((Get-Item -LiteralPath $runtimeTarget).Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
            throw '프로그램 폴더가 링크이면 자동으로 교체하지 않습니다.'
        }
        Remove-Item -LiteralPath $runtimeTarget -Recurse -Force
    }
    Copy-Item -LiteralPath (Join-Path $builtPackage '프로그램') -Destination $packageRoot -Recurse -Force
    # 재빌드한 동봉 런타임과 사용자 화면이 정상 초기화되는지 확인합니다.
    $checkPath = Join-Path $BuildDirectory 'runtime_check.json'
    $checkProcess = Start-Process -FilePath (Join-Path $packageRoot '수심분석.exe') -ArgumentList ('--self-check "' + $checkPath + '"') -WindowStyle Hidden -Wait -PassThru
    if ($checkProcess.ExitCode -ne 0) { throw '재빌드한 실행파일의 런타임 검사에 실패했습니다.' }
    $checkResult = Get-Content -LiteralPath $checkPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not $checkResult.passed) { throw '재빌드한 실행파일의 런타임 검사에 실패했습니다.' }
    # 재빌드하면 실행파일 해시도 바뀝니다. 사용자 분석 결과는 배포 목록에서 제외합니다.
    $manifestCode = @'
from pathlib import Path
import hashlib, json, sys
root = Path(sys.argv[1])
records = []
for path in sorted(root.rglob('*')):
    relative = path.relative_to(root)
    if (not path.is_file() or relative.parts[0] == '결과'
            or '__pycache__' in relative.parts or path.suffix in ('.pyc', '.pyo')
            or relative.as_posix() == '코드/배포파일목록.json'):
        continue
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    records.append({'path': relative.as_posix(), 'bytes': path.stat().st_size, 'sha256': digest})
(root / '코드/배포파일목록.json').write_text(
    json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
'@
    $manifestScript = Join-Path $BuildDirectory 'update_manifest.py'
    Set-Content -LiteralPath $manifestScript -Value $manifestCode -Encoding UTF8
    & $buildPython $manifestScript $packageRoot
    if ($LASTEXITCODE -ne 0) { throw '배포 파일 목록 갱신에 실패했습니다.' }
    Write-Host ('완료: ' + (Join-Path $packageRoot '수심분석.exe'))
} finally {
    Pop-Location
}
