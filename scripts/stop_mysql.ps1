# 停止本地免安装版 MySQL
$root = Split-Path -Parent $PSScriptRoot
$mysqladmin = Join-Path $root "mysql\bin\mysqladmin.exe"

if (-not (Get-NetTCPConnection -LocalPort 3306 -State Listen -ErrorAction SilentlyContinue)) {
    Write-Host "MySQL 未在运行。" -ForegroundColor Yellow
    exit 0
}
& $mysqladmin -u root shutdown
Start-Sleep -Seconds 2
if (Get-NetTCPConnection -LocalPort 3306 -State Listen -ErrorAction SilentlyContinue) {
    Write-Host "MySQL 停止失败。" -ForegroundColor Red
    exit 1
}
Write-Host "MySQL 已停止。" -ForegroundColor Green
