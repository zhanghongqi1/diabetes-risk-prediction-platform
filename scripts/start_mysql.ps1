# 启动本地免安装版 MySQL（糖尿病风险预测平台）
$root = Split-Path -Parent $PSScriptRoot
$mysqld = Join-Path $root "mysql\bin\mysqld.exe"
$ini = Join-Path $root "mysql\my.ini"

if (Get-NetTCPConnection -LocalPort 3306 -State Listen -ErrorAction SilentlyContinue) {
    Write-Host "MySQL 已在运行，端口 3306 正常监听。" -ForegroundColor Green
    exit 0
}
Write-Host "正在启动 MySQL ..."
Start-Process -FilePath $mysqld -ArgumentList "--defaults-file=`"$ini`"" -WindowStyle Hidden
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    if (Get-NetTCPConnection -LocalPort 3306 -State Listen -ErrorAction SilentlyContinue) {
        Write-Host "MySQL 已启动，监听端口 3306。" -ForegroundColor Green
        exit 0
    }
}
Write-Host "MySQL 启动超时，请检查 mysql\data 目录下的错误日志。" -ForegroundColor Red
exit 1
