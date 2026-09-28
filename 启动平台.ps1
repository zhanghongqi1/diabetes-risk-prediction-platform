# ============================================================
# 糖尿病风险预测平台 - 一键启动（MySQL + Flask 网页）
# 用法：右键“使用 PowerShell 运行”，或在项目目录执行
#       powershell -ExecutionPolicy Bypass -File .\启动平台.ps1
# ============================================================
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

Write-Host "==== 第 1 步：启动 MySQL ====" -ForegroundColor Cyan
& (Join-Path $root "scripts\start_mysql.ps1")
if ($LASTEXITCODE -ne 0) { Read-Host "按回车退出"; exit 1 }

Write-Host "`n==== 第 2 步：启动 Web 平台 ====" -ForegroundColor Cyan
$py = Join-Path $root "venv\Scripts\python.exe"
Start-Process -FilePath $py -ArgumentList "web\app.py" -WorkingDirectory $root

Write-Host "等待服务就绪 ..."
for ($i = 0; $i -lt 20; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/health" -TimeoutSec 2
        if ($r.mysql_ok -and $r.model_loaded) { break }
    } catch {}
}

Write-Host "`n平台已启动，正在打开浏览器：http://127.0.0.1:5000" -ForegroundColor Green
Start-Process "http://127.0.0.1:5000"
Write-Host "`n关闭平台：直接关掉弹出的 Python 窗口，并运行 scripts\stop_mysql.ps1 停止 MySQL。"
Read-Host "`n按回车退出本脚本（不会关闭平台）"
