REM 设置UTF-8编码。
chcp 65001 >nul
git reset --hard
git clean -fd
git pull https://github.com/qq1174565384/wan2.1_WebUI.git
if %errorlevel% neq 0 (
    echo 更新失败，请检查网络或仓库地址。
) else (
    echo 更新完成。
)
pause    