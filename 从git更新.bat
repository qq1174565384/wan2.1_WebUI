REM 设置UTF-8编码。
chcp 65001 >nul
git fetch --all  
git reset --hard origin/main
git pull https://gitee.com/bldm/wan2.1_WebUI.git

pause    