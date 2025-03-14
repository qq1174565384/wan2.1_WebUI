@echo off
REM 设置UTF-8编码
chcp 65001 >nul

echo 当前目录: %cd%


REM 切换到项目目录
cd /d ".\apps\gradio\Wan" || (
    echo [错误] 无法切换到项目目录
    pause
    exit /b 1
)


REM 激活虚拟环境
if not exist ".wan2.1env\Scripts\activate" (
    echo [错误] 虚拟环境激活脚本不存在
    pause
    exit /b 1
)
call ".wan2.1env\Scripts\activate" || (
    echo [错误] 虚拟环境激活失败
    pause
    exit /b 1
)

REM 检查虚拟环境
echo %PATH% | findstr /i ".wan2.1env" >nul || (
    echo [警告] 虚拟环境可能未正确激活
    echo 建议检查系统环境变量中的Python路径配置
)
echo 虚拟环境已成功激活


REM 检查并运行主脚本
if not exist "Wan2.1_WebUI.py" (
    echo [错误] 主脚本不存在
    pause
    exit /b 1
)


REM 启动服务
echo 正在启动服务...
python Wan2.1_WebUI.py 

pause


