@echo off
REM 设置UTF-8编码
chcp 65001 >nul


echo 当前目录: %cd%

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

REM 先更新pip
echo 检查更新pip...
python -m pip install --upgrade pip || (
    echo [错误] pip更新失败
    pause
    exit /b 1
)

REM 安装依赖
echo 检查依赖是否正确安装...
for %%i in (
    "gradio"
    "importlib_metadata"
    "opencv-python"
    "scikit-image"
    "scipy"
    "timm"
    "torch==2.6.0"
    "regex!=2019.12.17"
    "tokenizers>=0.20,<0.21"
    "fastrlock>=0.5"
) do (
    echo 正在检查: %%i
    pip show %%i >nul 2>&1
    if %errorlevel% equ 0 (
        echo %%i 已安装
    ) else (
        echo 正在安装: %%i
        call pip install %%i || (
            echo [错误] 依赖安装失败: %%i
            pause
            exit /b 1
        )
    )
)

pip install -r requirements.txt

pip install gradio

REM 切换到项目目录
cd /d ".\apps\gradio\Wan" || (
    echo [错误] 无法切换到项目目录
    pause
    exit /b 1
)
