@echo off
REM 设置UTF-8编码
chcp 65001 >nul

echo 当前目录: %cd%

echo 正在创建虚拟环境


%USERPROFILE%\AppData\Local\Programs\Python\Python310\python.exe -m venv .wan2.1env

echo 虚拟环境创建成功



REM 启用变量延迟扩展
setlocal enabledelayedexpansion

REM 激活虚拟环境
if not exist ".wan2.1env\Scripts\activate" (
    echo [错误] 虚拟环境激活脚本不存在
    pause
    exit /b 1
)
echo 正在尝试激活虚拟环境...
call ".wan2.1env\Scripts\activate"
if %errorlevel% neq 0 (
    echo [错误] 虚拟环境激活失败，错误码: %errorlevel%
    pause
    exit /b 1
)
echo 虚拟环境已成功激活

REM 检查虚拟环境
echo %PATH% | findstr /i ".wan2.1env" >nul
if %errorlevel% neq 0 (
    echo [警告] 虚拟环境可能未正确激活
    echo 建议检查系统环境变量中的Python路径配置
)

REM 先更新pip
echo 检查更新pip...
python -m pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/
if %errorlevel% neq 0 (
    echo [错误] pip更新失败，错误码: %errorlevel%
    pause
    exit /b 1
)

pip cache purge

echo 开始安装依赖...
pip install . -i https://mirrors.aliyun.com/pypi/simple/
if %errorlevel% neq 0 (
    echo [错误] 当前目录下的Python项目安装失败，错误码: %errorlevel%
    pause
    exit /b 1
)

pip install gradio -i https://mirrors.aliyun.com/pypi/simple/
pip install packaging -i https://mirrors.aliyun.com/pypi/simple/
pip install ninja -i https://mirrors.aliyun.com/pypi/simple/

REM 注意：torch、torchvision 和 torchaudio 使用的是 PyTorch 官方源，这里不替换
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

REM 确保安装 wheel 模块
echo 正在检查并安装 wheel 模块...
pip install wheel -i https://mirrors.aliyun.com/pypi/simple/
if %errorlevel% neq 0 (
    echo [错误] wheel 模块安装失败，错误码: %errorlevel%
    pause
    exit /b 1
)


REM 询问用户是否安装 flash-attn
set /p install_flash_attn=是否要安装 flash-attn？（大幅减少显存使用，但编译时间可能非常漫长）(Y/N): 
if /i "%install_flash_attn%"=="Y" (
    REM 安装 flash-attn
    echo 开始安装 flash-attn...（编译可能需要至少一小时，根据硬件配置而定）
    pip install flash-attn --no-build-isolation
    if %errorlevel% neq 0 (
        echo [错误] flash-attn 安装失败，错误码: %errorlevel%
        pause
        exit /b 1
    )
) else (
    if /i "%install_flash_attn%"=="N" (
        echo 跳过安装 flash-attn。
    ) else (
        echo 输入无效，请输入 Y 或 N。
        pause
        exit /b 1
    )
)

pip install modelscope[framework] -i https://mirrors.aliyun.com/pypi/simple/
pip install open_clip_torch -i https://mirrors.aliyun.com/pypi/simple/
echo 所有依赖安装成功！
pause

@REM echo 依赖安装成功，2s后退出


@REM ping -n 3 127.0.0.1 >nul 2>&1

@REM REM 切换到项目目录
@REM cd /d ".\apps\gradio\Wan" || (
@REM     echo [错误] 无法切换到项目目录
@REM     pause
@REM     exit /b 1
@REM )
