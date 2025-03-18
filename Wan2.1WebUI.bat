::[Bat To Exe Converter]
::
::YAwzoRdxOk+EWAjk
::fBw5plQjdCaDJG6F+hB+eSddQQWFOVezBboSpuH44Io=
::YAwzuBVtJxjWCl3EqQJgSA==
::ZR4luwNxJguZRRnk
::Yhs/ulQjdF+5
::cxAkpRVqdFKZSjk=
::cBs/ulQjdF+5
::ZR41oxFsdFKZSDk=
::eBoioBt6dFKZSDk=
::cRo6pxp7LAbNWATEpCI=
::egkzugNsPRvcWATEpCI=
::dAsiuh18IRvcCxnZtBJQ
::cRYluBh/LU+EWAnk
::YxY4rhs+aU+IeA==
::cxY6rQJ7JhzQF1fEqQK3wfPSsMgH07zgRuVSuKaqjw==
::ZQ05rAF9IBncCkqN+0xwdVvTpewEwKc/6WJGqLi1v6TWwg==
::ZQ05rAF9IAHYFVzEqQIHKRUGAlW1OWmPL9U=
::eg0/rx1wNQPfEVWB+kM9LVsJDDODMjn0V4IZ6t3Sjw==
::fBEirQZwNQPfEVWB+kM9LVsJDDODMjn0V4or7erOxoo=
::cRolqwZ3JBvQF1fEqQJQ
::dhA7uBVwLU+EWDk=
::YQ03rBFzNR3SWATElA==
::dhAmsQZ3MwfNWATElA==
::ZQ0/vhVqMQ3MEVWAtB9wSA==
::Zg8zqx1/OA3MEVWAtB9wSA==
::dhA7pRFwIByZRRkCCI649dTQl9IH2YYy0mx8
::Zh4grVQjdCaDJG6F+hB+eSdjTQrQcjqNA7cpwab+9+/n
::YB416Ek+ZG8=
::
::
::978f952a14a936cc963da21a135fa983
@echo off
REM 设置UTF-8编码
chcp 65001 >nul

echo 当前目录: %cd%



REM 激活虚拟环�?
if not exist ".wan2.1env\Scripts\activate" (
    echo [错误] 虚拟环境激活脚本不存在
    pause
    exit /b 1
)
call ".wan2.1env\Scripts\activate" || (
    echo [错误] 虚拟环境激活失�?
    pause
    exit /b 1
)

REM 检查虚拟环�?
echo %PATH% | findstr /i ".wan2.1env" >nul || (
    echo [警告] 虚拟环境可能未正确激�?
    echo 建议检查系统环境变量中的Python路径配置
)
echo 虚拟环境已成功激�?

REM 切换到项目目�?
cd /d ".\apps\gradio\Wan" || (
    echo [错误] 无法切换到项目目�?
    pause
    exit /b 1
)

REM 检查并运行主脚�?
if not exist "Wan2.1_WebUI.py" (
    echo [错误] 主脚本不存在
    pause
    exit /b 1
)


REM 启动服务
echo 正在启动服务...
python Wan2.1_WebUI.py 

pause


