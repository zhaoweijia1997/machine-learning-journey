@echo off
REM 紧急关闭脚本 - 强制结束所有 Python 进程

echo ================================================
echo 紧急关闭 - 强制结束 Python 进程
echo ================================================
echo.

echo 正在查找 Python 进程...
tasklist | findstr python.exe

echo.
echo 正在强制关闭...
taskkill //F //IM python.exe 2>nul

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ 已成功关闭所有 Python 进程
) else (
    echo.
    echo ⚠️ 未找到正在运行的 Python 进程
)

echo.
echo 完成！
pause
