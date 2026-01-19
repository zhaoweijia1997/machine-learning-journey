@echo off
REM Git 提交前清理脚本（Windows 版本）
REM 用法: utils\git-clean.bat

echo 🧹 开始清理不必要的文件...
echo.

REM 清理 Python 缓存
echo 清理 Python 缓存...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
del /s /q *.pyc 2>nul
del /s /q *.pyo 2>nul

REM 清理 Jupyter checkpoints
echo 清理 Jupyter checkpoints...
for /d /r . %%d in (.ipynb_checkpoints) do @if exist "%%d" rd /s /q "%%d" 2>nul

REM 清理临时文件
echo 清理临时文件...
del /s /q *.tmp 2>nul
del /s /q *.bak 2>nul

REM 清理系统文件
echo 清理系统文件...
del /s /q Thumbs.db 2>nul
del /s /q desktop.ini 2>nul

echo.
echo ✅ 清理完成！当前 Git 状态：
echo.
git status

echo.
echo 💡 提示：如果要提交，运行：
echo    git add .
echo    git commit -m "你的提交信息"
echo    git push
