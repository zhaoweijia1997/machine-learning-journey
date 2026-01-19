@echo off
REM 快速提交脚本（Windows 版本）
REM 用法: utils\quick-commit.bat "提交信息"

if "%~1"=="" (
    echo ❌ 错误：请提供提交信息
    echo 用法: utils\quick-commit.bat "你的提交信息"
    exit /b 1
)

set COMMIT_MSG=%~1

echo 🧹 步骤 1/4: 清理临时文件...
call utils\git-clean.bat >nul 2>&1

echo 📝 步骤 2/4: 添加文件到暂存区...
git add .

echo 💾 步骤 3/4: 提交到本地仓库...
git commit -m "%COMMIT_MSG%"

if %ERRORLEVEL% EQU 0 (
    echo 🚀 步骤 4/4: 推送到 GitHub...
    git push

    if %ERRORLEVEL% EQU 0 (
        echo.
        echo ✅ 全部完成！
        echo 📊 查看仓库: https://github.com/zhaoweijia1997/machine-learning-journey
    ) else (
        echo ❌ 推送失败，请手动运行: git push
    )
) else (
    echo ℹ️ 没有需要提交的更改
)
