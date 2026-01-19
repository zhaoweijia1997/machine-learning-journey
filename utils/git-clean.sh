#!/bin/bash
# Git 提交前清理脚本
# 用法: bash utils/git-clean.sh

echo "🧹 开始清理不必要的文件..."

# 删除 Python 缓存
echo "清理 Python 缓存..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null

# 删除 Jupyter checkpoints
echo "清理 Jupyter checkpoints..."
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null

# 删除临时文件
echo "清理临时文件..."
find . -type f -name "*.tmp" -delete 2>/dev/null
find . -type f -name "*.bak" -delete 2>/dev/null
find . -type f -name "*~" -delete 2>/dev/null

# 删除系统文件
echo "清理系统文件..."
find . -type f -name ".DS_Store" -delete 2>/dev/null
find . -type f -name "Thumbs.db" -delete 2>/dev/null
find . -type f -name "desktop.ini" -delete 2>/dev/null

# 删除空目录
echo "清理空目录..."
find . -type d -empty -delete 2>/dev/null

# 显示当前 Git 状态
echo ""
echo "✅ 清理完成！当前 Git 状态："
git status

echo ""
echo "💡 提示：如果要提交，运行："
echo "   git add ."
echo "   git commit -m \"你的提交信息\""
echo "   git push"
