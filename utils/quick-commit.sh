#!/bin/bash
# 快速提交脚本（自动清理 + 提交 + 推送）
# 用法: bash utils/quick-commit.sh "提交信息"

if [ -z "$1" ]; then
    echo "❌ 错误：请提供提交信息"
    echo "用法: bash utils/quick-commit.sh \"你的提交信息\""
    exit 1
fi

COMMIT_MSG="$1"

echo "🧹 步骤 1/4: 清理临时文件..."
bash utils/git-clean.sh > /dev/null 2>&1

echo "📝 步骤 2/4: 添加文件到暂存区..."
git add .

echo "💾 步骤 3/4: 提交到本地仓库..."
git commit -m "$COMMIT_MSG"

if [ $? -eq 0 ]; then
    echo "🚀 步骤 4/4: 推送到 GitHub..."
    git push

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ 全部完成！"
        echo "📊 查看仓库: https://github.com/zhaoweijia1997/machine-learning-journey"
    else
        echo "❌ 推送失败，请手动运行: git push"
    fi
else
    echo "ℹ️ 没有需要提交的更改"
fi
