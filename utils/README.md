# 工具脚本使用指南

这个目录包含了一些实用的脚本，帮助你更高效地管理项目。

## 📜 可用脚本

### 1. Git 清理脚本

**Windows 用户**：
```bash
utils\git-clean.bat
```

**Linux/Mac 用户**：
```bash
bash utils/git-clean.sh
```

**功能**：
- 删除 Python 缓存文件（`__pycache__`, `*.pyc`）
- 删除 Jupyter 检查点（`.ipynb_checkpoints`）
- 删除临时文件（`*.tmp`, `*.bak`, `*~`）
- 删除系统文件（`Thumbs.db`, `.DS_Store`）
- 删除空目录

**使用时机**：
- 提交代码前
- 感觉项目目录很乱时
- 定期清理（推荐每周一次）

### 2. 快速提交脚本

**Windows 用户**：
```bash
utils\quick-commit.bat "你的提交信息"
```

**Linux/Mac 用户**：
```bash
bash utils/quick-commit.sh "你的提交信息"
```

**功能**：
自动执行以下步骤：
1. 清理临时文件
2. `git add .`
3. `git commit -m "你的提交信息"`
4. `git push`

**示例**：
```bash
# Windows
utils\quick-commit.bat "完成人形检测模型训练"

# Linux/Mac
bash utils/quick-commit.sh "完成人形检测模型训练"
```

**等同于手动执行**：
```bash
utils\git-clean.bat
git add .
git commit -m "完成人形检测模型训练"
git push
```

## 🎯 推荐工作流

### 日常开发流程

```bash
# 1. 编写代码
# ... 你的工作 ...

# 2. 测试代码
python your_script.py

# 3. 一键提交（推荐）
utils\quick-commit.bat "添加了 xxx 功能"

# 或者分步操作
utils\git-clean.bat       # 清理
git add .                 # 添加
git commit -m "信息"      # 提交
git push                  # 推送
```

### 大型修改流程

```bash
# 1. 创建新分支
git checkout -b feature-new-model

# 2. 开发和测试
# ... 你的工作 ...

# 3. 清理并提交
utils\git-clean.bat
git add .
git commit -m "实现新模型"

# 4. 合并回主分支
git checkout main
git merge feature-new-model
git push
```

## 🔧 自定义脚本

你可以根据需要修改这些脚本，添加自己的清理规则。

### 添加新的清理规则

编辑 `git-clean.bat` 或 `git-clean.sh`，添加：

```bash
# 例如：清理所有 .log 文件
del /s /q *.log 2>nul          # Windows
find . -type f -name "*.log" -delete    # Linux/Mac
```

### 创建别名（可选）

**Windows PowerShell**：
在 PowerShell 配置文件中添加：
```powershell
function gc { utils\git-clean.bat }
function qc { utils\quick-commit.bat $args }
```

**Linux/Mac (Bash)**：
在 `~/.bashrc` 或 `~/.zshrc` 中添加：
```bash
alias gc='bash utils/git-clean.sh'
alias qc='bash utils/quick-commit.sh'
```

然后就可以简化命令：
```bash
gc                              # 清理
qc "提交信息"                   # 快速提交
```

## ⚠️ 注意事项

1. **清理脚本会永久删除文件**
   - 只删除缓存和临时文件
   - 不会删除你的代码和数据
   - 但仍建议先查看 `git status` 确认

2. **快速提交会提交所有更改**
   - 相当于 `git add .`
   - 如果只想提交特定文件，请手动操作

3. **推送前确认更改**
   - 可以先运行 `git status` 查看
   - 或使用 `git diff` 查看具体修改

## 📚 更多 Git 技巧

查看其他指南：
- [Git 使用指南](../00-environment/git-guide.md)
- [GitHub 配置指南](../00-environment/github-setup.md)

## 🛠️ 故障排除

### 脚本无法执行（Linux/Mac）

```bash
# 添加执行权限
chmod +x utils/git-clean.sh
chmod +x utils/quick-commit.sh
```

### Windows 执行策略问题

如果 PowerShell 不允许执行脚本：
```powershell
# 以管理员身份运行 PowerShell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Git Bash 在 Windows 上使用

Windows 用户也可以使用 Git Bash 运行 `.sh` 脚本：
```bash
bash utils/git-clean.sh
bash utils/quick-commit.sh "提交信息"
```
