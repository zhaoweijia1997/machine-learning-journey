# GitHub 远程仓库配置指南

## 步骤 1: 在 GitHub 上创建仓库

### 方法 A：通过网页创建（推荐）

1. **登录 GitHub**
   - 访问: https://github.com
   - 使用你的账号登录

2. **创建新仓库**
   - 点击右上角 `+` 号 → `New repository`
   - 或直接访问: https://github.com/new

3. **填写仓库信息**
   ```
   Repository name: machine-learning-journey
   Description: 我的机器学习学习之旅 - 专注计算机视觉和人形检测

   ✅ Public (公开仓库)
   ❌ 不要勾选 "Initialize this repository with:"
      - 不要添加 README
      - 不要添加 .gitignore
      - 不要添加 license

   原因：我们本地已经有了这些文件
   ```

4. **点击 `Create repository`**

5. **复制仓库 URL**
   - 创建后会显示两个 URL：
     - HTTPS: `https://github.com/你的用户名/machine-learning-journey.git`
     - SSH: `git@github.com:你的用户名/machine-learning-journey.git`

## 步骤 2: 选择认证方式

### 方式 A：HTTPS（简单，推荐新手）

**优点**：配置简单，不需要 SSH 密钥
**缺点**：每次推送需要输入密码或 token

**配置步骤**：
1. 使用 HTTPS URL 关联仓库（见步骤 3）
2. 推送时会要求认证（见步骤 4）

### 方式 B：SSH（推荐长期使用）

**优点**：配置后无需每次输入密码
**缺点**：初次配置稍复杂

**配置步骤**：
```bash
# 1. 生成 SSH 密钥（如果还没有）
ssh-keygen -t ed25519 -C "zhaoweijia1997@gmail.com"
# 一路回车，使用默认设置

# 2. 查看公钥
cat ~/.ssh/id_ed25519.pub
# 复制输出的内容

# 3. 添加到 GitHub
# 访问: https://github.com/settings/keys
# 点击 "New SSH key"
# Title: 随便填，如 "My Laptop"
# Key: 粘贴刚才复制的公钥
# 点击 "Add SSH key"

# 4. 测试连接
ssh -T git@github.com
# 成功会显示: Hi 你的用户名! You've successfully authenticated...
```

## 步骤 3: 关联本地仓库到 GitHub

### 使用 HTTPS（简单方式）
```bash
cd "c:\Users\zhaow\Desktop\Machine Learning"

# 添加远程仓库（替换成你的用户名）
git remote add origin https://github.com/你的用户名/machine-learning-journey.git

# 确认远程仓库
git remote -v
```

### 使用 SSH（需先完成 SSH 配置）
```bash
cd "c:\Users\zhaow\Desktop\Machine Learning"

# 添加远程仓库（替换成你的用户名）
git remote add origin git@github.com:你的用户名/machine-learning-journey.git

# 确认远程仓库
git remote -v
```

## 步骤 4: 推送代码到 GitHub

### 首次推送
```bash
# 将主分支重命名为 main（GitHub 新标准）
git branch -M main

# 推送到远程仓库
git push -u origin main
```

### HTTPS 认证问题

如果使用 HTTPS，推送时会要求认证：

**Windows 推荐方式：使用 Personal Access Token (PAT)**

1. **创建 Token**
   - 访问: https://github.com/settings/tokens
   - 点击 `Generate new token` → `Generate new token (classic)`
   - Note: 填写 "Machine Learning Project"
   - Expiration: 选择过期时间（建议 90 days 或 No expiration）
   - 勾选权限：
     - ✅ repo（完整权限）
   - 点击 `Generate token`
   - **重要**：复制生成的 token（只显示一次！）

2. **推送时使用 Token**
   ```bash
   git push -u origin main

   # 会提示输入用户名和密码
   Username: 你的GitHub用户名
   Password: 粘贴刚才的 token（不是密码！）
   ```

3. **保存凭证（避免每次输入）**
   ```bash
   # Windows 会自动使用凭据管理器保存
   # 首次输入后，后续推送无需再输入
   ```

## 步骤 5: 验证推送成功

1. 访问你的 GitHub 仓库页面
2. 应该能看到所有文件和提交历史
3. README.md 会自动显示在仓库首页

## 日常使用

### 推送新的修改
```bash
# 1. 查看状态
git status

# 2. 添加修改
git add .

# 3. 提交
git commit -m "描述你的修改"

# 4. 推送到 GitHub
git push
```

### 从 GitHub 拉取更新（如果在其他地方修改了）
```bash
git pull
```

### 查看远程仓库信息
```bash
git remote -v                    # 查看远程仓库
git remote show origin           # 查看详细信息
```

### 修改远程仓库 URL
```bash
# 如果需要切换 HTTPS 和 SSH
git remote set-url origin <新的URL>
```

## 常见问题

### 问题 1: 推送被拒绝
```
error: failed to push some refs to 'github.com:xxx/xxx.git'
```

**原因**：远程仓库有本地没有的提交

**解决**：
```bash
git pull --rebase origin main
git push
```

### 问题 2: 认证失败
```
Authentication failed
```

**HTTPS 解决方案**：
- 确保使用的是 Personal Access Token，不是密码
- 重新生成 token

**SSH 解决方案**：
```bash
ssh -T git@github.com           # 测试 SSH 连接
```

### 问题 3: 远程仓库已存在同名文件
如果你在 GitHub 创建仓库时不小心添加了 README：
```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```

## 推荐的 GitHub 仓库设置

创建仓库后，建议：

1. **添加 Topics**（标签）
   - machine-learning
   - computer-vision
   - deep-learning
   - pytorch
   - yolo

2. **编辑 About**
   - 添加描述和网站链接

3. **启用 GitHub Pages**（可选）
   - Settings → Pages
   - 可以展示你的学习笔记

## 可选：安装 GitHub CLI

更方便的命令行工具：

**Windows 安装**：
```bash
# 使用 winget
winget install GitHub.cli

# 或下载安装包
# https://cli.github.com/
```

**认证**：
```bash
gh auth login
```

**创建仓库**（以后可以直接用）：
```bash
gh repo create machine-learning-journey --public
```

## 学习资源

- [GitHub 官方文档](https://docs.github.com/)
- [Git 和 GitHub 教程](https://guides.github.com/)
- [GitHub Skills](https://skills.github.com/)
