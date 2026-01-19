# Git 本地使用指南

## 基础命令

### 查看状态
```bash
git status                  # 查看当前状态
git log                     # 查看提交历史
git log --oneline           # 简洁查看历史
git log --graph --all       # 图形化查看所有分支
```

### 提交流程
```bash
git add .                   # 添加所有修改
git add <file>              # 添加特定文件
git commit -m "提交信息"    # 提交修改
git commit -am "信息"       # 添加并提交（仅已跟踪文件）
```

## 回溯与撤销

### 查看历史版本（只看不改）
```bash
git checkout <commit-hash>           # 切换到历史版本查看
git checkout master                  # 返回最新版本
git show <commit-hash>               # 查看某次提交的详情
```

### 查看文件历史
```bash
git log <file>                       # 查看文件修改历史
git log -p <file>                    # 查看文件详细修改
git show <commit>:<file>             # 查看某版本的文件内容
git diff HEAD~1 <file>               # 对比文件变化
```

### 撤销未提交的修改
```bash
git restore <file>                   # 撤销工作区修改（推荐）
git checkout -- <file>               # 撤销工作区修改（旧版）
git restore --staged <file>          # 取消暂存
git reset HEAD <file>                # 取消暂存（旧版）
```

### 回退提交

#### 软回退（保留修改在暂存区）
```bash
git reset --soft HEAD^               # 回退1个提交
git reset --soft HEAD~3              # 回退3个提交
git reset --soft <commit-hash>       # 回退到指定提交
```
**用途**：想重新编辑提交信息或合并提交

#### 混合回退（保留修改在工作区）
```bash
git reset HEAD^                      # 回退1个提交（默认）
git reset <commit-hash>              # 回退到指定提交
```
**用途**：想重新组织提交内容

#### 硬回退（完全删除修改）⚠️
```bash
git reset --hard HEAD^               # 完全回退1个提交
git reset --hard <commit-hash>       # 完全回退到指定版本
```
**警告**：会永久删除修改，谨慎使用！

### 撤销已推送的提交（安全方式）
```bash
git revert <commit-hash>             # 创建新提交来撤销指定提交
git revert HEAD                      # 撤销最后一次提交
```
**说明**：推荐用于已推送的提交，不会改写历史

## 分支管理

### 创建和切换分支
```bash
git branch                           # 查看分支
git branch <name>                    # 创建分支
git checkout <name>                  # 切换分支
git checkout -b <name>               # 创建并切换分支
git switch <name>                    # 切换分支（新版推荐）
git switch -c <name>                 # 创建并切换（新版）
```

### 合并分支
```bash
git merge <branch>                   # 合并指定分支到当前分支
git merge --no-ff <branch>           # 保留分支历史的合并
```

### 删除分支
```bash
git branch -d <name>                 # 删除已合并分支
git branch -D <name>                 # 强制删除分支
```

## 实用技巧

### 临时保存修改
```bash
git stash                            # 保存当前修改
git stash list                       # 查看保存列表
git stash pop                        # 恢复最近保存的修改
git stash apply stash@{0}            # 恢复指定保存
git stash drop stash@{0}             # 删除指定保存
```

### 修改最后一次提交
```bash
git commit --amend                   # 修改最后一次提交信息
git commit --amend --no-edit         # 补充文件到最后一次提交
```

### 查找和对比
```bash
git diff                             # 查看工作区修改
git diff --staged                    # 查看暂存区修改
git diff HEAD~1 HEAD                 # 对比两个提交
git grep "search term"               # 在代码中搜索
```

### 标签管理
```bash
git tag v1.0.0                       # 创建轻量标签
git tag -a v1.0.0 -m "版本1.0"       # 创建附注标签
git tag                              # 查看所有标签
git show v1.0.0                      # 查看标签详情
```

## 常见场景

### 场景1：我改错了文件，想恢复
```bash
# 如果还没 add
git restore <file>

# 如果已经 add 但没 commit
git restore --staged <file>          # 先取消暂存
git restore <file>                   # 再恢复文件
```

### 场景2：我想撤销最后一次提交
```bash
# 保留修改，重新提交
git reset --soft HEAD^

# 完全删除这次提交
git reset --hard HEAD^               # ⚠️ 谨慎使用
```

### 场景3：我想回到某个历史版本查看
```bash
git log --oneline                    # 找到版本号
git checkout <commit-hash>           # 切换过去查看
git checkout master                  # 查看完回来
```

### 场景4：我想在某个历史版本基础上继续开发
```bash
git checkout <commit-hash>
git checkout -b new-branch-name      # 创建新分支继续开发
```

### 场景5：查看某个文件的历史版本
```bash
git log --oneline <file>             # 查看文件历史
git show <commit>:<file>             # 查看指定版本的文件内容
```

## 回溯提交的表示方法

```bash
HEAD              # 当前版本
HEAD^             # 上一个版本
HEAD^^            # 上上个版本
HEAD~3            # 往上3个版本
HEAD~10           # 往上10个版本
<commit-hash>     # 具体的提交哈希值
```

## 救命稻草：reflog

如果你 hard reset 后后悔了：
```bash
git reflog                           # 查看所有操作历史
git reset --hard <commit-hash>       # 恢复到任何历史操作点
```

Git 的 reflog 会记录你的每一步操作，即使是被删除的提交也能找回！

## 最佳实践

1. ✅ **频繁提交**：小步提交，方便回溯
2. ✅ **清晰的提交信息**：写明做了什么
3. ✅ **使用分支**：实验性修改在新分支进行
4. ✅ **提交前检查**：`git status` 和 `git diff` 确认修改
5. ⚠️ **谨慎使用 --hard**：会永久删除数据
6. ✅ **使用 .gitignore**：避免提交不必要的文件
7. ✅ **定期备份**：虽然 Git 很安全，但定期推送到远程仓库更保险

## 配置推荐

```bash
# 设置默认编辑器
git config --global core.editor "code --wait"

# 设置默认分支名
git config --global init.defaultBranch main

# 启用颜色
git config --global color.ui auto

# 设置别名
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.lg "log --graph --oneline --all"
```

## 资源链接

- [Git 官方文档](https://git-scm.com/doc)
- [GitHub Git 教程](https://docs.github.com/en/get-started/using-git)
- [Pro Git 中文版](https://git-scm.com/book/zh/v2)
