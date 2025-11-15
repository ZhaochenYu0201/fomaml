# 如何上传项目到GitHub

本文档提供详细的步骤说明如何将本项目上传到GitHub。

---

## 📋 前置要求

### 1. 安装Git

检查是否已安装Git：
```bash
git --version
```

如果未安装，请下载安装：
- Windows: https://git-scm.com/download/win
- Mac: `brew install git`
- Linux: `sudo apt-get install git` 或 `sudo yum install git`

### 2. 配置Git（首次使用）

```bash
# 设置你的用户名
git config --global user.name "你的GitHub用户名"

# 设置你的邮箱（与GitHub账号一致）
git config --global user.email "你的邮箱@example.com"
```

### 3. GitHub账号

确保你已经有GitHub账号，如果没有请访问 https://github.com 注册。

---

## 🚀 方法一：通过GitHub网页创建仓库（推荐）

### 步骤1: 在GitHub上创建新仓库

1. 登录GitHub
2. 点击右上角的 `+` → `New repository`
3. 填写仓库信息：
   - **Repository name**: `meta_learning` 或其他名字
   - **Description**: `FOMAML implementation for LLM meta-learning`
   - **Public/Private**: 选择公开或私有
   - ⚠️ **不要勾选** "Initialize this repository with a README"
4. 点击 `Create repository`

### 步骤2: 初始化本地Git仓库

在项目目录下打开命令行，执行：

```bash
# 初始化git仓库
git init

# 添加所有文件到暂存区
git add .

# 查看将要提交的文件
git status

# 创建第一次提交
git commit -m "Initial commit: FOMAML implementation with Qwen3-4B support"
```

### 步骤3: 连接到GitHub远程仓库

将GitHub上显示的命令复制执行（替换为你的用户名和仓库名）：

```bash
# 添加远程仓库（替换为你的仓库地址）
git remote add origin https://github.com/你的用户名/meta_learning.git

# 或者使用SSH（需要先配置SSH key）
# git remote add origin git@github.com:你的用户名/meta_learning.git

# 推送到GitHub
git branch -M main
git push -u origin main
```

**如果推送失败**，可能需要身份验证：

#### 使用Personal Access Token (推荐)

1. 访问 https://github.com/settings/tokens
2. 点击 `Generate new token` → `Generate new token (classic)`
3. 设置：
   - Note: `meta_learning_upload`
   - Expiration: 选择过期时间
   - 勾选: `repo` (所有权限)
4. 点击 `Generate token`
5. **复制token**（只显示一次！）
6. 推送时使用token作为密码

```bash
# 推送时会要求输入用户名和密码
# Username: 你的GitHub用户名
# Password: 刚才复制的token（不是GitHub密码！）
git push -u origin main
```

---

## 🔄 方法二：使用自动化脚本

我为你准备了一个自动化脚本，运行即可：

### Windows用户

创建 `upload_to_github.bat`:

```batch
@echo off
echo ========================================
echo 上传项目到GitHub
echo ========================================

REM 检查是否已初始化
git status >nul 2>&1
if %errorlevel% neq 0 (
    echo 初始化Git仓库...
    git init
)

echo.
echo 添加文件到Git...
git add .

echo.
echo 查看状态...
git status

echo.
set /p CONTINUE="继续提交? (y/n) "
if /i not "%CONTINUE%"=="y" exit /b 0

echo.
set /p COMMIT_MSG="输入提交信息 (或按Enter使用默认): "
if "%COMMIT_MSG%"=="" set COMMIT_MSG=Initial commit: FOMAML implementation

git commit -m "%COMMIT_MSG%"

echo.
echo ========================================
echo 现在需要添加GitHub远程仓库
echo ========================================
echo 请按照以下步骤操作:
echo 1. 访问 https://github.com
echo 2. 创建新仓库 (不要初始化README)
echo 3. 复制仓库URL
echo.

set /p REPO_URL="输入仓库URL: "

git remote add origin %REPO_URL%
git branch -M main
git push -u origin main

echo.
echo ========================================
echo 完成！访问你的GitHub仓库查看
echo ========================================
pause
```

### Linux/Mac用户

创建 `upload_to_github.sh`:

```bash
#!/bin/bash
echo "========================================"
echo "上传项目到GitHub"
echo "========================================"

# 检查是否已初始化
if ! git status &> /dev/null; then
    echo "初始化Git仓库..."
    git init
fi

echo ""
echo "添加文件到Git..."
git add .

echo ""
echo "查看状态..."
git status

echo ""
read -p "继续提交? (y/n) " CONTINUE
if [[ ! $CONTINUE =~ ^[Yy]$ ]]; then
    exit 0
fi

echo ""
read -p "输入提交信息 (或按Enter使用默认): " COMMIT_MSG
if [ -z "$COMMIT_MSG" ]; then
    COMMIT_MSG="Initial commit: FOMAML implementation"
fi

git commit -m "$COMMIT_MSG"

echo ""
echo "========================================"
echo "现在需要添加GitHub远程仓库"
echo "========================================"
echo "请按照以下步骤操作:"
echo "1. 访问 https://github.com"
echo "2. 创建新仓库 (不要初始化README)"
echo "3. 复制仓库URL"
echo ""

read -p "输入仓库URL: " REPO_URL

git remote add origin $REPO_URL
git branch -M main
git push -u origin main

echo ""
echo "========================================"
echo "完成！访问你的GitHub仓库查看"
echo "========================================"
```

运行脚本：
```bash
# Windows
upload_to_github.bat

# Linux/Mac
chmod +x upload_to_github.sh
./upload_to_github.sh
```

---

## 📝 之后的更新

项目上传后，如果有修改要推送到GitHub：

```bash
# 1. 查看修改的文件
git status

# 2. 添加修改的文件
git add .

# 3. 提交修改
git commit -m "描述你的修改"

# 4. 推送到GitHub
git push
```

---

## 🔧 常见问题

### Q1: 推送时提示 "fatal: remote origin already exists"

```bash
# 删除已存在的remote
git remote remove origin

# 重新添加
git remote add origin https://github.com/你的用户名/meta_learning.git
```

### Q2: 推送时提示认证失败

使用Personal Access Token：
1. 生成token (见上文)
2. 推送时输入token作为密码

或配置SSH key：
```bash
# 生成SSH key
ssh-keygen -t ed25519 -C "你的邮箱@example.com"

# 复制公钥
cat ~/.ssh/id_ed25519.pub

# 添加到GitHub: Settings → SSH and GPG keys → New SSH key
```

### Q3: 文件太大无法推送

检查 `.gitignore` 是否正确配置：
```bash
# 查看即将提交的文件大小
git ls-files | xargs du -sh | sort -h

# 如果有大文件，添加到.gitignore
echo "大文件路径" >> .gitignore
git rm --cached 大文件路径
git commit -m "Remove large files"
```

### Q4: 如何忽略已经提交的文件

```bash
# 从Git中移除但保留本地文件
git rm --cached 文件名

# 添加到.gitignore
echo "文件名" >> .gitignore

# 提交修改
git commit -m "Remove tracked file"
```

---

## 📚 Git常用命令速查

```bash
# 查看状态
git status

# 查看提交历史
git log --oneline

# 查看远程仓库
git remote -v

# 创建分支
git branch 分支名

# 切换分支
git checkout 分支名

# 合并分支
git merge 分支名

# 拉取远程更新
git pull

# 查看差异
git diff
```

---

## ⚠️ 重要提示

### 不要上传的文件（已在.gitignore中）

- ✅ 模型文件 (*.bin, *.safetensors, models/)
- ✅ 数据文件 (*.parquet, data/)
- ✅ Checkpoint (checkpoints/)
- ✅ 日志文件 (wandb/, logs/)
- ✅ Python缓存 (__pycache__/)

### 应该上传的文件

- ✅ 所有 .py 脚本
- ✅ 配置文件 (.yaml)
- ✅ 文档 (.md)
- ✅ 运行脚本 (.sh, .bat)
- ✅ requirements.txt
- ✅ .gitignore

---

## 🎉 完成！

上传成功后，你的GitHub仓库地址为：
```
https://github.com/你的用户名/meta_learning
```

分享给其他人时，他们可以这样使用：
```bash
git clone https://github.com/你的用户名/meta_learning.git
cd meta_learning
pip install -r requirements.txt
python test_environment.py
```

---

如有问题，参考 [GitHub官方文档](https://docs.github.com/cn)
