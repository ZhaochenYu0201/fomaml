@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo 上传项目到GitHub
echo ========================================
echo.

REM 检查Git是否安装
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 错误: Git未安装
    echo.
    echo 请先安装Git:
    echo https://git-scm.com/download/win
    echo.
    pause
    exit /b 1
)

echo ✅ Git已安装
echo.

REM 检查是否已初始化
git status >nul 2>&1
if %errorlevel% neq 0 (
    echo 📦 初始化Git仓库...
    git init
    echo ✅ Git仓库已初始化
    echo.
)

REM 检查Git用户配置
git config user.name >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Git用户未配置
    echo.
    set /p GIT_USER="请输入你的GitHub用户名: "
    set /p GIT_EMAIL="请输入你的GitHub邮箱: "
    git config --global user.name "!GIT_USER!"
    git config --global user.email "!GIT_EMAIL!"
    echo ✅ Git用户配置完成
    echo.
)

echo 📝 添加文件到Git...
git add .

echo.
echo 📊 查看将要提交的文件:
echo ----------------------------------------
git status --short
echo ----------------------------------------
echo.

set /p CONTINUE="继续提交? (y/n) "
if /i not "!CONTINUE!"=="y" (
    echo 操作已取消
    pause
    exit /b 0
)

echo.
echo 💬 请输入提交信息
echo    提示: 简短描述这次提交的内容
echo    例如: "Initial commit: FOMAML implementation"
echo.
set /p COMMIT_MSG="提交信息 (按Enter使用默认): "
if "!COMMIT_MSG!"=="" set COMMIT_MSG=Initial commit: FOMAML implementation with Qwen3-4B support

echo.
echo 📤 创建提交...
git commit -m "!COMMIT_MSG!"

if %errorlevel% neq 0 (
    echo ❌ 提交失败
    pause
    exit /b 1
)

echo ✅ 提交成功
echo.

REM 检查是否已添加远程仓库
git remote -v | findstr "origin" >nul 2>&1
if %errorlevel% equ 0 (
    echo ⚠️  远程仓库已存在
    git remote -v
    echo.
    set /p OVERWRITE="是否要更新远程仓库地址? (y/n) "
    if /i "!OVERWRITE!"=="y" (
        git remote remove origin
    ) else (
        goto PUSH
    )
)

echo.
echo ========================================
echo 设置GitHub远程仓库
echo ========================================
echo.
echo 📌 请先在GitHub上创建新仓库:
echo    1. 访问 https://github.com/new
echo    2. 输入仓库名: meta_learning
echo    3. 选择Public或Private
echo    4. ⚠️  不要勾选 "Initialize this repository with a README"
echo    5. 点击 "Create repository"
echo    6. 复制仓库URL
echo.
echo 仓库URL格式:
echo    HTTPS: https://github.com/你的用户名/meta_learning.git
echo    SSH:   git@github.com:你的用户名/meta_learning.git
echo.

set /p REPO_URL="请输入GitHub仓库URL: "

if "!REPO_URL!"=="" (
    echo ❌ 错误: 仓库URL不能为空
    pause
    exit /b 1
)

echo.
echo 🔗 添加远程仓库...
git remote add origin !REPO_URL!

if %errorlevel% neq 0 (
    echo ❌ 添加远程仓库失败
    pause
    exit /b 1
)

echo ✅ 远程仓库已添加
echo.

:PUSH
echo 🚀 推送到GitHub...
echo.
echo 提示: 如果使用HTTPS，可能需要输入:
echo   - Username: 你的GitHub用户名
echo   - Password: Personal Access Token (不是GitHub密码!)
echo.
echo 如何获取Personal Access Token:
echo   1. 访问 https://github.com/settings/tokens
echo   2. Generate new token (classic)
echo   3. 勾选 repo 权限
echo   4. 复制生成的token作为密码使用
echo.

git branch -M main
git push -u origin main

if %errorlevel% neq 0 (
    echo.
    echo ❌ 推送失败
    echo.
    echo 常见问题:
    echo   1. 认证失败: 使用Personal Access Token而非密码
    echo   2. 远程仓库已有内容: 先pull或使用 git push -f
    echo   3. 网络问题: 检查网络连接
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo ✅ 上传成功！
echo ========================================
echo.
echo 🎉 你的项目已经上传到GitHub！
echo.
echo 📍 仓库地址: !REPO_URL!
echo 🌐 访问: !REPO_URL:.git=!
echo.
echo 📚 下次更新代码:
echo    git add .
echo    git commit -m "你的提交信息"
echo    git push
echo.
pause
