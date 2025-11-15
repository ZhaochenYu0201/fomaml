@echo off
REM FOMAML训练一键运行脚本 (Windows版本)
REM 模型: Qwen3-4B-Instruct
REM 数据集: MATH

setlocal enabledelayedexpansion

echo ========================================
echo FOMAML-SFT 训练脚本
echo 模型: Qwen3-4B-Instruct
echo 数据集: MATH
echo ========================================

REM ============================================
REM 配置区域 - 请根据你的环境修改
REM ============================================

REM 模型路径（修改为你的本地路径）
set MODEL_PATH=./models/Qwen3-4B-Instruct-2507

REM 数据输出目录
set DATA_DIR=./data/math_meta

REM Checkpoint保存目录
set CHECKPOINT_DIR=./checkpoints/fomaml_qwen3_4b_math

REM GPU数量（根据你的机器修改）
set NUM_GPUS=4

REM 配置文件
set CONFIG_NAME=config_qwen3_4b_math

REM ============================================
REM 步骤1: 检查环境
REM ============================================

echo.
echo 步骤1/4: 检查环境
echo ----------------------------------------

REM 检查Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 错误: 未找到Python
    exit /b 1
)
python --version

REM 检查必要的包
echo 检查Python包...
python -c "import torch; print('✅ PyTorch:', torch.__version__)"
python -c "import datasets; print('✅ datasets:', datasets.__version__)"
python -c "import pandas; print('✅ pandas:', pandas.__version__)"
python -c "import transformers; print('✅ transformers:', transformers.__version__)"

REM 检查GPU
echo GPU信息:
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo.

REM ============================================
REM 步骤2: 准备数据
REM ============================================

echo.
echo 步骤2/4: 准备MATH数据集
echo ----------------------------------------

if exist "%DATA_DIR%\meta_train" (
    echo ⚠️  数据目录已存在: %DATA_DIR%
    set /p REPLY="   是否重新准备数据? (y/n) "
    if /i "!REPLY!"=="y" (
        echo 正在准备数据...
        python prepare_math_data.py --output-dir "%DATA_DIR%" --support-ratio 0.30 --query-ratio 0.40 --validate
    ) else (
        echo ✅ 使用现有数据
    )
) else (
    echo 正在准备数据...
    python prepare_math_data.py --output-dir "%DATA_DIR%" --support-ratio 0.30 --query-ratio 0.40 --validate
)

REM ============================================
REM 步骤3: 检查配置文件
REM ============================================

echo.
echo 步骤3/4: 检查配置文件
echo ----------------------------------------

if exist "%CONFIG_NAME%.yaml" (
    echo ✅ 配置文件: %CONFIG_NAME%.yaml
    echo    请确认配置文件中的模型路径已正确设置
) else (
    echo ❌ 错误: 配置文件不存在: %CONFIG_NAME%.yaml
    exit /b 1
)

REM ============================================
REM 步骤4: 开始训练
REM ============================================

echo.
echo 步骤4/4: 开始FOMAML训练
echo ----------------------------------------
echo 配置信息:
echo   - GPU数量: %NUM_GPUS%
echo   - 配置文件: %CONFIG_NAME%.yaml
echo   - Checkpoint目录: %CHECKPOINT_DIR%
echo.

set /p REPLY="开始训练? (y/n) "
if /i not "!REPLY!"=="y" (
    echo 训练已取消
    exit /b 0
)

REM 创建checkpoint目录
if not exist "%CHECKPOINT_DIR%" mkdir "%CHECKPOINT_DIR%"

REM 启动训练
echo.
echo 🚀 启动训练...
echo ========================================

torchrun --nproc_per_node=%NUM_GPUS% --master_port=29500 maml_sft_trainer.py --config-name %CONFIG_NAME%

echo.
echo ========================================
echo ✅ 训练完成！
echo ========================================
echo.
echo Checkpoint保存在: %CHECKPOINT_DIR%
echo.
echo 下一步:
echo   1. 查看Wandb日志分析训练情况
echo   2. 使用checkpoint进行few-shot评估
echo.

pause
