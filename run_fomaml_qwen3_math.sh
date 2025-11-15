#!/bin/bash
# FOMAML训练一键运行脚本
# 模型: Qwen3-4B-Instruct
# 数据集: MATH

set -e  # 遇到错误立即退出

echo "========================================"
echo "FOMAML-SFT 训练脚本"
echo "模型: Qwen3-4B-Instruct"
echo "数据集: MATH"
echo "========================================"

# ============================================
# 配置区域 - 请根据你的环境修改
# ============================================

# 模型路径（修改为你的本地路径）
MODEL_PATH="./models/Qwen3-4B-Instruct-2507"

# 数据输出目录
DATA_DIR="./data/math_meta"

# Checkpoint保存目录
CHECKPOINT_DIR="./checkpoints/fomaml_qwen3_4b_math"

# GPU数量（根据你的机器修改）
NUM_GPUS=4

# 配置文件
CONFIG_NAME="config_qwen3_4b_math"

# ============================================
# 步骤1: 检查环境
# ============================================

echo ""
echo "步骤1/4: 检查环境"
echo "----------------------------------------"

# 检查Python
if ! command -v python &> /dev/null; then
    echo "❌ 错误: 未找到Python"
    exit 1
fi
echo "✅ Python: $(python --version)"

# 检查必要的包
echo "检查Python包..."
python -c "import torch; print('✅ PyTorch:', torch.__version__)"
python -c "import datasets; print('✅ datasets:', datasets.__version__)"
python -c "import pandas; print('✅ pandas:', pandas.__version__)"
python -c "import transformers; print('✅ transformers:', transformers.__version__)"

# 检查模型路径
if [ ! -d "$MODEL_PATH" ]; then
    echo "⚠️  警告: 模型路径不存在: $MODEL_PATH"
    echo "   请在脚本顶部修改 MODEL_PATH 变量"
    read -p "   继续吗? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 检查GPU
echo "GPU信息:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ============================================
# 步骤2: 准备数据
# ============================================

echo ""
echo "步骤2/4: 准备MATH数据集"
echo "----------------------------------------"

if [ -d "$DATA_DIR/meta_train" ] && [ "$(ls -A $DATA_DIR/meta_train)" ]; then
    echo "⚠️  数据目录已存在: $DATA_DIR"
    read -p "   是否重新准备数据? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "正在准备数据..."
        python prepare_math_data.py \
            --output-dir "$DATA_DIR" \
            --support-ratio 0.30 \
            --query-ratio 0.40 \
            --validate
    else
        echo "✅ 使用现有数据"
    fi
else
    echo "正在准备数据..."
    python prepare_math_data.py \
        --output-dir "$DATA_DIR" \
        --support-ratio 0.30 \
        --query-ratio 0.40 \
        --validate
fi

# ============================================
# 步骤3: 更新配置文件
# ============================================

echo ""
echo "步骤3/4: 更新配置文件"
echo "----------------------------------------"

# 使用sed更新配置文件中的路径（如果需要）
if [ -f "${CONFIG_NAME}.yaml" ]; then
    echo "✅ 配置文件: ${CONFIG_NAME}.yaml"
    echo "   请确认配置文件中的模型路径已正确设置"
else
    echo "❌ 错误: 配置文件不存在: ${CONFIG_NAME}.yaml"
    exit 1
fi

# ============================================
# 步骤4: 开始训练
# ============================================

echo ""
echo "步骤4/4: 开始FOMAML训练"
echo "----------------------------------------"
echo "配置信息:"
echo "  - GPU数量: $NUM_GPUS"
echo "  - 配置文件: ${CONFIG_NAME}.yaml"
echo "  - Checkpoint目录: $CHECKPOINT_DIR"
echo ""

read -p "开始训练? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "训练已取消"
    exit 0
fi

# 创建checkpoint目录
mkdir -p "$CHECKPOINT_DIR"

# 启动训练
echo ""
echo "🚀 启动训练..."
echo "========================================"

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    maml_sft_trainer.py \
    --config-name "$CONFIG_NAME"

echo ""
echo "========================================"
echo "✅ 训练完成！"
echo "========================================"
echo ""
echo "Checkpoint保存在: $CHECKPOINT_DIR"
echo ""
echo "下一步:"
echo "  1. 查看Wandb日志分析训练情况"
echo "  2. 使用checkpoint进行few-shot评估"
echo ""
