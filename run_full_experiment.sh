#!/bin/bash
# 完整实验运行脚本
# 用于一键运行FOMAML-SFT vs Baseline SFT对比实验

set -e  # 遇到错误立即退出

echo "=================================="
echo "FOMAML-SFT Full Experiment Pipeline"
echo "=================================="

# 配置变量
DATA_DIR="./data/math_science_meta"
FOMAML_CKPT_DIR="./checkpoints/fomaml_math_science"
SFT_CKPT_DIR="./checkpoints/baseline_sft"
RESULTS_DIR="./results"
NUM_GPUS=4

# ====== Step 1: 数据准备 ======
echo ""
echo "Step 1: Preparing data..."
echo ""

if [ ! -d "$DATA_DIR" ]; then
    python prepare_math_science_data.py \
        --output-dir "$DATA_DIR" \
        --seed 42 \
        --support-ratio 0.2 \
        --query-ratio 0.3

    echo "✓ Data preparation complete"
else
    echo "✓ Data already exists, skipping preparation"
fi

# ====== Step 2: Baseline SFT训练 ======
echo ""
echo "Step 2: Training Baseline SFT..."
echo ""

if [ ! -d "$SFT_CKPT_DIR" ]; then
    echo "Starting Baseline SFT training with $NUM_GPUS GPUs..."

    torchrun --nproc_per_node=$NUM_GPUS \
        verl/verl/trainer/fsdp_sft_trainer.py \
        --config-path "../$DATA_DIR" \
        --config-name config_baseline_sft

    echo "✓ Baseline SFT training complete"
else
    echo "✓ Baseline SFT checkpoint exists, skipping training"
fi

# ====== Step 3: FOMAML-SFT训练 ======
echo ""
echo "Step 3: Training FOMAML-SFT..."
echo ""

if [ ! -d "$FOMAML_CKPT_DIR" ]; then
    echo "Starting FOMAML-SFT training with $NUM_GPUS GPUs..."

    torchrun --nproc_per_node=$NUM_GPUS \
        maml_sft_trainer.py \
        --config-path "$DATA_DIR" \
        --config-name config_fomaml_math_science

    echo "✓ FOMAML-SFT training complete"
else
    echo "✓ FOMAML-SFT checkpoint exists, skipping training"
fi

# ====== Step 4: Few-Shot评估 ======
echo ""
echo "Step 4: Evaluating Few-Shot Learning..."
echo ""

# 定义评估任务
EVAL_TASKS="algebra geometry number_theory word_problems"
N_SHOTS="0 5 10 25 50"
N_RUNS=5

# 评估Base Model
echo "Evaluating Base Model..."
python evaluate_few_shot.py \
    --model-path "meta-llama/Llama-3.2-1B" \
    --model-type base \
    --data-dir "$DATA_DIR/few_shot_eval" \
    --eval-tasks $EVAL_TASKS \
    --n-shots $N_SHOTS \
    --n-runs $N_RUNS \
    --output-dir "$RESULTS_DIR/base_model"

echo "✓ Base Model evaluation complete"

# 评估Baseline SFT
echo "Evaluating Baseline SFT..."

# 找到最新的checkpoint
LATEST_SFT_CKPT=$(ls -td $SFT_CKPT_DIR/global_step_* | head -1)

python evaluate_few_shot.py \
    --model-path "$LATEST_SFT_CKPT" \
    --model-type sft \
    --data-dir "$DATA_DIR/few_shot_eval" \
    --eval-tasks $EVAL_TASKS \
    --n-shots $N_SHOTS \
    --n-runs $N_RUNS \
    --output-dir "$RESULTS_DIR/baseline_sft"

echo "✓ Baseline SFT evaluation complete"

# 评估FOMAML-SFT
echo "Evaluating FOMAML-SFT..."

# 找到最新的checkpoint
LATEST_FOMAML_CKPT=$(ls -t $FOMAML_CKPT_DIR/maml_checkpoint_step_*.pt | head -1)

python evaluate_few_shot.py \
    --model-path "$LATEST_FOMAML_CKPT" \
    --model-type fomaml \
    --data-dir "$DATA_DIR/few_shot_eval" \
    --eval-tasks $EVAL_TASKS \
    --n-shots $N_SHOTS \
    --n-runs $N_RUNS \
    --output-dir "$RESULTS_DIR/fomaml_sft"

echo "✓ FOMAML-SFT evaluation complete"

# ====== Step 5: 结果分析 ======
echo ""
echo "Step 5: Analyzing results..."
echo ""

python << EOF
import json
import numpy as np
from pathlib import Path

results_dir = Path("$RESULTS_DIR")

# 加载结果
with open(results_dir / "base_model" / "base_results.json") as f:
    base_results = json.load(f)

with open(results_dir / "baseline_sft" / "sft_results.json") as f:
    sft_results = json.load(f)

with open(results_dir / "fomaml_sft" / "fomaml_results.json") as f:
    fomaml_results = json.load(f)

print("="*60)
print("EXPERIMENT RESULTS SUMMARY")
print("="*60)

for task in fomaml_results.keys():
    print(f"\n{task.upper()}:")
    print("-" * 40)

    for n_shots in [0, 5, 10, 25, 50]:
        n_str = str(n_shots)

        if n_str not in fomaml_results[task]:
            continue

        base_acc = base_results[task][n_str]['accuracy_mean'] * 100
        sft_acc = sft_results[task][n_str]['accuracy_mean'] * 100
        fomaml_acc = fomaml_results[task][n_str]['accuracy_mean'] * 100

        print(f"{n_shots:3d}-shot:")
        print(f"  Base:    {base_acc:5.1f}%")
        print(f"  SFT:     {sft_acc:5.1f}%")
        print(f"  FOMAML:  {fomaml_acc:5.1f}%")
        print(f"  Gain:    {fomaml_acc - sft_acc:+5.1f}% ({(fomaml_acc/sft_acc-1)*100:+.1f}%)")

# 计算总体统计
print("\n" + "="*60)
print("OVERALL STATISTICS")
print("="*60)

all_fomaml_accs = []
all_sft_accs = []

for task in fomaml_results.keys():
    for n_shots in [5, 10, 25, 50]:  # 排除zero-shot
        n_str = str(n_shots)
        if n_str in fomaml_results[task] and n_str in sft_results[task]:
            all_fomaml_accs.append(fomaml_results[task][n_str]['accuracy_mean'])
            all_sft_accs.append(sft_results[task][n_str]['accuracy_mean'])

fomaml_mean = np.mean(all_fomaml_accs) * 100
sft_mean = np.mean(all_sft_accs) * 100
improvement = fomaml_mean - sft_mean

print(f"Average Few-Shot Accuracy:")
print(f"  Baseline SFT:  {sft_mean:.1f}%")
print(f"  FOMAML-SFT:    {fomaml_mean:.1f}%")
print(f"  Improvement:   {improvement:+.1f}% ({(improvement/sft_mean)*100:+.1f}%)")

# 计算样本效率
print(f"\nSample Efficiency (to reach 70% accuracy):")

def compute_sample_efficiency(results, target=0.7):
    for n_shots in sorted([int(k) for k in results.keys()]):
        acc = results[str(n_shots)]['accuracy_mean']
        if acc >= target:
            return n_shots
    return None

for task in fomaml_results.keys():
    fomaml_samples = compute_sample_efficiency(fomaml_results[task])
    sft_samples = compute_sample_efficiency(sft_results[task])

    if fomaml_samples and sft_samples:
        efficiency = sft_samples / fomaml_samples
        print(f"  {task}: FOMAML={fomaml_samples}, SFT={sft_samples}, Gain={efficiency:.1f}x")

print("\n" + "="*60)
print("Results saved to: $RESULTS_DIR")
print("="*60)
EOF

echo ""
echo "=================================="
echo "Experiment Complete! 🎉"
echo "=================================="
echo ""
echo "Results are available in: $RESULTS_DIR"
echo ""
echo "Next steps:"
echo "1. Check learning curve plots in $RESULTS_DIR/*/learning_curve_*.png"
echo "2. Review detailed results in $RESULTS_DIR/*/results.json"
echo "3. Run custom analysis scripts on the results"
echo ""
