# FOMAML-SFT完整实验运行手册

## 目标

验证FOMAML-SFT在数学和科学推理任务上相比标准SFT的few-shot学习优势。

---

## 实验流程总览

```
Step 1: 数据准备 (1-2天)
   ↓
Step 2: Baseline训练 (2-3天)
   ↓
Step 3: FOMAML-SFT训练 (3-5天)
   ↓
Step 4: Few-Shot评估 (2-3天)
   ↓
Step 5: 结果分析 (1-2天)
```

**总计：9-15天**

---

## Step 1: 数据准备

### 1.1 安装依赖

```bash
# 基础环境
pip install torch transformers datasets pandas pyarrow numpy scipy matplotlib seaborn tqdm

# verl框架
git clone https://github.com/volcengine/verl.git
cd verl
pip install -e .
cd ..

# 其他依赖
pip install hydra-core omegaconf tensordict wandb pyyaml
```

### 1.2 运行数据准备脚本

```bash
# 准备所有数据集（推荐）
python prepare_math_science_data.py \
    --output-dir ./data/math_science_meta \
    --seed 42 \
    --support-ratio 0.2 \
    --query-ratio 0.3

# 这会创建以下结构：
# data/math_science_meta/
# ├── meta_train/
# │   ├── algebra_support.parquet
# │   ├── algebra_query.parquet
# │   ├── geometry_support.parquet
# │   ├── geometry_query.parquet
# │   └── ... (其他任务)
# ├── few_shot_eval/
# │   ├── algebra_test.parquet
# │   ├── algebra_5shot.parquet
# │   ├── algebra_10shot.parquet
# │   └── ... (其他任务和shot数)
# ├── baseline_sft_all_mixed.parquet  # Baseline SFT训练数据
# ├── config_fomaml_math_science.yaml  # FOMAML配置
# └── config_baseline_sft.yaml         # Baseline配置
```

### 1.3 验证数据

```bash
# 检查数据统计
python -c "
import pandas as pd
from pathlib import Path

data_dir = Path('./data/math_science_meta/meta_train')
for file in sorted(data_dir.glob('*_support.parquet')):
    df = pd.read_parquet(file)
    print(f'{file.name}: {len(df)} samples')
    print(f'  Columns: {df.columns.tolist()}')
    print(f'  Example prompt length: {len(df.iloc[0][\"prompt\"])} chars')
    print()
"
```

**预期输出：**
```
algebra_support.parquet: ~300 samples
  Columns: ['prompt', 'response', 'metadata']
  Example prompt length: ~200 chars

geometry_support.parquet: ~250 samples
...
```

---

## Step 2: Baseline SFT训练

### 2.1 使用verl训练Baseline SFT

```bash
# 单GPU
python verl/verl/trainer/sft_trainer.py \
    --config-path ../data/math_science_meta \
    --config-name config_baseline_sft

# 多GPU (4卡)
torchrun --nproc_per_node=4 \
    verl/verl/trainer/sft_trainer.py \
    --config-path ../data/math_science_meta \
    --config-name config_baseline_sft
```

### 2.2 监控训练

```python
# 使用wandb监控
# 关键指标：
# - train/loss: 应该持续下降
# - train/lr: 学习率曲线
# - val/loss: 验证集损失

# 预期：
# - 训练loss: 从~2.5降到~0.5-1.0
# - 大约需要3 epochs，~5000 steps
# - 4卡A100约2-3小时
```

### 2.3 保存Baseline checkpoint

```bash
# Checkpoint保存在:
# ./checkpoints/baseline_sft/global_step_XXXX/
```

---

## Step 3: FOMAML-SFT训练

### 3.1 启动FOMAML-SFT训练

```bash
# 单GPU (不推荐，内存可能不够)
python maml_sft_trainer.py \
    --config-path data/math_science_meta \
    --config-name config_fomaml_math_science

# 多GPU (4卡，推荐)
torchrun --nproc_per_node=4 \
    maml_sft_trainer.py \
    --config-path data/math_science_meta \
    --config-name config_fomaml_math_science
```

### 3.2 FOMAML训练监控

```python
# 关键指标（wandb）:

# Meta-level指标:
# - meta/loss: 元损失，应该下降
# - meta/grad_norm: 梯度范数

# Task-specific指标:
# - {task_name}/support_loss: 内循环support loss
# - {task_name}/query_loss: 外循环query loss
# - {task_name}/adaptation_gap: query_loss - support_loss

# 健康训练的标志:
# 1. meta/loss 持续下降
# 2. adaptation_gap 逐渐减小（说明元初始化变好）
# 3. 不同任务的query_loss相对平衡
```

### 3.3 预期训练时间和资源

```
配置: 4×A100 (80GB)
模型: Llama-3.2-1B with LoRA (rank=16)
任务数: 6-8个

预期:
- 每个meta-iteration: ~30-60秒
- 5000 steps: ~42-83小时 (2-3.5天)
- 峰值内存: 每GPU ~40-50GB
- 可用梯度accumulation减少内存
```

### 3.4 故障排除

```bash
# 问题1: OOM (内存不足)
解决方案:
1. 减小meta_batch_size: 3 → 2
2. 减小inner_batch_size: 4 → 2
3. 减小num_inner_steps: 5 → 3
4. 启用gradient checkpointing
5. 使用更小的LoRA rank: 16 → 8

# 问题2: 训练不稳定 (loss震荡)
解决方案:
1. 降低inner_lr: 5e-5 → 1e-5
2. 降低outer_lr: 2e-5 → 1e-5
3. 增加梯度裁剪: clip_grad=0.5
4. 检查数据质量

# 问题3: 适应效果差 (adaptation_gap不减小)
解决方案:
1. 增加num_inner_steps: 5 → 10
2. 检查任务相关性（任务是否真的相关）
3. 增加support set大小
4. 训练更多steps
```

---

## Step 4: Few-Shot评估

### 4.1 评估Baseline SFT

```bash
python evaluate_few_shot.py \
    --model-path ./checkpoints/baseline_sft/global_step_5000 \
    --model-type sft \
    --data-dir ./data/math_science_meta/few_shot_eval \
    --eval-tasks algebra geometry number_theory word_problems \
    --n-shots 0 5 10 25 50 \
    --n-runs 5 \
    --output-dir ./results/baseline_sft \
    --adaptation-lr 1e-4 \
    --adaptation-steps 100

# 这会：
# 1. 对每个任务在不同few-shot设置下评估
# 2. 重复5次取平均（减少随机性）
# 3. 生成学习曲线图
# 4. 保存结果到 results/baseline_sft/sft_results.json
```

### 4.2 评估FOMAML-SFT

```bash
python evaluate_few_shot.py \
    --model-path ./checkpoints/fomaml_math_science/maml_checkpoint_step_5000.pt \
    --model-type fomaml \
    --data-dir ./data/math_science_meta/few_shot_eval \
    --eval-tasks algebra geometry number_theory word_problems \
    --n-shots 0 5 10 25 50 \
    --n-runs 5 \
    --output-dir ./results/fomaml_sft \
    --adaptation-lr 1e-4 \
    --adaptation-steps 100
```

### 4.3 Base Model评估（可选）

```bash
# 评估未fine-tune的base model作为下界参考
python evaluate_few_shot.py \
    --model-path meta-llama/Llama-3.2-1B \
    --model-type base \
    --data-dir ./data/math_science_meta/few_shot_eval \
    --eval-tasks algebra geometry number_theory word_problems \
    --n-shots 0 5 10 25 50 \
    --n-runs 5 \
    --output-dir ./results/base_model
```

### 4.4 评估时间估算

```
单任务单个few-shot设置单次运行：
- Zero-shot (0-shot): ~5分钟 (100 samples)
- 5-shot: ~10分钟 (adapt 100 steps + eval 100 samples)
- 10-shot: ~10分钟
- 25-shot: ~15分钟
- 50-shot: ~20分钟

总计单任务5个设置5次运行: ~4-5小时

4个任务 × 5小时 = 20小时

3个模型 × 20小时 = 60小时 (2.5天)

并行评估可大幅加速！
```

---

## Step 5: 结果分析

### 5.1 比较学习曲线

```bash
# 创建对比图
python -c "
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 加载结果
with open('results/fomaml_sft/fomaml_results.json') as f:
    fomaml_results = json.load(f)

with open('results/baseline_sft/sft_results.json') as f:
    sft_results = json.load(f)

# 为每个任务绘制对比图
for task in fomaml_results.keys():
    fig, ax = plt.subplots(figsize=(10, 6))

    # FOMAML curve
    fomaml_shots = sorted([int(k) for k in fomaml_results[task].keys()])
    fomaml_accs = [fomaml_results[task][str(s)]['accuracy_mean'] * 100 for s in fomaml_shots]

    # SFT curve
    sft_shots = sorted([int(k) for k in sft_results[task].keys()])
    sft_accs = [sft_results[task][str(s)]['accuracy_mean'] * 100 for s in sft_shots]

    ax.plot(fomaml_shots, fomaml_accs, marker='o', label='FOMAML-SFT', linewidth=2)
    ax.plot(sft_shots, sft_accs, marker='s', label='Baseline SFT', linewidth=2)

    ax.set_xlabel('Number of Few-Shot Examples', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'{task} - Few-Shot Learning Curve', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'results/comparison_{task}.png', dpi=300)
    print(f'Saved: results/comparison_{task}.png')
"
```

### 5.2 计算样本效率

```python
# 计算达到目标准确率所需的样本数
def compute_sample_efficiency(results, target_acc=0.7):
    \"\"\"计算达到目标准确率需要的样本数\"\"\"
    for n_shots in sorted(results.keys(), key=int):
        acc = results[n_shots]['accuracy_mean']
        if acc >= target_acc:
            return int(n_shots)
    return float('inf')

# 对比
target = 0.7  # 70%准确率
for task in fomaml_results.keys():
    fomaml_samples = compute_sample_efficiency(fomaml_results[task], target)
    sft_samples = compute_sample_efficiency(sft_results[task], target)

    efficiency_gain = sft_samples / fomaml_samples if fomaml_samples < float('inf') else float('inf')

    print(f"{task}:")
    print(f"  FOMAML-SFT: {fomaml_samples} samples to reach {target:.0%}")
    print(f"  Baseline SFT: {sft_samples} samples to reach {target:.0%}")
    print(f"  Efficiency gain: {efficiency_gain:.1f}x")
    print()
```

### 5.3 统计显著性检验

```python
from scipy import stats

# 对每个任务的每个few-shot设置进行配对t检验
for task in fomaml_results.keys():
    print(f"\n{task}:")

    for n_shots in fomaml_results[task].keys():
        # 假设我们保存了多次运行的所有结果
        fomaml_runs = fomaml_results[task][n_shots].get('runs', [])
        sft_runs = sft_results[task][n_shots].get('runs', [])

        if len(fomaml_runs) > 1 and len(sft_runs) > 1:
            fomaml_accs = [r['accuracy'] for r in fomaml_runs]
            sft_accs = [r['accuracy'] for r in sft_runs]

            # 配对t检验
            t_stat, p_value = stats.ttest_ind(fomaml_accs, sft_accs)

            fomaml_mean = np.mean(fomaml_accs)
            sft_mean = np.mean(sft_accs)
            diff = fomaml_mean - sft_mean

            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

            print(f"  {n_shots}-shot:")
            print(f"    FOMAML: {fomaml_mean:.2%}, SFT: {sft_mean:.2%}")
            print(f"    Diff: {diff:+.2%}, p={p_value:.4f} {sig}")
```

### 5.4 生成实验报告

```python
# 创建LaTeX表格
print("\\begin{table}[h]")
print("\\centering")
print("\\caption{Few-Shot Learning Performance Comparison}")
print("\\begin{tabular}{lcccccc}")
print("\\toprule")
print("Task & Method & 0-shot & 5-shot & 10-shot & 25-shot & 50-shot \\\\")
print("\\midrule")

for task in fomaml_results.keys():
    # FOMAML row
    fomaml_row = [task, "FOMAML-SFT"]
    for n in [0, 5, 10, 25, 50]:
        acc = fomaml_results[task][str(n)]['accuracy_mean']
        fomaml_row.append(f"{acc:.2%}")
    print(" & ".join(fomaml_row) + " \\\\")

    # SFT row
    sft_row = ["", "Baseline SFT"]
    for n in [0, 5, 10, 25, 50]:
        acc = sft_results[task][str(n)]['accuracy_mean']
        sft_row.append(f"{acc:.2%}")
    print(" & ".join(sft_row) + " \\\\")

    print("\\midrule")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")
```

---

## 预期结果

### 成功标准

✅ **主要成功标准：**

1. **样本效率提升 ≥ 3倍**
   ```
   达到70%准确率所需样本：
   FOMAML-SFT: 10 shots
   Baseline SFT: 30+ shots
   效率比: 3x+
   ```

2. **Zero-shot迁移提升 ≥ 5%**
   ```
   在未见过任务上的zero-shot准确率：
   FOMAML-SFT: 45%
   Baseline SFT: 40%
   提升: +5%
   ```

3. **统计显著性 p < 0.05**
   ```
   在多个任务和few-shot设置下一致显著
   ```

### 预期学习曲线形状

```
Accuracy (%)
   100 ┤
       │
    80 ┤                        ●───● FOMAML-SFT
       │                    ●──●
    60 ┤               ●──●          ○───○ Baseline SFT
       │          ●──●           ○──○
    40 ┤     ●──●           ○──○
       │  ●──               ○
    20 ┤●               ○
       └───┴───┴───┴───┴───┴───┴───────> N-shot
           0   5  10  25  50  100

关键特征：
1. FOMAML-SFT起点更高（better zero-shot）
2. FOMAML-SFT上升更快（better few-shot learning）
3. FOMAML-SFT在5-10 shot时已达到较高性能
```

---

## 故障排除

### 问题1: FOMAML-SFT效果不如SFT

**可能原因：**
1. 任务间相关性不够
2. 超参数不合适
3. 训练不充分

**诊断步骤：**
```bash
# 1. 检查任务相似度
python -c "
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# 计算任务间的文本相似度
tasks_data = {}
for task in ['algebra', 'geometry', 'number_theory']:
    df = pd.read_parquet(f'data/meta_train/{task}_support.parquet')
    tasks_data[task] = ' '.join(df['prompt'].tolist()[:100])

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(tasks_data.values())
similarity = cosine_similarity(vectors)

print('Task Similarity Matrix:')
print(similarity)
# 如果相似度 < 0.3，任务可能太不相关
"

# 2. 检查meta-training指标
# 查看 adaptation_gap 是否在下降
# 如果不下降，说明meta-learning没有学到好的初始化

# 3. 尝试不同超参数
# inner_lr: [1e-5, 5e-5, 1e-4]
# num_inner_steps: [3, 5, 10]
# outer_lr: [1e-5, 2e-5, 5e-5]
```

### 问题2: 评估结果方差太大

**解决方案：**
```bash
# 增加评估重复次数
--n-runs 10  # 从3增加到10

# 固定随机种子
--seed 42

# 使用更多测试样本
# 修改evaluate_few_shot.py中的test_data[:100]为test_data[:500]
```

### 问题3: 训练时间太长

**加速方案：**
```bash
# 1. 减少任务数量
# 从8个减到5-6个核心任务

# 2. 减少训练步数
# 5000 → 3000 steps（可能牺牲少许性能）

# 3. 减少内循环步数
# num_inner_steps: 5 → 3

# 4. 使用更小的模型
# Llama-3.2-1B → Qwen2.5-0.5B

# 5. 并行评估
# 在多个GPU上并行评估不同任务
```

---

## Checklist

### 数据准备
- [ ] 下载数据集 (MATH, GSM8K, ScienceQA)
- [ ] 运行 `prepare_math_science_data.py`
- [ ] 验证数据格式和统计
- [ ] 检查生成的配置文件

### Baseline训练
- [ ] 启动Baseline SFT训练
- [ ] 监控训练指标
- [ ] 保存checkpoint
- [ ] 记录训练时间和资源

### FOMAML训练
- [ ] 启动FOMAML-SFT训练
- [ ] 监控meta-training指标
- [ ] 检查adaptation_gap趋势
- [ ] 保存最佳checkpoint

### 评估
- [ ] 评估Base Model
- [ ] 评估Baseline SFT
- [ ] 评估FOMAML-SFT
- [ ] 生成学习曲线图
- [ ] 进行统计显著性检验

### 分析
- [ ] 计算样本效率
- [ ] 分析跨任务泛化
- [ ] 错误案例分析
- [ ] 生成实验报告
- [ ] 准备可视化结果

---

## 下一步

完成基础实验后，可以探索：

1. **更多任务**：添加物理、化学等科学任务
2. **更大模型**：Llama-3.2-3B, Llama-3.1-8B
3. **完整MAML**：对比FOMAML vs MAML
4. **Reptile对比**：对比三种元学习算法
5. **混合方法**：Reptile + LoRA
6. **任务课程**：从简单到困难的任务顺序
7. **多模态**：添加图像输入（geometry图形题）

---

祝实验顺利！🚀
