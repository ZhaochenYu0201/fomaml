# META-LORA vs FOMAML 对比实验指南

## 概述

我们现在有三种元学习方法的实现：
1. **FOMAML-SFT**: 全模型参数元学习
2. **META-LORA**: 只对LoRA参数元学习
3. **Reptile-SFT**: 简化版元学习

本指南帮助你设计和执行对比实验。

---

## 核心区别总结

| 特性 | FOMAML-SFT | META-LORA | 标准LoRA |
|------|------------|-----------|----------|
| **优化对象** | 全模型参数 | 只有LoRA参数 | 只有LoRA参数 |
| **Base model** | 需要更新 | **完全冻结** | **完全冻结** |
| **元学习** | ✅ 两循环 | ✅ 两阶段 | ❌ 单任务 |
| **参数量** | 100% | ~0.1-1% | ~0.1-1% |
| **内存占用** | 高 | **极低** | **极低** |
| **训练速度** | 慢 | **快** | **快** |
| **数据需求** | 300/任务 | **100/任务** | 数千/任务 |
| **Few-shot性能** | 优秀 | 优秀（预期） | 一般 |

### 关键洞察

**META-LORA = FOMAML + LoRA-only optimization**

```
META-LORA的优势:
✅ 保留FOMAML的元学习能力
✅ 只优化LoRA参数，计算效率高10-100倍
✅ Base model冻结，避免灾难性遗忘
✅ checkpoint极小（只有LoRA参数）

潜在劣势:
⚠️ LoRA容量有限，可能无法学习复杂模式
⚠️ 依赖base model质量
```

---

## 实验设计

### 实验目标

验证META-LORA是否能：
1. **保持FOMAML的few-shot性能**（性能不降低）
2. **大幅提升计算效率**（训练速度快10-100倍）
3. **降低数据需求**（100样本 vs 300样本）
4. **降低内存占用**（可在消费级GPU上训练）

### 实验矩阵

| 实验 | 方法 | 每任务样本数 | 预期特点 |
|------|------|--------------|----------|
| **Exp 1** | FOMAML-SFT | 300 | 性能基准 |
| **Exp 2** | META-LORA | 100 | **核心对比** |
| **Exp 3** | META-LORA | 300 | 更多数据是否提升 |
| **Exp 4** | Std LoRA (单任务) | 100 | 不用元学习的baseline |
| **Exp 5** | Std LoRA (混合) | 600 (所有任务) | 全数据baseline |

### 评估指标

#### 1. Few-Shot学习性能

```python
评估任务: {未见过的任务}
N-shots: [0, 5, 10, 25, 50]

核心问题:
- META-LORA (100样本/任务) vs FOMAML (300样本/任务)
- 性能差距多大？
- 如果相近或更好，则META-LORA胜出（更高效）
```

#### 2. 计算效率

```python
指标:
1. 训练时间 (小时)
2. 峰值GPU内存 (GB)
3. 每步耗时 (秒/step)
4. Checkpoint大小 (MB)

预期META-LORA优势:
- 训练时间: 1/10 - 1/100
- GPU内存: 1/2 - 1/5
- Checkpoint: 1/100 (只有LoRA参数)
```

#### 3. 数据效率

```python
实验:
- 固定计算预算
- 变化每任务样本数: [50, 100, 200, 300]

核心问题:
- META-LORA在100样本时 vs FOMAML在300样本时
- 谁的性能更好？
```

---

## 运行实验

### Step 1: 准备数据

```bash
# 使用相同的数据准备脚本
python prepare_math_science_data.py \
    --output-dir ./data/math_science_meta \
    --support-ratio 0.15 \  # 调整比例以获得100样本
    --query-ratio 0.25

# 验证每个任务的样本数
python -c "
import pandas as pd
from pathlib import Path

for file in Path('./data/math_science_meta/meta_train').glob('*_support.parquet'):
    df = pd.read_parquet(file)
    print(f'{file.name}: {len(df)} samples')
"
```

### Step 2a: 训练FOMAML-SFT (Baseline)

```bash
# 使用现有的FOMAML训练器
torchrun --nproc_per_node=4 \
    maml_sft_trainer.py \
    --config-path data/math_science_meta \
    --config-name config_fomaml_math_science

# 预期: 4×A100, ~40-60小时
```

### Step 2b: 训练META-LORA

```bash
# 使用新的META-LORA训练器
torchrun --nproc_per_node=4 \
    meta_lora_trainer.py \
    --config-path data/math_science_meta \
    --config-name config_meta_lora_example

# 预期: 4×A100, ~4-6小时 (快10倍！)
```

### Step 2c: 训练标准LoRA (Baseline)

```bash
# 单任务LoRA
for task in algebra geometry number_theory; do
    python verl/verl/trainer/sft_trainer.py \
        --config-path data/meta_train/${task}_train.yaml
done

# 混合任务LoRA
python verl/verl/trainer/sft_trainer.py \
    --config-path data/baseline_lora_mixed.yaml
```

### Step 3: Few-Shot评估

```bash
# 评估FOMAML
python evaluate_few_shot.py \
    --model-path ./checkpoints/fomaml_math_science/step_5000 \
    --model-type fomaml \
    --eval-tasks calculus theorem_proving \
    --n-shots 0 5 10 25 50 \
    --output-dir ./results/fomaml

# 评估META-LORA
python evaluate_few_shot.py \
    --model-path ./checkpoints/meta_lora/meta_lora_checkpoint_step_3000.pt \
    --model-type meta_lora \
    --eval-tasks calculus theorem_proving \
    --n-shots 0 5 10 25 50 \
    --output-dir ./results/meta_lora

# 评估标准LoRA
python evaluate_few_shot.py \
    --model-path ./checkpoints/std_lora_mixed \
    --model-type lora \
    --eval-tasks calculus theorem_proving \
    --n-shots 0 5 10 25 50 \
    --output-dir ./results/std_lora
```

### Step 4: 对比分析

```python
import json
import matplotlib.pyplot as plt
import numpy as np

# 加载结果
with open('results/fomaml/fomaml_results.json') as f:
    fomaml_results = json.load(f)

with open('results/meta_lora/meta_lora_results.json') as f:
    meta_lora_results = json.load(f)

with open('results/std_lora/lora_results.json') as f:
    std_lora_results = json.load(f)

# 对比图
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Few-shot学习曲线
ax = axes[0]
for method, results in [('FOMAML', fomaml_results),
                        ('META-LORA', meta_lora_results),
                        ('Std LoRA', std_lora_results)]:
    task = 'calculus'  # 示例任务
    shots = sorted([int(k) for k in results[task].keys()])
    accs = [results[task][str(s)]['accuracy_mean'] * 100 for s in shots]
    ax.plot(shots, accs, marker='o', label=method, linewidth=2)

ax.set_xlabel('Number of Shots')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Few-Shot Learning Curve')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 计算效率对比
ax = axes[1]
methods = ['FOMAML', 'META-LORA', 'Std LoRA']
train_times = [50, 5, 8]  # 示例数据（小时）
gpu_memory = [70, 30, 25]  # 示例数据（GB）

x = np.arange(len(methods))
width = 0.35

ax.bar(x - width/2, train_times, width, label='Train Time (h)', alpha=0.8)
ax.bar(x + width/2, gpu_memory, width, label='GPU Memory (GB)', alpha=0.8)

ax.set_ylabel('Value')
ax.set_title('Computational Efficiency')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()

# 3. 性能vs效率scatter plot
ax = axes[2]

# 计算平均few-shot准确率
methods_data = []
for method, results in [('FOMAML', fomaml_results),
                        ('META-LORA', meta_lora_results),
                        ('Std LoRA', std_lora_results)]:
    avg_acc = np.mean([results[task]['10']['accuracy_mean'] for task in results.keys()])
    methods_data.append((method, avg_acc, train_times[methods.index(method)]))

for method, acc, time in methods_data:
    ax.scatter(time, acc * 100, s=200, alpha=0.6, label=method)
    ax.annotate(method, (time, acc * 100), fontsize=12, ha='center')

ax.set_xlabel('Training Time (hours)')
ax.set_ylabel('Average 10-shot Accuracy (%)')
ax.set_title('Performance vs Efficiency Trade-off')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/comparison_meta_lora_vs_fomaml.png', dpi=300)
print('Saved comparison plot')
```

---

## 预期结果

### Scenario 1: META-LORA成功 ✅

```
Few-Shot性能:
- META-LORA (100样本) ≈ FOMAML (300样本)
- 差距 < 3%

计算效率:
- 训练时间: META-LORA 1/10 FOMAML
- GPU内存: META-LORA 1/3 FOMAML

结论: META-LORA是更优选择！
```

### Scenario 2: META-LORA部分成功 ⚠️

```
Few-Shot性能:
- META-LORA (100样本) < FOMAML (300样本)
- 但 META-LORA (300样本) ≈ FOMAML (300样本)

计算效率:
- 仍然快得多

结论: META-LORA需要稍多数据，但仍有价值
```

### Scenario 3: META-LORA失败 ❌

```
Few-Shot性能:
- META-LORA << FOMAML (大幅差距 >10%)

可能原因:
1. LoRA容量不足
2. 任务复杂度太高
3. 超参数需要调整

解决方案:
- 增加LoRA rank
- 调整inner/meta learning rates
- 增加inner steps
```

---

## 消融实验

### 1. LoRA Rank的影响

```yaml
实验:
  - META-LORA r=4
  - META-LORA r=8
  - META-LORA r=16 (default)
  - META-LORA r=32

问题: rank多大才足够？

预期:
- r=4: 可能太小，性能差
- r=8-16: sweet spot
- r=32: 过大，过拟合风险
```

### 2. 内循环步数的影响

```yaml
实验:
  - num_inner_steps = 3
  - num_inner_steps = 5
  - num_inner_steps = 10 (default)
  - num_inner_steps = 20

问题: 多少步适应最优？

预期:
- 太少(<5): 适应不充分
- 适中(5-10): best
- 太多(>20): 过拟合support set
```

### 3. 样本数量的影响

```yaml
实验:
  - 50 samples/task
  - 100 samples/task (META-LORA论文设定)
  - 200 samples/task
  - 300 samples/task (FOMAML设定)

问题: 100样本是否足够？

预期:
- 50: too few
- 100: 论文声称足够
- 200-300: 略有提升但边际效益递减
```

### 4. 两阶段vs单阶段

```yaml
实验:
  - Only Stage 1 (无共享LoRA更新)
  - Full two-stage (完整META-LORA)

问题: Stage 2的价值？

预期:
- 只Stage 1: 类似独立LoRA，性能差
- 两阶段: 显著提升
```

---

## 实现验证清单

### META-LORA关键特性

- [ ] Base model完全冻结
- [ ] 只优化LoRA参数
- [ ] Stage 1: 任务特定适应（K步）
- [ ] Stage 2: 梯度聚合更新shared LoRA
- [ ] 每任务只用100样本
- [ ] Checkpoint只包含LoRA参数（很小）

### 训练监控

```python
关键指标:
1. Stage 1 loss: 应该快速下降
2. Stage 2 val loss: 应该下降
3. 各任务的适应gap: 应该减小
4. GPU内存: 应该显著低于FOMAML
5. 每步时间: 应该显著快于FOMAML
```

### 评估验证

```python
验证点:
1. Few-shot性能接近或优于FOMAML
2. Zero-shot性能显著优于标准LoRA
3. 在新任务上快速适应（<20 steps）
4. checkpoint可以直接加载到新的base model
```

---

## 实验时间表

| 阶段 | 任务 | 时间（META-LORA） | 时间（FOMAML） |
|------|------|-------------------|----------------|
| 数据准备 | 数据下载和处理 | 1天 | 1天 |
| 训练 | 元学习训练 | **0.5天** | **3-5天** |
| 评估 | Few-shot评估 | 1天 | 1天 |
| 分析 | 结果对比 | 0.5天 | 0.5天 |
| **总计** | | **3天** | **6-8天** |

META-LORA最大优势：**实验迭代快速**

---

## 论文复现清单

基于META-LORA论文声称的结果：

- [ ] 100样本/任务达到全数据LoRA的性能
- [ ] 多任务学习场景下优于标准LoRA
- [ ] 多语言学习场景下有效
- [ ] 计算效率显著高于MAML
- [ ] 在新任务上快速适应

如果我们的实现达到以上效果，则成功复现！

---

## 后续探索

如果META-LORA成功，可以探索：

1. **更大模型**: Llama-3.1-8B, 70B
2. **更多任务**: 10-20个任务
3. **跨模态**: 图文多模态任务
4. **任务组合**: 不同粒度的任务划分
5. **与其他PEFT结合**: IA³, Prefix-tuning等
6. **理论分析**: 为什么LoRA足够学到元知识

---

## 总结

META-LORA的核心价值：

**在保持FOMAML优秀的few-shot性能的同时，将计算成本降低10-100倍**

如果实验验证成功，这将是LLM元学习的重大突破！
