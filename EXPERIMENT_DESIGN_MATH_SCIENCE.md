# FOMAML-SFT实验设计：MATH & Science推理任务

## 目录
1. [实验目标](#实验目标)
2. [数据集选择与构建](#数据集选择与构建)
3. [实验设计](#实验设计)
4. [评估方案](#评估方案)
5. [预期结果](#预期结果)

---

## 实验目标

### 核心研究问题

**主要问题：FOMAML-SFT能否在数学和科学推理任务上展现出更好的few-shot学习能力？**

具体验证：
1. **适应速度**：FOMAML-SFT在新任务上需要多少样本才能达到SFT的性能？
2. **泛化能力**：在未见过的推理任务上，FOMAML-SFT的zero-shot/few-shot性能如何？
3. **数据效率**：给定相同的总训练数据量，FOMAML-SFT vs SFT谁更好？
4. **任务转移**：FOMAML-SFT学到的元知识能否迁移到不同类型的推理任务？

---

## 数据集选择与构建

### 1. Meta-Training数据集（多任务）

#### 推荐策略：**领域分解**（Domain Decomposition）

将数学推理任务按**子领域**划分，每个子领域作为一个meta-learning任务：

```
Meta-Training Tasks (6-8个任务):
├── Task 1: 代数 (Algebra)
│   ├── 线性方程
│   ├── 二次方程
│   └── 不等式
│
├── Task 2: 几何 (Geometry)
│   ├── 平面几何
│   ├── 立体几何
│   └── 解析几何
│
├── Task 3: 数论 (Number Theory)
│   ├── 质数与因数分解
│   ├── 同余
│   └── 整除性
│
├── Task 4: 组合数学 (Combinatorics)
│   ├── 排列组合
│   ├── 概率
│   └── 计数原理
│
├── Task 5: 微积分 (Calculus)
│   ├── 极限与连续
│   ├── 导数
│   └── 积分
│
├── Task 6: 统计 (Statistics)
│   ├── 描述统计
│   ├── 概率分布
│   └── 假设检验
│
├── Task 7: 物理 (Physics)
│   ├── 力学
│   ├── 电磁学
│   └── 热力学
│
└── Task 8: 化学 (Chemistry)
    ├── 化学反应
    ├── 化学计量
    └── 平衡与动力学
```

### 2. 数据来源与构建

#### 推荐数据集

| 数据集 | 规模 | 领域 | 用途 | 链接 |
|--------|------|------|------|------|
| **MATH** | 12.5k | 数学各领域 | Meta-train主力 | [MATH dataset](https://github.com/hendrycks/math) |
| **GSM8K** | 8.5k | 小学数学应用题 | Meta-train补充 | [GSM8K](https://github.com/openai/grade-school-math) |
| **ScienceQA** | 21k | 科学多模态QA | Meta-train科学部分 | [ScienceQA](https://scienceqa.github.io/) |
| **Geometry3K** | 3k | 几何证明 | 特定任务 | [Geometry3K](https://geometry3k.github.io/) |
| **AMPS** | 5k | 代数文字题 | 补充代数任务 | [AMPS](https://github.com/hendrycks/math) |
| **TheoremQA** | 800 | 大学数学定理 | Few-shot评估 | [TheoremQA](https://github.com/wenhuchen/TheoremQA) |
| **MMLU-STEM** | ~6k | STEM知识 | Zero-shot评估 | [MMLU](https://github.com/hendrycks/test) |

#### 具体数据划分方案

##### 方案A：基于MATH数据集（推荐）

```python
# MATH数据集的7个子领域，天然适合meta-learning
MATH_SUBJECTS = [
    'algebra',           # ~1500 samples
    'geometry',          # ~1300 samples
    'counting_and_probability',  # ~500 samples
    'number_theory',     # ~600 samples
    'prealgebra',        # ~1500 samples
    'precalculus',       # ~1000 samples
    'intermediate_algebra',  # ~1000 samples
]

# 划分策略
for subject in MATH_SUBJECTS:
    subject_data = load_math_subject(subject)

    # 按难度分层采样
    levels = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']

    # Support set: 20% 数据，涵盖所有难度
    support = stratified_sample(subject_data, ratio=0.2, by='level')

    # Query set: 30% 数据
    query = stratified_sample(subject_data, ratio=0.3, by='level')

    # Hold-out: 50% 数据用于最终评估
    test = remaining_data
```

**数据量建议：**
```
每个任务（以algebra为例）:
├── Support set: ~300 samples  (用于内循环适应)
├── Query set: ~450 samples    (用于元梯度计算)
└── Test set: ~750 samples     (用于评估)

总计meta-training:
- 7个任务 × 750 samples = 5250 samples (support + query)
```

##### 方案B：混合数据集（更丰富）

```python
# 构建更丰富的meta-training任务
TASKS = {
    # 数学任务 (来自MATH)
    'algebra': {
        'source': 'MATH',
        'subject': 'algebra',
        'support': 300,
        'query': 450,
    },
    'geometry': {
        'source': 'MATH',
        'subject': 'geometry',
        'support': 300,
        'query': 450,
    },
    'number_theory': {
        'source': 'MATH',
        'subject': 'number_theory',
        'support': 300,
        'query': 450,
    },

    # 应用题任务 (来自GSM8K)
    'word_problems': {
        'source': 'GSM8K',
        'support': 400,
        'query': 600,
    },

    # 科学任务 (来自ScienceQA)
    'physics': {
        'source': 'ScienceQA',
        'subject': 'physics',
        'support': 300,
        'query': 450,
    },
    'chemistry': {
        'source': 'ScienceQA',
        'subject': 'chemistry',
        'support': 300,
        'query': 450,
    },

    # 几何证明 (来自Geometry3K)
    'geometry_proof': {
        'source': 'Geometry3K',
        'support': 200,
        'query': 300,
    },
}
```

### 3. Few-Shot评估数据集

#### 评估任务选择（关键！）

选择**未在meta-training中见过**的任务：

```python
FEW_SHOT_EVAL_TASKS = {
    # In-Domain但未训练的难度/子类
    'hard_algebra': {
        'source': 'MATH',
        'subject': 'algebra',
        'level': 'Level 5',  # 只用最难的
        'n_shots': [0, 5, 10, 25, 50],
        'test_size': 200,
    },

    # 相关但不同的任务
    'calculus': {
        'source': 'MATH',
        'subject': 'precalculus',  # 如果meta-train没用
        'n_shots': [0, 5, 10, 25, 50],
        'test_size': 200,
    },

    # 新领域任务
    'theorem_proving': {
        'source': 'TheoremQA',
        'n_shots': [0, 5, 10, 25, 50],
        'test_size': 200,
    },

    # 跨领域迁移
    'mmlu_stem': {
        'source': 'MMLU',
        'subjects': ['physics', 'chemistry', 'mathematics'],
        'n_shots': [0, 5],
        'test_size': 500,
    },

    # 组合推理
    'aime': {
        'source': 'AIME',  # 美国高中数学邀请赛
        'n_shots': [0, 5, 10],
        'test_size': 100,
    },
}
```

### 4. 数据预处理

#### 统一格式

```python
# 所有数据转换为统一的SFT格式
def convert_to_sft_format(problem):
    """
    输入: 原始问题
    {
        'problem': '求解方程 2x + 3 = 7',
        'solution': 'x = 2',
        'level': 'Level 2',
        'type': 'algebra'
    }

    输出: SFT格式
    {
        'prompt': '请解决以下数学问题：\n\n问题：求解方程 2x + 3 = 7\n\n请提供详细的解题步骤和最终答案。',
        'response': '让我们一步步解决这个方程：\n\n1. 从等式两边减去3：\n   2x + 3 - 3 = 7 - 3\n   2x = 4\n\n2. 两边除以2：\n   x = 4/2 = 2\n\n因此，方程的解是 x = 2。'
    }
    """
    # 构建prompt
    prompt = f"请解决以下数学问题：\n\n问题：{problem['problem']}\n\n请提供详细的解题步骤和最终答案。"

    # 构建response（需要包含推理过程）
    response = generate_solution_with_reasoning(
        problem['problem'],
        problem['solution']
    )

    return {
        'prompt': prompt,
        'response': response,
        'metadata': {
            'level': problem.get('level'),
            'type': problem.get('type'),
            'source': problem.get('source'),
        }
    }
```

#### CoT (Chain-of-Thought) 处理

**重要：对于推理任务，response应该包含完整的推理链！**

```python
# 如果数据集只有最终答案，需要生成推理过程
def augment_with_cot(problem, solution):
    """使用GPT-4生成推理链"""
    prompt = f"""
为以下数学问题生成详细的解题步骤：

问题: {problem}
答案: {solution}

要求:
1. 逐步分解问题
2. 解释每一步的推理
3. 最后给出明确的答案

格式:
让我们一步步解决这个问题：

步骤1: ...
步骤2: ...
...

因此，答案是 {solution}。
"""
    cot_solution = call_gpt4(prompt)
    return cot_solution
```

---

## 实验设计

### Baseline设置

#### Baseline 1: 标准SFT (All-Tasks Mixed)

```yaml
# config_baseline_sft.yaml
name: "baseline_sft_all_mixed"

data:
  # 将所有meta-training任务的数据混合
  train_files:
    - "data/processed/algebra_all.parquet"
    - "data/processed/geometry_all.parquet"
    - "data/processed/number_theory_all.parquet"
    - "data/processed/combinatorics_all.parquet"
    - "data/processed/physics_all.parquet"
    - "data/processed/chemistry_all.parquet"

  train_batch_size: 32
  total_samples: 5250  # 与FOMAML-SFT相同的数据量

trainer:
  total_epochs: 3
  lr: 2e-5
```

**关键：确保与FOMAML-SFT使用相同的数据量！**

#### Baseline 2: 标准SFT (Task-Specific)

```python
# 对每个few-shot评估任务，单独训练一个SFT模型
# 这代表"有足够数据时"的性能上界

for task in FEW_SHOT_EVAL_TASKS:
    # 使用该任务的大量数据训练
    sft_model = train_sft(
        data=task.train_data,  # 完整训练集
        epochs=3,
    )
    # 这是性能上界参考
```

#### Baseline 3: Zero-Shot Base Model

```python
# 不经过任何fine-tuning的base model
# 这是性能下界参考
base_model = load_model("meta-llama/Llama-3.2-1B")
```

### FOMAML-SFT设置

```yaml
# config_fomaml_sft.yaml
name: "fomaml_sft_math_science"

model:
  partial_pretrain: "meta-llama/Llama-3.2-1B"
  lora_rank: 16  # 推理任务可能需要更大的rank
  lora_alpha: 32
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  use_fsdp: true

meta:
  use_fomaml: true

  # 内循环参数（关键超参数）
  inner_lr: 5e-5        # 数学推理可能需要更小的LR
  num_inner_steps: 5    # 开始用5，可以试3-10
  inner_batch_size: 4

  # 外循环参数
  outer_lr: 2e-5        # 略低于标准SFT
  meta_batch_size: 3    # 每次更新3个任务
  query_batch_size: 4

  # 任务定义
  tasks:
    - name: "algebra"
      support_files: ["data/meta_train/algebra_support.parquet"]
      query_files: ["data/meta_train/algebra_query.parquet"]
      support_max_samples: 300
      query_max_samples: 450

    - name: "geometry"
      support_files: ["data/meta_train/geometry_support.parquet"]
      query_files: ["data/meta_train/geometry_query.parquet"]
      support_max_samples: 300
      query_max_samples: 450

    # ... 其他任务

data:
  max_length: 2048  # 推理任务可能需要更长的上下文
  truncation: "right"

optim:
  optimizer_type: "AdamW"
  weight_decay: 0.01
  clip_grad: 1.0

trainer:
  total_steps: 5000  # 可能需要更多步数
  save_freq: 500
  test_freq: 100
```

### 对照实验矩阵

| 实验组 | 方法 | 训练数据 | 训练步数 | 超参数 |
|--------|------|----------|----------|--------|
| **Exp 1** | FOMAML-SFT | 6任务×750样本 | 5000 | inner_lr=5e-5, K=5 |
| **Exp 2** | SFT-Mixed | 所有任务混合 | 匹配样本数 | lr=2e-5 |
| **Exp 3** | FOMAML-SFT | 同Exp1 | 5000 | inner_lr=1e-4, K=3 |
| **Exp 4** | FOMAML-SFT | 同Exp1 | 5000 | inner_lr=5e-5, K=10 |
| **Exp 5** | SFT-Mixed | 2×数据量 | 2×步数 | lr=2e-5 |

---

## 评估方案

### 1. Few-Shot学习曲线

**核心评估：在新任务上，需要多少样本才能达到目标性能？**

```python
def evaluate_few_shot_learning_curve(
    model,  # FOMAML-SFT 或 SFT模型
    task_name,
    n_shots_list=[0, 5, 10, 25, 50, 100],
    test_data,
    adaptation_steps=100,  # 在few-shot样本上fine-tune的步数
):
    """
    评估few-shot学习曲线

    返回: {n_shots: accuracy}
    """
    results = {}

    for n_shots in n_shots_list:
        # 1. 采样n_shots个样本
        few_shot_data = sample_n_shots(task_name, n=n_shots)

        # 2. 在few-shot样本上fine-tune
        if n_shots == 0:
            # Zero-shot: 直接评估
            adapted_model = model
        else:
            # Few-shot: fine-tune
            adapted_model = clone_model(model)
            optimizer = AdamW(adapted_model.parameters(), lr=1e-4)

            for step in range(adaptation_steps):
                batch = sample_batch(few_shot_data)
                loss = compute_loss(adapted_model, batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        # 3. 在测试集上评估
        accuracy = evaluate_accuracy(adapted_model, test_data)
        results[n_shots] = accuracy

        print(f"{task_name} - {n_shots}-shot: {accuracy:.2%}")

    return results
```

### 2. 评估指标

#### 主要指标

```python
METRICS = {
    # 正确率相关
    'accuracy': compute_exact_match,  # 最终答案完全匹配
    'acc@1': top_1_accuracy,
    'acc@5': top_5_accuracy,

    # 推理质量
    'reasoning_steps': count_reasoning_steps,  # CoT步骤数
    'step_correctness': check_intermediate_steps,  # 中间步骤正确率

    # Few-shot学习指标
    'sample_efficiency': compute_sample_efficiency,  # 达到目标性能所需样本数
    'adaptation_speed': compute_convergence_steps,  # 收敛所需步数

    # 迁移学习
    'zero_shot_transfer': zero_shot_performance,
    'cross_domain_transfer': evaluate_on_unseen_domains,
}
```

#### 具体计算

```python
def compute_exact_match(prediction, ground_truth):
    """精确匹配：答案是否完全正确"""
    pred_answer = extract_answer(prediction)
    gt_answer = extract_answer(ground_truth)
    return normalize_answer(pred_answer) == normalize_answer(gt_answer)

def compute_sample_efficiency(learning_curve, target_acc=0.7):
    """
    样本效率：达到目标准确率需要多少样本

    Example:
    learning_curve = {0: 0.3, 5: 0.5, 10: 0.65, 25: 0.75, 50: 0.80}
    target_acc = 0.7
    return: 25 (需要25个样本达到70%准确率)
    """
    for n_shots, acc in sorted(learning_curve.items()):
        if acc >= target_acc:
            return n_shots
    return float('inf')  # 未达到目标

def compute_adaptation_gap(support_loss, query_loss):
    """适应间隙：支撑集和查询集的性能差距"""
    return abs(query_loss - support_loss)
```

### 3. 评估协议

#### 协议A: Few-Shot适应评估

```python
# 标准few-shot评估流程
for model_name in ['FOMAML-SFT', 'SFT-Mixed', 'Base-Model']:
    model = load_model(model_name)

    for task in FEW_SHOT_EVAL_TASKS:
        print(f"\n=== Evaluating {model_name} on {task.name} ===")

        # 1. Zero-shot性能
        zero_shot_acc = evaluate(model, task.test_data)
        print(f"Zero-shot: {zero_shot_acc:.2%}")

        # 2. Few-shot学习曲线
        for n_shots in [5, 10, 25, 50]:
            # 重复5次取平均（减少采样随机性）
            accs = []
            for seed in range(5):
                few_shot_data = sample_n_shots(
                    task, n=n_shots, seed=seed
                )
                adapted_model = adapt_model(
                    model, few_shot_data, steps=100
                )
                acc = evaluate(adapted_model, task.test_data)
                accs.append(acc)

            avg_acc = np.mean(accs)
            std_acc = np.std(accs)
            print(f"{n_shots}-shot: {avg_acc:.2%} ± {std_acc:.2%}")
```

#### 协议B: 跨任务泛化评估

```python
# 评估在完全未见过的任务上的表现
UNSEEN_TASKS = [
    'theorem_proving',  # 定理证明（新任务类型）
    'competition_math',  # 数学竞赛题（新难度）
    'applied_physics',   # 应用物理（新应用场景）
]

for task in UNSEEN_TASKS:
    zero_shot_scores = {
        'FOMAML-SFT': evaluate(fomaml_model, task),
        'SFT-Mixed': evaluate(sft_model, task),
        'Base': evaluate(base_model, task),
    }
    print(f"\nTask: {task}")
    print(f"  FOMAML-SFT: {zero_shot_scores['FOMAML-SFT']:.2%}")
    print(f"  SFT-Mixed:  {zero_shot_scores['SFT-Mixed']:.2%}")
    print(f"  Base:       {zero_shot_scores['Base']:.2%}")
```

### 4. 统计显著性检验

```python
from scipy import stats

def compare_methods(fomaml_results, sft_results, n_runs=5):
    """
    比较两种方法的性能差异

    使用配对t检验（paired t-test）
    """
    # 收集多次运行的结果
    fomaml_accs = []
    sft_accs = []

    for run in range(n_runs):
        fomaml_acc = evaluate_with_seed(fomaml_model, task, seed=run)
        sft_acc = evaluate_with_seed(sft_model, task, seed=run)

        fomaml_accs.append(fomaml_acc)
        sft_accs.append(sft_acc)

    # 配对t检验
    t_stat, p_value = stats.ttest_rel(fomaml_accs, sft_accs)

    print(f"FOMAML-SFT: {np.mean(fomaml_accs):.2%} ± {np.std(fomaml_accs):.2%}")
    print(f"SFT-Mixed:  {np.mean(sft_accs):.2%} ± {np.std(sft_accs):.2%}")
    print(f"Difference: {np.mean(fomaml_accs) - np.mean(sft_accs):.2%}")
    print(f"p-value: {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")

    return {
        'fomaml_mean': np.mean(fomaml_accs),
        'fomaml_std': np.std(fomaml_accs),
        'sft_mean': np.mean(sft_accs),
        'sft_std': np.std(sft_accs),
        'p_value': p_value,
        'significant': p_value < 0.05,
    }
```

---

## 预期结果

### 1. 核心假设

**H1: Few-Shot样本效率**
```
FOMAML-SFT在5-10个样本上的性能 ≈ SFT-Mixed在50-100个样本上的性能

即：FOMAML-SFT的样本效率是SFT的5-10倍
```

**H2: Zero-Shot迁移**
```
在未见过的推理任务上：
FOMAML-SFT (zero-shot) > SFT-Mixed (zero-shot) > Base Model
```

**H3: 适应速度**
```
达到70%目标准确率所需的fine-tuning步数：
FOMAML-SFT: ~20-50步
SFT-Mixed: ~100-200步
```

### 2. 预期学习曲线

```
Accuracy
    │
80% ├─────────────────● FOMAML-SFT
    │              ●─●
70% ├───────────●──────────────● SFT-Mixed
    │        ●─●           ●─●
60% ├─────●──────────────●
    │  ●─●          ●─●
50% ├●──────────●──────────────● Base Model
    │      ●───────────●
    └──────┴────┴────┴────┴────┴────> N-shot
           0    5   10   25   50  100

关键观察:
1. FOMAML-SFT的zero-shot性能更高
2. FOMAML-SFT在5-10 shot时就能达到较高性能
3. SFT-Mixed需要更多样本才能达到相同性能
```

### 3. 跨任务迁移预期

```
任务相似度 vs 迁移性能

High Transfer
    │   ● Algebra → Geometry (FOMAML)
    │  ●● Math → Physics (FOMAML)
    │ ●
    │●    ○ Algebra → Geometry (SFT)
    │   ○   Math → Physics (SFT)
Low │  ○
    └───────────────────> Task Distance
    Close          Far

FOMAML-SFT在任务迁移上应该全面优于SFT-Mixed
```

---

## 实验执行清单

### Phase 1: 数据准备（1-2天）

- [ ] 下载MATH、GSM8K、ScienceQA数据集
- [ ] 划分meta-training任务（6-8个）
- [ ] 为每个任务生成support/query split
- [ ] 准备few-shot评估任务数据
- [ ] 数据格式转换和验证
- [ ] 生成或验证CoT推理链

### Phase 2: Baseline训练（2-3天）

- [ ] 训练SFT-Mixed baseline
- [ ] 训练task-specific SFT models（可选）
- [ ] 评估base model性能
- [ ] 记录baseline指标

### Phase 3: FOMAML-SFT训练（3-5天）

- [ ] 训练FOMAML-SFT（默认超参数）
- [ ] 监控meta-training指标
- [ ] 超参数调优（K, inner_lr, outer_lr）
- [ ] 选择最佳checkpoint

### Phase 4: 评估（2-3天）

- [ ] Few-shot学习曲线评估（5次重复）
- [ ] 跨任务泛化评估
- [ ] 适应速度评估
- [ ] 统计显著性检验
- [ ] 结果可视化

### Phase 5: 分析（1-2天）

- [ ] 学习曲线对比分析
- [ ] 样本效率计算
- [ ] 错误案例分析
- [ ] 撰写实验报告

---

## 成功标准

### 核心成功指标

1. **样本效率提升 > 3倍**
   - FOMAML-SFT在10-shot上的性能 ≥ SFT-Mixed在30-shot上的性能

2. **Zero-shot迁移提升 > 5%**
   - 在未见过的任务上，FOMAML-SFT的zero-shot准确率比SFT-Mixed高5%以上

3. **统计显著性**
   - p-value < 0.05，在多个评估任务上一致显著

### 次要成功指标

4. **适应速度提升 > 2倍**
   - 达到目标性能所需的fine-tuning步数减少50%以上

5. **跨领域泛化**
   - 在科学任务（MMLU-STEM）上的性能提升

---

## 常见问题

### Q1: 为什么要用CoT格式？

**A**: 推理任务需要展示思考过程：
- 让模型学习推理模式，而不只是答案
- CoT有助于few-shot学习（GPT-3论文证明）
- Meta-learning更容易学到可迁移的推理策略

### Q2: 任务数量选多少合适？

**A**: 建议6-8个任务：
- 太少（<4）：元知识不够丰富
- 太多（>10）：训练时间长，任务间可能冲突
- 6-8个：平衡覆盖度和效率

### Q3: 如何确保公平对比？

**A**: 关键控制变量：
- ✅ 相同的总训练样本数
- ✅ 相同的base model
- ✅ 相同的数据预处理
- ✅ 相同的评估协议
- ✅ 多次运行取平均

### Q4: 如果FOMAML-SFT效果不好怎么办？

**A**: 调试步骤：
1. 检查任务是否真的相关（任务相似度分析）
2. 调整内循环步数K（试3, 5, 10）
3. 调整学习率（inner_lr和outer_lr）
4. 增加support set大小
5. 检查是否需要更长的meta-training

---

下一步：我将创建完整的数据准备和评估脚本，让你可以直接运行实验！
