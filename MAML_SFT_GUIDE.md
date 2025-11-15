# MAML-SFT Implementation Guide for LLMs

基于verl框架的大语言模型元学习监督微调实现指南

## 目录
1. [概述](#概述)
2. [理论背景](#理论背景)
3. [实现细节](#实现细节)
4. [使用方法](#使用方法)
5. [优化技巧](#优化技巧)
6. [实验建议](#实验建议)

---

## 概述

### 什么是MAML-SFT？

MAML-SFT将Model-Agnostic Meta-Learning (MAML)应用于大语言模型的监督微调(SFT)过程，使模型能够：

- **快速适应新任务**：在新领域只需少量样本即可fine-tune
- **跨领域泛化**：学习到通用的语言理解和生成能力
- **个性化**：为不同用户/场景快速定制模型

### 核心区别：传统SFT vs MAML-SFT

| 特性 | 传统SFT | MAML-SFT |
|------|---------|----------|
| 训练目标 | 在特定任务上最小化损失 | 学习易于适应的初始化 |
| 数据需求 | 单一任务大量数据 | 多任务少量数据 |
| 泛化能力 | 单任务性能好 | 跨任务适应快 |
| 训练复杂度 | O(N) | O(N × K) (K为内循环步数) |

---

## 理论背景

### MAML算法流程

```
初始化：元参数 θ

对于每个meta-iteration:
    1. 采样任务批次 {T₁, T₂, ..., Tₘ}

    2. 对每个任务 Tᵢ:
        a. 内循环（任务适应）：
           从 support set Dᵢˢᵘᵖᵖᵒʳᵗ 采样数据
           执行 K 步梯度下降：
           θ'ᵢ = θ - α∇θ L_Tᵢ(f_θ; Dᵢˢᵘᵖᵖᵒʳᵗ)

        b. 外循环（元学习）：
           从 query set Dᵢqᵘᵉʳʸ 采样数据
           计算元损失：
           meta_lossᵢ = L_Tᵢ(f_θ'ᵢ; Dᵢqᵘᵉʳʸ)

    3. 元参数更新：
       θ = θ - β∇θ Σᵢ meta_lossᵢ
```

### FOMAML简化

FOMAML忽略二阶梯度，计算效率更高：

```python
# MAML (二阶)
meta_grad = grad(meta_loss, θ)  # 需要计算 d(θ')/d(θ)

# FOMAML (一阶)
meta_grad = grad(meta_loss, θ')  # 直接使用一阶梯度
```

对于LLM等大模型，**推荐使用FOMAML**：
- 内存开销小（不需要存储完整计算图）
- 速度快（避免二阶梯度计算）
- 性能接近完整MAML

---

## 实现细节

### 1. verl SFT实现分析

#### 核心组件

```
verl/trainer/sft_trainer.py (标准训练器)
├── _build_engine()          # 构建训练引擎
├── _build_dataset()         # 构建数据集
├── _build_dataloader()      # 构建数据加载器
└── fit()                    # 训练循环

verl/trainer/fsdp_sft_trainer.py (FSDP训练器)
├── _build_model_optimizer() # 构建模型和优化器
├── training_step()          # 单步训练
├── _compute_loss_and_backward() # 损失计算
└── fit()                    # 训练循环

verl/workers/roles/utils/losses.py
└── sft_loss()              # SFT损失函数
```

#### 损失计算机制

```python
# verl的SFT损失计算 (losses.py:27-53)
def sft_loss(config, model_output, data, dp_group=None):
    log_prob = model_output["log_probs"]
    loss_mask = data["loss_mask"]  # 只对response计算loss

    # 关键：masked sum，只计算有效token
    loss = -masked_sum(log_prob, loss_mask) / batch_num_tokens

    return loss, {"loss": loss.detach().item()}
```

**重要特性：**
1. **Masked Loss**: 只对response部分计算损失，prompt被mask
2. **Token Normalization**: 损失除以有效token数
3. **数据并行**: 自动处理分布式训练

#### 数据格式

```python
# sft_dataset.py返回的数据格式
{
    'input_ids': torch.Tensor,      # [seq_len]
    'attention_mask': torch.Tensor, # [seq_len]
    'position_ids': torch.Tensor,   # [seq_len]
    'loss_mask': torch.Tensor,      # [seq_len], prompt部分为0
}
```

### 2. MAML-SFT关键实现

#### 双循环结构

```python
class MAMLSFTTrainer:
    def _meta_update_step(self, task_batch):
        meta_loss = 0.0

        # 保存原始参数
        original_params = clone_params(self.model)

        for task in task_batch:
            # === 内循环：任务适应 ===
            support_batch = sample_support(task)

            # K步梯度下降
            for k in range(self.num_inner_steps):
                loss = self._compute_sft_loss(support_batch)
                grads = compute_gradients(loss)
                update_params(grads, lr=self.inner_lr)

            # === 外循环：元损失 ===
            query_batch = sample_query(task)
            query_loss = self._compute_sft_loss(query_batch)
            meta_loss += query_loss

            # 恢复原始参数
            restore_params(original_params)

        # 元梯度更新
        meta_loss.backward()
        self.meta_optimizer.step()
```

#### 梯度计算细节

```python
def _inner_loop_update(self, support_batch):
    # 克隆当前参数
    fast_weights = {n: p.clone() for n, p in self.model.named_parameters()}

    for step in range(self.num_inner_steps):
        loss = self._compute_sft_loss(support_batch, self.model)

        # 计算梯度
        grads = torch.autograd.grad(
            loss,
            self.model.parameters(),
            create_graph=not self.use_fomaml,  # FOMAML不需要计算图
            retain_graph=True,
        )

        # 更新fast weights
        for (name, param), grad in zip(self.model.named_parameters(), grads):
            fast_weights[name] = fast_weights[name] - self.inner_lr * grad

        # 应用fast weights到模型
        load_params(self.model, fast_weights)

    return fast_weights
```

---

## 使用方法

### Step 1: 准备数据

每个任务需要两个数据集：
- **Support Set**: 用于内循环任务适应（少量样本，如100-500条）
- **Query Set**: 用于外循环元学习（评估样本，如500-1000条）

#### 数据格式示例

```json
// medical_support.jsonl
{"prompt": "患者主诉头痛，如何诊断？", "response": "需要询问..."}
{"prompt": "高血压的治疗方案有哪些？", "response": "主要包括..."}

// medical_query.jsonl
{"prompt": "糖尿病患者的饮食建议", "response": "应该注意..."}
```

#### 使用数据准备脚本

```bash
# 创建任务配置文件
cat > task_config.json << EOF
{
  "medical": {
    "input_file": "raw_data/medical.parquet",
    "support_ratio": 0.2,
    "max_samples": 5000
  },
  "legal": {
    "input_file": "raw_data/legal.parquet",
    "support_ratio": 0.2,
    "max_samples": 5000
  },
  "coding": {
    "input_file": "raw_data/coding.parquet",
    "support_ratio": 0.2,
    "max_samples": 5000
  }
}
EOF

# 运行数据准备
python prepare_maml_data.py \
    --config task_config.json \
    --output-dir ./data/maml \
    --balance \
    --support-size 500 \
    --query-size 1000 \
    --verify
```

### Step 2: 配置训练参数

编辑 `config_maml_sft_example.yaml`:

```yaml
meta:
  use_fomaml: true  # 推荐用FOMAML

  # 内循环参数
  inner_lr: 1e-4     # 任务适应学习率
  num_inner_steps: 5 # 适应步数（K）
  inner_batch_size: 4

  # 外循环参数
  outer_lr: 3e-5     # 元学习率
  meta_batch_size: 4 # 每次meta-update使用几个任务
  query_batch_size: 4

  # 任务定义
  tasks:
    - name: "medical"
      support_files: ["data/maml/medical/support.parquet"]
      query_files: ["data/maml/medical/query.parquet"]
    # ... 更多任务
```

### Step 3: 启动训练

```bash
# 单卡训练
python maml_sft_trainer.py

# 多卡训练
torchrun --nproc_per_node=4 maml_sft_trainer.py

# 使用自定义配置
python maml_sft_trainer.py --config-name my_maml_config
```

### Step 4: 评估和部署

训练完成后，可以快速适应到新任务：

```python
# 加载元学习的模型
model = load_checkpoint("checkpoints/maml_sft/step_10000.pt")

# 在新任务的少量样本上fine-tune
new_task_data = load_data("new_task/support.parquet")  # 只需10-50条
optimizer = AdamW(model.parameters(), lr=1e-4)

for epoch in range(5):  # 只需几步
    for batch in new_task_data:
        loss = compute_loss(model, batch)
        loss.backward()
        optimizer.step()

# 现在模型已经适应了新任务！
```

---

## 优化技巧

### 1. 内存优化

MAML需要存储多个任务的梯度，内存开销大：

```python
# 技巧1: 使用FOMAML而不是MAML
use_fomaml: true  # 节省50%+ 内存

# 技巧2: 减少meta_batch_size
meta_batch_size: 2  # 从4降到2

# 技巧3: 使用梯度累积
accumulation_steps: 2

# 技巧4: 使用LoRA
model:
  lora_rank: 8
  target_modules: ["q_proj", "v_proj"]
```

### 2. 速度优化

```python
# 技巧1: 减少内循环步数
num_inner_steps: 3  # 从5降到3，通常影响不大

# 技巧2: 使用更小的内循环batch size
inner_batch_size: 2  # 每步更快

# 技巧3: 使用Flash Attention
model:
  attn_implementation: "flash_attention_2"

# 技巧4: 启用梯度检查点
model:
  enable_gradient_checkpointing: true
```

### 3. 性能优化

```python
# 技巧1: 调整学习率比例
# 内循环学习率应该比外循环大
inner_lr: 1e-4
outer_lr: 3e-5  # inner_lr / outer_lr ≈ 3-5

# 技巧2: 使用预训练初始化
# 从已经SFT过的模型开始，而不是base model
model:
  partial_pretrain: "path/to/sft_checkpoint"

# 技巧3: 平衡任务难度
# 确保各任务的support set大小和难度相近

# 技巧4: 任务采样策略
# 可以根据任务损失动态调整采样概率
```

### 4. 数据优化

```python
# 技巧1: Support set质量 > 数量
# 精心挑选有代表性的样本，100条高质量 > 1000条低质量

# 技巧2: Query set多样性
# Query set应该涵盖任务的各个方面

# 技巧3: 任务相关性
# 选择相关但不重叠的任务（如医疗各子领域）

# 技巧4: 数据增强
# 可以对support set做paraphrase等增强
```

---

## 实验建议

### 基准实验设计

```python
实验1：验证MAML有效性
├── Baseline: 标准SFT在所有任务上训练
├── MAML-SFT: 元学习训练
└── 评估: 在新任务上few-shot适应性能

实验2：MAML vs FOMAML
├── MAML (二阶)
├── FOMAML (一阶)
└── 对比: 性能、速度、内存

实验3：超参数敏感性
├── inner_lr: [1e-5, 1e-4, 1e-3]
├── num_inner_steps: [1, 3, 5, 10]
└── meta_batch_size: [2, 4, 8]
```

### 评估指标

1. **Few-shot适应速度**
   - 在新任务的N-shot (N=10, 50, 100)上fine-tune
   - 记录达到目标性能所需的步数

2. **跨任务泛化**
   - 在未见过的任务上zero-shot性能
   - 与base model对比提升

3. **训练效率**
   - 收敛速度
   - 内存占用
   - 训练时间

### 建议的任务组合

#### 方案1：多领域通用
```yaml
tasks:
  - medical        # 医疗对话
  - legal          # 法律咨询
  - coding         # 代码生成
  - math           # 数学问题
  - creative       # 创意写作
```

#### 方案2：垂直领域深化
```yaml
tasks:
  - diagnosis      # 疾病诊断
  - treatment      # 治疗方案
  - medication     # 用药指导
  - nutrition      # 营养建议
  - mental_health  # 心理健康
```

#### 方案3：能力分解
```yaml
tasks:
  - reasoning      # 推理能力
  - summarization  # 总结能力
  - translation    # 翻译能力
  - qa             # 问答能力
  - instruction    # 指令遵循
```

---

## 进阶话题

### 1. Reptile作为替代方案

Reptile是MAML的简化版本，更容易实现：

```python
# Reptile伪代码
for epoch in range(num_epochs):
    for task in tasks:
        # 克隆当前参数
        old_params = clone(model.params)

        # 在任务上训练K步
        for k in range(K):
            batch = sample_batch(task)
            loss = compute_loss(batch)
            optimizer.step()

        # 向任务参数方向移动
        model.params = old_params + epsilon * (model.params - old_params)
```

优点：
- 实现简单，不需要双循环
- 内存效率高
- 性能接近MAML

### 2. 与LoRA结合

只对LoRA参数做MAML，固定backbone：

```python
# 只更新LoRA参数
fast_weights = {
    name: param.clone()
    for name, param in model.named_parameters()
    if 'lora' in name
}
```

优点：
- 极大降低内存和计算开销
- 适配速度更快
- 保持base model稳定性

### 3. 任务聚类

将相似任务聚类，每个cluster独立训练：

```python
clusters = {
    'medical': ['diagnosis', 'treatment', 'medication'],
    'technical': ['coding', 'debugging', 'documentation'],
}

# 每个cluster独立训练
for cluster_name, cluster_tasks in clusters.items():
    train_maml(tasks=cluster_tasks, ...)
```

---

## 故障排除

### 问题1: 内存不足

```
RuntimeError: CUDA out of memory
```

**解决方案：**
- 使用FOMAML: `use_fomaml: true`
- 减小batch size和meta_batch_size
- 启用梯度检查点
- 使用LoRA适配器

### 问题2: 训练不稳定

```
Loss出现NaN或震荡
```

**解决方案：**
- 降低inner_lr和outer_lr
- 增加梯度裁剪: `clip_grad: 0.5`
- 检查数据质量
- 使用warmup

### 问题3: 适应效果差

```
Few-shot性能不如预期
```

**解决方案：**
- 增加num_inner_steps
- 提高support set质量
- 检查任务相关性
- 增加训练步数

---

## 参考资源

### 论文
1. **MAML原论文**: "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks" (Finn et al., ICML 2017)
2. **Reptile**: "On First-Order Meta-Learning Algorithms" (Nichol et al., 2018)
3. **LLM Meta-Learning**: "Language Models are Few-Shot Learners" (Brown et al., NeurIPS 2020)

### 代码参考
- verl框架: https://github.com/volcengine/verl
- learn2learn: https://github.com/learnables/learn2learn (MAML实现参考)
- higher: https://github.com/facebookresearch/higher (二阶优化库)

---

## 总结

### 何时使用MAML-SFT？

**适合场景：**
- 需要快速适应多个领域/任务
- 有多个相关任务的数据
- 希望提升few-shot学习能力
- 需要个性化模型

**不适合场景：**
- 只有单一任务
- 数据量充足
- 对训练效率要求极高
- 任务之间完全无关

### 关键要点

1. **FOMAML优先**: 对LLM来说，FOMAML性能接近MAML但效率高得多
2. **数据质量**: Support set的质量比数量更重要
3. **超参数**: inner_lr通常是outer_lr的3-5倍
4. **任务选择**: 选择相关但不重叠的任务
5. **评估**: 关注few-shot适应速度，而不只是最终性能

祝实验顺利！🚀
