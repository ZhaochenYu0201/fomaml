# Meta-Learning for LLM SFT

基于verl框架的大语言模型元学习监督微调实现

## 📋 项目概述

本项目提供了将Meta-Learning技术应用于大语言模型SFT的完整实现，包括：

- **MAML-SFT**: Model-Agnostic Meta-Learning的LLM适配版本（全参数优化）
- **FOMAML-SFT**: 一阶MAML近似，更高效的实现（全参数优化）
- **Reptile-SFT**: 更简单的元学习算法，易于实现和使用（全参数优化）
- **META-LORA**: 参数高效的元学习方法（只优化LoRA参数，10-100倍速度提升）

基于verl强化学习框架的SFT实现，支持FSDP等高效训练技术。

⚠️ **重要说明**：FOMAML/MAML/Reptile实现是**全参数优化**，不使用LoRA。如果需要参数高效的元学习，请使用**META-LORA**实现（`meta_lora_trainer.py`）。

## 🎯 核心功能

### 为什么需要Meta-Learning SFT？

传统SFT的局限：
- 需要大量特定任务数据
- 难以快速适应新领域
- 跨任务泛化能力弱

Meta-Learning SFT的优势：
- ✅ **快速适应**: 在新任务上只需10-50条样本即可fine-tune
- ✅ **跨领域泛化**: 学习通用的学习能力
- ✅ **个性化**: 快速为不同用户/场景定制模型
- ✅ **数据效率**: 多任务共享知识，降低每个任务的数据需求

## 📁 项目结构

```
meta_learning/
├── README.md                          # 本文件
├── MAML_SFT_GUIDE.md                 # 详细实现指南
├── maml_sft_trainer.py               # MAML/FOMAML训练器（全参数）
├── meta_lora_trainer.py              # META-LORA训练器（参数高效）⭐
├── reptile_sft_trainer.py            # Reptile训练器（简化版）
├── prepare_maml_data.py              # 数据准备脚本
├── config_maml_sft_example.yaml      # FOMAML配置示例
├── config_meta_lora_example.yaml     # META-LORA配置示例⭐
├── FOMAML_FULL_PARAM_VS_LORA.md     # 全参数 vs LoRA对比说明
├── META_LORA_VS_FOMAML_COMPARISON.md # 详细对比实验指南
└── verl/                             # verl框架源码
    ├── trainer/
    │   ├── sft_trainer.py        # verl标准SFT训练器
    │   └── fsdp_sft_trainer.py   # verl FSDP SFT训练器
    ├── utils/
    │   └── dataset/
    │       └── sft_dataset.py    # SFT数据集
    └── workers/
        └── roles/utils/
            └── losses.py         # 损失函数
```

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆verl框架（如果还没有）
git clone https://github.com/volcengine/verl.git

# 安装依赖
pip install torch transformers pandas pyarrow
pip install -e verl/

# 安装其他依赖
pip install hydra-core omegaconf tensordict wandb
```

### 2. 准备数据

#### 方式A: 使用数据准备脚本

```bash
# 创建任务配置
cat > task_config.json << EOF
{
  "medical": {
    "input_file": "data/medical.parquet",
    "support_ratio": 0.2,
    "max_samples": 5000
  },
  "legal": {
    "input_file": "data/legal.parquet",
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
    --query-size 1000
```

#### 方式B: 手动组织数据

确保每个任务有以下结构：
```
data/maml/
├── medical/
│   ├── support.parquet  # 用于内循环适应
│   └── query.parquet    # 用于元损失计算
├── legal/
│   ├── support.parquet
│   └── query.parquet
└── ...
```

数据格式（parquet文件包含以下列）：
```python
{
    "prompt": "患者主诉头痛，如何诊断？",
    "response": "需要询问病史、体格检查..."
}
```

### 3. 配置训练参数

编辑 `config_maml_sft_example.yaml`:

```yaml
model:
  partial_pretrain: "meta-llama/Llama-3.2-1B"
  use_fsdp: true  # 推荐使用FSDP处理大模型
  enable_gradient_checkpointing: true  # 降低内存

meta:
  use_fomaml: true  # 推荐FOMAML（相比MAML节省50%内存和时间）
  inner_lr: 1e-4
  num_inner_steps: 5
  outer_lr: 3e-5
  meta_batch_size: 4

  tasks:
    - name: "medical"
      support_files: ["data/maml/medical/support.parquet"]
      query_files: ["data/maml/medical/query.parquet"]
    - name: "legal"
      support_files: ["data/maml/legal/support.parquet"]
      query_files: ["data/maml/legal/query.parquet"]

# 注意：这是全参数FOMAML配置
# 如需参数高效版本，请使用META-LORA（config_meta_lora_example.yaml）
```

### 4. 启动训练

#### MAML-SFT训练

```bash
# 单卡
python maml_sft_trainer.py

# 多卡
torchrun --nproc_per_node=4 maml_sft_trainer.py
```

#### Reptile-SFT训练（推荐新手）

```bash
# Reptile更简单，内存占用更小
python reptile_sft_trainer.py
```

#### META-LORA训练（推荐资源受限场景）⭐

```bash
# META-LORA: 参数高效 + 快速训练
# 只需30GB内存，训练时间仅4-6小时（vs FOMAML的40-60小时）

# 单卡即可运行！
python meta_lora_trainer.py --config-name config_meta_lora_example

# 多卡更快
torchrun --nproc_per_node=4 meta_lora_trainer.py --config-name config_meta_lora_example
```

**META-LORA优势：**
- ✅ 只优化LoRA参数（0.1-1%），base model完全冻结
- ✅ 训练速度快10-100倍
- ✅ 只需100样本/任务（vs FOMAML的300样本）
- ✅ Checkpoint超小（~10MB vs 2-5GB）
- ✅ 单卡A100即可训练

详见：[RUN_META_LORA.md](RUN_META_LORA.md) 和 [META_LORA_VS_FOMAML_COMPARISON.md](META_LORA_VS_FOMAML_COMPARISON.md)

### 5. 使用元学习的模型

训练完成后，可以快速适应新任务：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载元学习的模型
model = AutoModelForCausalLM.from_pretrained("checkpoints/maml_sft/step_10000")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# 在新任务上快速fine-tune（只需少量样本！）
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=1e-4)
new_task_loader = create_dataloader(new_task_data)  # 只需10-50条数据

# 只需3-5个epoch
for epoch in range(5):
    for batch in new_task_loader:
        loss = compute_loss(model, batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# 模型已经适应新任务！
```

## 📊 算法对比

| 算法 | 优化参数 | 复杂度 | 内存占用 | 训练时间 | 性能 | 推荐场景 |
|------|----------|--------|----------|----------|------|----------|
| **MAML** | 100% | 高 | 高 (70GB+) | 慢 (60h+) | 最佳 | 性能优先，资源充足 |
| **FOMAML** | 100% | 中 | 中 (70GB+) | 中 (40-60h) | 很好 | 全参数元学习，4×A100 |
| **Reptile** | 100% | 低 | 中 (50GB+) | 快 (20-30h) | 好 | 快速实验，简单实现 |
| **META-LORA** | 0.1-1% | 低 | 低 (30GB) | **很快 (4-6h)** | 很好 | **资源受限，快速迭代** |

⚠️ **注意**：MAML/FOMAML/Reptile都是**全参数优化**。如果资源有限或需要快速实验，强烈推荐使用**META-LORA**。

### 算法原理简述

#### MAML (Model-Agnostic Meta-Learning)
```python
# 双循环结构
for tasks in meta_batches:
    for task in tasks:
        # 内循环: 在support set上适应
        θ' = θ - α∇L_support(θ)

        # 外循环: 在query set上计算元损失
        meta_loss += L_query(θ')

    # 更新元参数
    θ = θ - β∇meta_loss
```

#### FOMAML (First-Order MAML)
```python
# 与MAML相同，但忽略二阶梯度
# 用一阶近似替代完整的meta-gradient
θ = θ - β∇θ' L_query(θ')  # 不计算 ∂θ'/∂θ
```

#### Reptile
```python
# 更简单：直接向任务参数移动
for task in tasks:
    θ_old = θ

    # 在任务上训练K步
    for k in range(K):
        θ = θ - α∇L_task(θ)

    # 插值更新
    θ = θ_old + ε(θ - θ_old)
```

## 🔧 verl SFT实现分析

### 核心组件

```
verl框架SFT实现:
├── Trainer (sft_trainer.py)
│   ├── _build_engine()       # 构建训练引擎
│   ├── _build_dataset()      # 构建数据集
│   └── fit()                 # 训练循环
│
├── Dataset (sft_dataset.py)
│   ├── 读取parquet文件
│   ├── 应用chat template
│   ├── Tokenization
│   └── 创建loss_mask         # 关键：只对response计算loss
│
└── Loss (losses.py)
    └── sft_loss()            # masked cross-entropy loss
```

### 关键实现细节

#### 1. Loss Masking
```python
# verl/workers/roles/utils/losses.py:27-53
def sft_loss(config, model_output, data, dp_group=None):
    log_prob = model_output["log_probs"]
    loss_mask = data["loss_mask"]  # prompt部分为0，response部分为1

    # 只计算response的损失
    loss = -masked_sum(log_prob, loss_mask) / batch_num_tokens
    return loss
```

#### 2. 数据格式
```python
# verl/utils/dataset/sft_dataset.py:136-204
def __getitem__(self, item):
    prompt = self.prompts[item]
    response = self.responses[item]

    # 应用chat template
    prompt_str = tokenizer.apply_chat_template([{"role": "user", "content": prompt}])

    # 创建loss_mask
    loss_mask = attention_mask.clone()
    loss_mask[:prompt_length-1] = 0  # mask掉prompt

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'position_ids': position_ids,
        'loss_mask': loss_mask,
    }
```

## 🎓 实现要点

### 1. MAML-SFT关键点

```python
class MAMLSFTTrainer:
    def _meta_update_step(self, task_batch):
        # 保存原始参数
        original_params = clone_params(self.model)

        for task in task_batch:
            # 内循环：在support set上适应
            for k in range(self.num_inner_steps):
                loss = sft_loss(support_batch)
                grads = compute_grads(loss, create_graph=not use_fomaml)
                update_params(grads, lr=inner_lr)

            # 外循环：在query set上计算损失
            query_loss = sft_loss(query_batch)
            meta_loss += query_loss

            # 恢复原始参数
            restore_params(original_params)

        # 元梯度更新
        meta_loss.backward()
        meta_optimizer.step()
```

### 2. 与verl SFT的兼容性

我们的实现完全兼容verl的SFT数据格式和损失计算：

```python
# 使用verl的SFT数据集
from verl.utils.dataset import SFTDataset

dataset = SFTDataset(
    parquet_files=data_files,
    tokenizer=tokenizer,
    config=data_config
)

# 使用verl的损失计算逻辑
def _compute_sft_loss(self, batch, model):
    # 与verl/workers/roles/utils/losses.py中的sft_loss相同
    log_prob = compute_log_prob(model, batch)
    loss_mask = batch["loss_mask"]
    loss = -masked_sum(log_prob, loss_mask) / num_tokens
    return loss
```

## 💡 优化建议

### 内存优化

⚠️ **重要**：FOMAML是全参数优化，内存需求较高（~70GB）。如果内存不足，推荐使用**META-LORA**（只需30GB）。

```yaml
# 1. 使用FOMAML（相比MAML节省50%）
meta:
  use_fomaml: true

# 2. 使用META-LORA（相比FOMAML节省50%以上）
# 见 meta_lora_trainer.py 和 config_meta_lora_example.yaml

# 3. 减小batch size
meta:
  inner_batch_size: 2
  query_batch_size: 2
  meta_batch_size: 2

# 4. 梯度检查点
model:
  enable_gradient_checkpointing: true

# 5. 使用FSDP
model:
  use_fsdp: true
```

### 速度优化
```yaml
# 1. 减少内循环步数
meta:
  num_inner_steps: 3  # 从5降到3

# 2. Flash Attention
model:
  attn_implementation: "flash_attention_2"

# 3. 使用Reptile（更快）
# python reptile_sft_trainer.py
```

### 性能优化
```yaml
# 1. 调整学习率比例
meta:
  inner_lr: 1e-4
  outer_lr: 3e-5  # inner_lr / outer_lr ≈ 3-5

# 2. 足够的内循环步数
meta:
  num_inner_steps: 5  # 通常5-10步

# 3. 高质量support set
# 精心挑选有代表性的样本
```

## 📚 详细文档

- **[MAML_SFT_GUIDE.md](MAML_SFT_GUIDE.md)**: 完整的实现指南
  - 理论背景
  - 实现细节
  - 优化技巧
  - 故障排除
  - 实验建议

- **代码文件**:
  - `maml_sft_trainer.py`: MAML/FOMAML完整实现
  - `reptile_sft_trainer.py`: Reptile简化实现
  - `prepare_maml_data.py`: 数据准备工具

## 🔬 实验建议

### 基准实验

```python
# 实验1: 验证元学习有效性
baselines = {
    'standard_sft': train_on_all_tasks(),
    'maml_sft': meta_train_then_adapt(),
    'fomaml_sft': meta_train_then_adapt(use_fomaml=True),
    'reptile_sft': reptile_train_then_adapt(),
}

# 评估: Few-shot适应性能
for n_shots in [10, 50, 100]:
    for method, model in baselines.items():
        adapted_model = adapt(model, n_shots)
        performance = evaluate(adapted_model)
```

### 超参数搜索

```python
# 关键超参数
hyperparams = {
    'inner_lr': [1e-5, 1e-4, 1e-3],
    'outer_lr': [1e-5, 3e-5, 1e-4],
    'num_inner_steps': [1, 3, 5, 10],
    'meta_batch_size': [2, 4, 8],
}
```

## 🤝 参考资料

### 论文
1. **MAML**: Finn et al. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks" (ICML 2017)
2. **FOMAML**: Finn et al. "On First-Order Meta-Learning Algorithms" (2018)
3. **Reptile**: Nichol et al. "Reptile: A Scalable Meta-Learning Algorithm" (2018)

### 代码
- verl框架: https://github.com/volcengine/verl
- learn2learn: https://github.com/learnables/learn2learn
- higher: https://github.com/facebookresearch/higher

## 📧 问题反馈

如有问题，请：
1. 查看 [MAML_SFT_GUIDE.md](MAML_SFT_GUIDE.md) 中的故障排除部分
2. 检查代码注释
3. 参考verl框架文档

## 📄 License

本项目基于verl框架实现，遵循Apache 2.0 License。

---

**祝实验顺利！🚀**

如果这个项目对你有帮助，欢迎star和分享！
