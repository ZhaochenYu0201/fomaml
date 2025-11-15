# Qwen3-4B + MATH 快速开始指南

本指南帮助你快速使用Qwen3-4B-Instruct模型在MATH数据集上进行FOMAML训练。

---

## 📋 前置要求

### 硬件要求
- **GPU**: 4×A100 (80GB) 或同等算力
- **内存**: 128GB+
- **存储**: 100GB+

### 软件要求
```bash
# Python版本
Python 3.8+

# 核心依赖
torch >= 2.0.0
transformers >= 4.35.0
datasets >= 2.14.0
pandas >= 2.0.0
numpy
tensordict
tqdm
wandb  # 用于日志记录
omegaconf
```

### 安装依赖

```bash
# 基础环境
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# verl框架（如果还没安装）
cd verl
pip install -e .

# 其他依赖
pip install datasets pandas transformers wandb omegaconf tensordict tqdm
```

---

## 🚀 快速开始

### 步骤0: 准备模型

确保你已经下载了Qwen3-4B-Instruct-2507模型到本地，例如：
```
./models/Qwen3-4B-Instruct-2507/
```

### 步骤1: 准备数据

运行数据准备脚本：

```bash
python prepare_math_data.py \
    --output-dir ./data/math_meta \
    --support-ratio 0.30 \
    --query-ratio 0.40 \
    --validate
```

**参数说明**：
- `--output-dir`: 数据输出目录
- `--support-ratio`: Support集比例（用于内循环适应）
- `--query-ratio`: Query集比例（用于元梯度计算）
- `--validate`: 验证生成的数据格式

**预期输出**：
```
找到 7 个数学领域：
  📌 Algebra: 1187 个问题
  📌 Number Theory: 869 个问题
  📌 Precalculus: 546 个问题
  📌 Intermediate Algebra: 1207 个问题
  📌 Counting and Probability: 474 个问题
  📌 Geometry: 479 个问题
  📌 Prealgebra: 871 个问题

数据集统计:
                       task  support  query  test  total
                    algebra      356    475   356   1187
  counting_and_probability      142    189   143    474
                   geometry      143    191   145    479
       intermediate_algebra      362    482   363   1207
              number_theory      260    347   262    869
                precalculus      163    218   165    546
                 prealgebra      261    348   262    871
```

### 步骤2: 修改配置文件

编辑 `config_qwen3_4b_math.yaml`，确保模型路径正确：

```yaml
model:
  partial_pretrain: "./models/Qwen3-4B-Instruct-2507"  # 修改为你的路径
```

可选：调整训练参数
```yaml
meta:
  inner_lr: 1.0e-4          # 内循环学习率
  num_inner_steps: 5        # 内循环步数
  outer_lr: 3.0e-5          # 外循环学习率
  meta_batch_size: 4        # 每次元更新使用的任务数

trainer:
  total_steps: 5000         # 总训练步数
  save_freq: 500            # checkpoint保存频率
  test_freq: 100            # 评估频率
```

### 步骤3: 开始训练

#### 方式A: 使用一键脚本（推荐）

**Windows:**
```cmd
run_fomaml_qwen3_math.bat
```

**Linux/Mac:**
```bash
chmod +x run_fomaml_qwen3_math.sh
./run_fomaml_qwen3_math.sh
```

#### 方式B: 手动运行

```bash
torchrun --nproc_per_node=4 \
    --master_port=29500 \
    maml_sft_trainer.py \
    --config-name config_qwen3_4b_math
```

**参数说明**：
- `--nproc_per_node`: GPU数量
- `--master_port`: 分布式训练端口
- `--config-name`: 配置文件名（不含.yaml后缀）

---

## 📊 监控训练

### Wandb Dashboard

训练开始后，访问 [https://wandb.ai](https://wandb.ai) 查看实时训练日志。

**关键指标**：

1. **meta/loss**: 元损失（应持续下降）
2. **meta/avg_adaptation_gap**: 平均适应间隙
   - 定义: `query_loss - support_loss`
   - 意义: 越小说明模型的元初始化越好
   - 期望: 随训练逐渐减小

3. **meta/grad_norm**: 梯度范数
   - 应保持稳定，不应爆炸或消失
   - 如果过大(>10)，考虑降低学习率

4. **任务特定指标**:
   - `{task_name}/support_loss`: 每个任务的support集损失
   - `{task_name}/query_loss`: 每个任务的query集损失
   - `{task_name}/adaptation_gap`: 每个任务的适应间隙

### 训练日志示例

```
Step 100/5000:
  meta/loss: 2.45
  meta/avg_adaptation_gap: 0.32  ← 期望这个值下降
  meta/grad_norm: 1.2
  algebra/support_loss: 2.10
  algebra/query_loss: 2.42
  algebra/adaptation_gap: 0.32
  ...

Step 500/5000:
  meta/loss: 1.85
  meta/avg_adaptation_gap: 0.18  ← 已经下降！
  meta/grad_norm: 0.9
  ...
```

---

## 💾 Checkpoint管理

### Checkpoint结构

训练会自动保存checkpoint到 `./checkpoints/fomaml_qwen3_4b_math/`：

```
checkpoints/fomaml_qwen3_4b_math/
├── maml_checkpoint_step_500.pt
├── maml_checkpoint_step_1000.pt
├── maml_checkpoint_step_1500.pt
└── ...
```

### 加载Checkpoint

```python
import torch

# 加载checkpoint
checkpoint = torch.load('checkpoints/fomaml_qwen3_4b_math/maml_checkpoint_step_5000.pt')

# 查看包含的内容
print(checkpoint.keys())
# dict_keys(['step', 'model_state_dict', 'optimizer_state_dict', 'config'])

# 恢复训练（在配置文件中设置）
# trainer:
#   resume_from_path: "./checkpoints/fomaml_qwen3_4b_math/maml_checkpoint_step_2000.pt"
```

---

## 🔧 常见问题

### Q1: 显存不足 (CUDA out of memory)

**解决方案**：

1. **减小batch size**:
```yaml
meta:
  inner_batch_size: 2      # 从4降到2
  meta_batch_size: 2       # 从4降到2
  query_batch_size: 2
```

2. **开启CPU offload**:
```yaml
model:
  fsdp_config:
    cpu_offload: true
```

3. **减少GPU数量但增加gradient accumulation**:
```bash
# 使用2卡而不是4卡
torchrun --nproc_per_node=2 maml_sft_trainer.py ...
```

### Q2: 训练速度太慢

**当前预期速度**: ~10-12 steps/小时 (4×A100)

**加速方法**:

1. **减少内循环步数**:
```yaml
meta:
  num_inner_steps: 3  # 从5降到3
```

2. **减少任务数**:
   - 只选择3-4个核心任务进行训练

3. **使用混合精度**:
   - 配置文件已默认使用bf16

### Q3: adaptation_gap不下降

可能原因和解决方案：

1. **学习率过大或过小**:
```yaml
meta:
  inner_lr: 5.0e-5    # 尝试降低
  outer_lr: 1.0e-5    # 尝试降低
```

2. **内循环步数不够**:
```yaml
meta:
  num_inner_steps: 10  # 增加到10
```

3. **任务之间差异太大**:
   - 检查各个任务的loss，如果某些任务特别高，考虑移除

### Q4: 数据准备失败

```bash
# 错误: 无法下载MATH数据集

# 解决方案1: 使用镜像
export HF_ENDPOINT=https://hf-mirror.com
python prepare_math_data.py ...

# 解决方案2: 手动下载
# 1. 访问 https://github.com/hendrycks/math
# 2. 下载数据集
# 3. 使用本地路径加载
```

---

## 📈 预期结果

### 训练时长
- **总步数**: 5000 steps
- **速度**: ~10 steps/小时 (4×A100)
- **总时长**: ~40-50小时

### 性能指标

成功的训练应该显示：

```
初始阶段 (0-500 steps):
  meta/loss: 3.0 → 2.2
  meta/avg_adaptation_gap: 0.8 → 0.4

中期 (500-2500 steps):
  meta/loss: 2.2 → 1.5
  meta/avg_adaptation_gap: 0.4 → 0.2

后期 (2500-5000 steps):
  meta/loss: 1.5 → 1.0
  meta/avg_adaptation_gap: 0.2 → 0.1
```

**关键观察**：
- ✅ `meta/loss` 持续下降
- ✅ `adaptation_gap` 逐渐减小（说明元初始化越来越好）
- ✅ 各任务的loss相对平衡

---

## 🎯 下一步

训练完成后：

1. **Few-shot评估**:
```bash
python evaluate_few_shot.py \
    --model-path ./checkpoints/fomaml_qwen3_4b_math/maml_checkpoint_step_5000.pt \
    --n-shots 0 5 10 25 50
```

2. **对比实验**:
   - 训练baseline SFT模型（不使用FOMAML）
   - 对比few-shot性能

3. **分析结果**:
   - 查看wandb日志
   - 绘制学习曲线
   - 统计显著性检验

---

## 📚 相关文档

- [FOMAML实现详解](FOMAML_IMPLEMENTATION_DETAILS.md)
- [数据准备完整指南](DATA_PREPARATION_GUIDE.md)
- [实验设计方案](EXPERIMENT_DESIGN_MATH_SCIENCE.md)

---

## 📧 问题反馈

遇到问题？

1. 检查日志输出
2. 查看wandb dashboard
3. 参考常见问题部分
4. 在项目中提issue

---

祝训练顺利！🚀
