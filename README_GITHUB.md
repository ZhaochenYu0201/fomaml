# FOMAML for Large Language Models

基于verl框架的FOMAML（First-Order Model-Agnostic Meta-Learning）实现，用于大语言模型的元学习训练。

## 🎯 项目简介

本项目实现了FOMAML算法用于LLM的few-shot学习能力提升，支持在多个数学推理任务上进行元学习训练。

### 主要特性

- ✅ **FOMAML实现**: 完整的一阶MAML算法实现
- ✅ **verl集成**: 与verl框架深度集成，复用SFT组件
- ✅ **FSDP支持**: 支持大模型的分布式训练
- ✅ **Wandb集成**: 丰富的训练指标和日志记录
- ✅ **一键运行**: 自动化的数据准备和训练脚本

### 支持的模型

- Qwen3-4B-Instruct
- Llama系列
- 其他HuggingFace格式的Causal LM模型

### 支持的数据集

- MATH (7个数学领域作为元学习任务)
- 可扩展到其他任务

## 📦 安装

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+
- GPU: 4×A100 (80GB) 推荐

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/你的用户名/meta_learning.git
cd meta_learning

# 2. 安装verl框架
cd verl
pip install -e .
cd ..

# 3. 安装依赖
pip install -r requirements.txt

# 4. 测试环境
python test_environment.py
```

## 🚀 快速开始

### 步骤1: 准备数据

```bash
python prepare_math_data.py \
    --output-dir ./data/math_meta \
    --support-ratio 0.30 \
    --query-ratio 0.40 \
    --validate
```

### 步骤2: 修改配置

编辑 `config_qwen3_4b_math.yaml`，设置你的模型路径：

```yaml
model:
  partial_pretrain: "你的模型路径"
```

### 步骤3: 开始训练

```bash
# 使用一键脚本（推荐）
./run_fomaml_qwen3_math.sh

# 或手动运行
torchrun --nproc_per_node=4 \
    maml_sft_trainer.py \
    --config-name config_qwen3_4b_math
```

详细说明请查看 [QUICKSTART_QWEN3_MATH.md](QUICKSTART_QWEN3_MATH.md)

## 📊 实验结果

训练过程中关注以下关键指标：

- `meta/avg_adaptation_gap`: 适应间隙，应随训练下降
- `meta/loss`: 元损失
- `{task}/adaptation_gap`: 各任务的适应情况

## 📁 项目结构

```
meta_learning/
├── maml_sft_trainer.py              # FOMAML训练器（主要代码）
├── meta_lora_trainer.py             # META-LORA训练器（LoRA版本）
├── prepare_math_data.py             # 数据准备脚本
├── config_qwen3_4b_math.yaml        # Qwen3-4B配置文件
├── run_fomaml_qwen3_math.sh         # 一键运行脚本
├── test_environment.py              # 环境测试脚本
├── QUICKSTART_QWEN3_MATH.md         # 快速开始指南
├── FOMAML_IMPLEMENTATION_DETAILS.md # 实现细节文档
├── DATA_PREPARATION_GUIDE.md        # 数据准备指南
└── verl/                            # verl框架（子模块）
```

## 📖 文档

- [快速开始指南](QUICKSTART_QWEN3_MATH.md) - 详细的使用教程
- [FOMAML实现详解](FOMAML_IMPLEMENTATION_DETAILS.md) - 算法实现细节
- [数据准备指南](DATA_PREPARATION_GUIDE.md) - 数据格式说明
- [修复日志](CHANGELOG_FIXES.md) - 代码改进记录

## 🔧 代码改进

本项目对原始FOMAML实现进行了以下改进：

1. ✅ 修复了内循环参数更新逻辑
2. ✅ 修复了evaluate函数的参数恢复问题
3. ✅ 增强了Wandb指标记录（新增5个FOMAML特有指标）
4. ✅ 添加了完整的数据准备和训练脚本

详见 [CHANGELOG_FIXES.md](CHANGELOG_FIXES.md)

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目基于verl框架开发，遵循Apache 2.0许可证。

## 🙏 致谢

- [verl](https://github.com/volcengine/verl) - 强化学习框架
- [MAML](https://arxiv.org/abs/1703.03400) - 原始MAML论文
- [MATH Dataset](https://github.com/hendrycks/math) - 数学推理数据集

## 📧 联系方式

如有问题，请提交Issue或联系项目维护者。

---

**注意**: 本项目需要大量GPU资源（推荐4×A100），请确保有足够的计算资源。
