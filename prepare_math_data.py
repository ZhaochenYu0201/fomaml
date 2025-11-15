"""
MATH数据集准备脚本 - FOMAML训练专用

功能：
1. 下载和加载MATH数据集
2. 按数学领域划分任务（代数、几何、概率等）
3. 生成support/query split用于meta-training
4. 保存为parquet格式

使用方法：
    python prepare_math_data.py --output-dir ./data/math_meta --support-ratio 0.30
"""

import os
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm


def prepare_math_dataset(
    output_dir: str,
    support_ratio: float = 0.30,
    query_ratio: float = 0.40,
    seed: int = 42,
    max_samples_per_task: int = -1,
):
    """
    准备MATH数据集用于FOMAML训练

    Args:
        output_dir: 输出目录
        support_ratio: Support集比例
        query_ratio: Query集比例
        seed: 随机种子
        max_samples_per_task: 每个任务的最大样本数（-1表示使用全部）
    """
    random.seed(seed)
    np.random.seed(seed)

    output_path = Path(output_dir)
    (output_path / "meta_train").mkdir(parents=True, exist_ok=True)
    (output_path / "few_shot_eval").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MATH数据集准备 - FOMAML训练")
    print("=" * 60)

    # 加载MATH数据集
    print("\n📥 正在加载MATH数据集...")
    try:
        dataset = load_dataset("hendrycks/competition_math", split="train")
        print(f"✅ 成功加载 {len(dataset)} 个样本")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        print("\n解决方案：")
        print("1. 确保已安装: pip install datasets")
        print("2. 或手动下载: https://github.com/hendrycks/math")
        return

    # 按数学领域分组
    print("\n📊 按数学领域分组...")
    tasks = defaultdict(list)
    for example in tqdm(dataset, desc="分组中"):
        subject = example['type']  # 'Algebra', 'Geometry', 'Number Theory', etc.
        tasks[subject].append(example)

    print(f"\n找到 {len(tasks)} 个数学领域：")
    for subject, examples in sorted(tasks.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  📌 {subject}: {len(examples)} 个问题")

    # 为每个领域创建support/query/test split
    print(f"\n🔄 创建数据划分 (support={support_ratio:.0%}, query={query_ratio:.0%})...")

    task_summary = []

    for subject, examples in tasks.items():
        task_name = subject.lower().replace(' ', '_')

        # 可选：限制样本数
        if max_samples_per_task > 0 and len(examples) > max_samples_per_task:
            examples = random.sample(examples, max_samples_per_task)

        # 打乱数据
        random.shuffle(examples)

        # 计算划分点
        n_total = len(examples)
        n_support = int(n_total * support_ratio)
        n_query = int(n_total * query_ratio)

        # 划分数据
        support_examples = examples[:n_support]
        query_examples = examples[n_support:n_support + n_query]
        test_examples = examples[n_support + n_query:]

        # 格式转换
        support_data = [format_math_example(ex) for ex in support_examples]
        query_data = [format_math_example(ex) for ex in query_examples]
        test_data = [format_math_example(ex) for ex in test_examples]

        # 保存为parquet
        support_path = output_path / "meta_train" / f"{task_name}_support.parquet"
        query_path = output_path / "meta_train" / f"{task_name}_query.parquet"
        test_path = output_path / "few_shot_eval" / f"{task_name}_test.parquet"

        pd.DataFrame(support_data).to_parquet(support_path, index=False)
        pd.DataFrame(query_data).to_parquet(query_path, index=False)
        pd.DataFrame(test_data).to_parquet(test_path, index=False)

        task_summary.append({
            'task': task_name,
            'support': len(support_data),
            'query': len(query_data),
            'test': len(test_data),
            'total': n_total
        })

        print(f"  ✅ {task_name:20s} → support={len(support_data):4d}, query={len(query_data):4d}, test={len(test_data):4d}")

    # 保存统计信息
    summary_df = pd.DataFrame(task_summary)
    summary_path = output_path / "dataset_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\n📈 数据集统计:")
    print(summary_df.to_string(index=False))
    print(f"\n💾 统计信息已保存至: {summary_path}")

    # 生成配置文件模板
    generate_config_template(output_path, task_summary)

    print("\n" + "=" * 60)
    print("✅ 数据准备完成！")
    print("=" * 60)
    print(f"\n📁 数据保存位置: {output_path.absolute()}")
    print(f"📝 配置模板: {output_path / 'config_template.yaml'}")
    print(f"\n下一步:")
    print(f"  1. 检查配置: {output_path / 'config_template.yaml'}")
    print(f"  2. 开始训练: torchrun --nproc_per_node=4 maml_sft_trainer.py")


def format_math_example(example: Dict) -> Dict:
    """
    将MATH数据集样本转换为verl SFT格式

    Args:
        example: MATH数据集原始样本

    Returns:
        verl格式的样本
    """
    problem = example['problem']
    solution = example['solution']
    level = example['level']
    subject = example['type']

    # 构建prompt（带指令）
    prompt = f"""请解决以下数学问题。请提供详细的解题步骤，并在最后用 \\boxed{{}} 标注最终答案。

问题：{problem}

请一步步推理并给出答案。"""

    # response就是完整的解答
    response = solution

    # metadata（方便后续分析）
    metadata = json.dumps({
        'source': 'MATH',
        'subject': subject,
        'level': level,
    })

    return {
        'prompt': prompt,
        'response': response,
        'metadata': metadata,
    }


def generate_config_template(output_dir: Path, task_summary: List[Dict]):
    """生成配置文件模板"""

    config_template = f"""# FOMAML-SFT 配置文件
# 自动生成于: {pd.Timestamp.now()}

model:
  partial_pretrain: "你的模型路径"  # 例如: "./models/Qwen2.5-7B-Instruct"
  trust_remote_code: true
  use_fsdp: true
  enable_gradient_checkpointing: true

  fsdp_config:
    wrap_policy:
      transformer_layer_cls_to_wrap: "Qwen2DecoderLayer"  # Qwen模型使用此配置
    model_dtype: "bf16"
    cpu_offload: false

data:
  max_length: 2048
  truncation: "right"
  prompt_key: "prompt"
  response_key: "response"

meta:
  use_fomaml: true

  # 内循环参数
  inner_lr: 1.0e-4
  num_inner_steps: 5
  inner_batch_size: 4

  # 外循环参数
  outer_lr: 3.0e-5
  meta_batch_size: 4
  query_batch_size: 4

  # 任务定义
  tasks:
"""

    for task_info in task_summary:
        task_name = task_info['task']
        config_template += f"""    - name: "{task_name}"
      support_files: ["{output_dir}/meta_train/{task_name}_support.parquet"]
      query_files: ["{output_dir}/meta_train/{task_name}_query.parquet"]
      support_max_samples: {task_info['support']}
      query_max_samples: {task_info['query']}

"""

    config_template += """
optim:
  optimizer_type: "AdamW"
  lr: 3.0e-5
  weight_decay: 0.01
  betas: [0.9, 0.999]
  clip_grad: 1.0

  lr_scheduler: "cosine"
  lr_warmup_steps_ratio: 0.1

trainer:
  device: "cuda"
  total_steps: 5000
  save_freq: 500
  test_freq: 100

  # Wandb配置
  project_name: "fomaml-math"
  experiment_name: "qwen-math-meta-learning"
  logger: "wandb"

  default_local_dir: "./checkpoints/fomaml_math"
"""

    config_path = output_dir / "config_template.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_template)

    print(f"\n✅ 配置模板已生成: {config_path}")


def validate_dataset(file_path: Path):
    """验证parquet文件格式"""
    try:
        df = pd.read_parquet(file_path)

        # 检查必需列
        required_cols = ['prompt', 'response']
        missing = [col for col in required_cols if col not in df.columns]

        if missing:
            print(f"❌ {file_path.name}: 缺少列 {missing}")
            return False

        # 检查空值
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            print(f"⚠️  {file_path.name}: 存在空值\n{null_counts}")

        print(f"✅ {file_path.name}: {len(df)} 个样本, 格式正确")
        return True

    except Exception as e:
        print(f"❌ {file_path.name}: 读取失败 - {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="准备MATH数据集用于FOMAML训练")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/math_meta",
        help="输出目录"
    )
    parser.add_argument(
        "--support-ratio",
        type=float,
        default=0.30,
        help="Support集比例 (默认: 0.30)"
    )
    parser.add_argument(
        "--query-ratio",
        type=float,
        default=0.40,
        help="Query集比例 (默认: 0.40)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="每个任务的最大样本数 (-1表示使用全部)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="数据准备完成后验证数据格式"
    )

    args = parser.parse_args()

    # 准备数据
    prepare_math_dataset(
        output_dir=args.output_dir,
        support_ratio=args.support_ratio,
        query_ratio=args.query_ratio,
        seed=args.seed,
        max_samples_per_task=args.max_samples,
    )

    # 验证数据
    if args.validate:
        print("\n" + "=" * 60)
        print("🔍 验证数据格式...")
        print("=" * 60)

        output_path = Path(args.output_dir)
        all_valid = True

        for parquet_file in (output_path / "meta_train").glob("*.parquet"):
            if not validate_dataset(parquet_file):
                all_valid = False

        if all_valid:
            print("\n✅ 所有数据文件验证通过！")
        else:
            print("\n⚠️  部分数据文件验证失败，请检查。")


if __name__ == "__main__":
    main()
