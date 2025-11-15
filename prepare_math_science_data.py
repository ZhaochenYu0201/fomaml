"""
数据准备脚本：MATH & Science推理任务

用于准备FOMAML-SFT vs SFT对比实验的数据集

功能：
1. 下载和加载MATH、GSM8K、ScienceQA数据集
2. 按任务划分数据
3. 生成support/query split用于meta-training
4. 准备few-shot评估数据
5. 数据格式转换和验证

使用方法：
    python prepare_math_science_data.py --output-dir ./data/math_science_meta
"""

import os
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm


class MathScienceDataPreparator:
    """数学和科学推理任务的数据准备器"""

    def __init__(self, output_dir: str, seed: int = 42):
        self.output_dir = Path(output_dir)
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "meta_train").mkdir(exist_ok=True)
        (self.output_dir / "few_shot_eval").mkdir(exist_ok=True)

    def prepare_math_dataset(
        self,
        support_ratio: float = 0.2,
        query_ratio: float = 0.3,
    ):
        """
        准备MATH数据集

        MATH数据集包含7个数学领域，每个领域作为一个meta-learning任务
        """
        print("\n=== Preparing MATH Dataset ===")

        # 加载MATH数据集
        print("Loading MATH dataset...")
        try:
            dataset = load_dataset("hendrycks/competition_math", split="train")
        except Exception as e:
            print(f"Error loading MATH dataset: {e}")
            print("请确保已安装datasets库：pip install datasets")
            print("或手动下载数据集：https://github.com/hendrycks/math")
            return

        # 按type分组
        tasks = defaultdict(list)
        for example in tqdm(dataset, desc="Grouping by subject"):
            subject = example['type']  # 'Algebra', 'Geometry', etc.
            tasks[subject].append(example)

        print(f"\nFound {len(tasks)} subjects in MATH dataset")
        for subject, examples in tasks.items():
            print(f"  {subject}: {len(examples)} examples")

        # 为每个subject创建support/query/test split
        for subject, examples in tasks.items():
            self._process_task_split(
                task_name=subject.lower().replace(' ', '_'),
                examples=examples,
                support_ratio=support_ratio,
                query_ratio=query_ratio,
                format_func=self._format_math_example,
            )

    def prepare_gsm8k_dataset(self):
        """准备GSM8K数据集（小学数学应用题）"""
        print("\n=== Preparing GSM8K Dataset ===")

        try:
            dataset = load_dataset("gsm8k", "main", split="train")
        except Exception as e:
            print(f"Error loading GSM8K: {e}")
            return

        examples = list(dataset)
        print(f"Loaded {len(examples)} GSM8K examples")

        self._process_task_split(
            task_name="word_problems",
            examples=examples,
            support_ratio=0.25,
            query_ratio=0.35,
            format_func=self._format_gsm8k_example,
        )

    def prepare_scienceqa_dataset(self):
        """准备ScienceQA数据集"""
        print("\n=== Preparing ScienceQA Dataset ===")

        try:
            dataset = load_dataset("derek-thomas/ScienceQA", split="train")
        except Exception as e:
            print(f"Error loading ScienceQA: {e}")
            return

        # 按subject分组
        tasks = defaultdict(list)
        for example in tqdm(dataset, desc="Grouping by subject"):
            subject = example.get('subject', 'general')
            if subject in ['physics', 'chemistry', 'biology']:
                tasks[subject].append(example)

        for subject, examples in tasks.items():
            print(f"\n{subject}: {len(examples)} examples")
            self._process_task_split(
                task_name=f"science_{subject}",
                examples=examples,
                support_ratio=0.2,
                query_ratio=0.3,
                format_func=self._format_scienceqa_example,
            )

    def _process_task_split(
        self,
        task_name: str,
        examples: List,
        support_ratio: float,
        query_ratio: float,
        format_func,
    ):
        """
        将任务数据划分为support/query/test三个集合

        Args:
            task_name: 任务名称
            examples: 原始样本列表
            support_ratio: support集比例
            query_ratio: query集比例
            format_func: 格式转换函数
        """
        print(f"\n--- Processing task: {task_name} ---")

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

        print(f"  Support: {len(support_examples)}")
        print(f"  Query:   {len(query_examples)}")
        print(f"  Test:    {len(test_examples)}")

        # 格式转换
        support_data = [format_func(ex) for ex in tqdm(support_examples, desc="  Formatting support")]
        query_data = [format_func(ex) for ex in tqdm(query_examples, desc="  Formatting query")]
        test_data = [format_func(ex) for ex in tqdm(test_examples, desc="  Formatting test")]

        # 保存为parquet格式（verl友好）
        support_df = pd.DataFrame(support_data)
        query_df = pd.DataFrame(query_data)
        test_df = pd.DataFrame(test_data)

        # Meta-training数据
        support_path = self.output_dir / "meta_train" / f"{task_name}_support.parquet"
        query_path = self.output_dir / "meta_train" / f"{task_name}_query.parquet"

        support_df.to_parquet(support_path, index=False)
        query_df.to_parquet(query_path, index=False)

        print(f"  Saved to: {support_path}")
        print(f"  Saved to: {query_path}")

        # Few-shot评估数据
        test_path = self.output_dir / "few_shot_eval" / f"{task_name}_test.parquet"
        test_df.to_parquet(test_path, index=False)
        print(f"  Saved to: {test_path}")

        # 创建few-shot采样
        for n_shots in [5, 10, 25, 50]:
            few_shot_path = self.output_dir / "few_shot_eval" / f"{task_name}_{n_shots}shot.parquet"
            few_shot_df = test_df.sample(n=min(n_shots, len(test_df)), random_state=self.seed)
            few_shot_df.to_parquet(few_shot_path, index=False)

    def _format_math_example(self, example: Dict) -> Dict:
        """
        格式化MATH数据集样本

        MATH数据格式:
        {
            'problem': '求解方程...',
            'solution': '详细解答...',
            'level': 'Level 3',
            'type': 'Algebra'
        }
        """
        problem = example['problem']
        solution = example['solution']
        level = example.get('level', 'Unknown')
        subject = example.get('type', 'Unknown')

        # 构建prompt（使用CoT格式）
        prompt = f"""请解决以下数学问题。请提供详细的解题步骤和最终答案。

问题：{problem}

请一步步推理并给出答案。"""

        # 构建response（包含推理过程）
        response = f"""让我来一步步解决这个问题：

{solution}"""

        return {
            'prompt': prompt,
            'response': response,
            'metadata': json.dumps({
                'level': level,
                'subject': subject,
                'source': 'MATH',
            })
        }

    def _format_gsm8k_example(self, example: Dict) -> Dict:
        """
        格式化GSM8K数据集样本

        GSM8K格式:
        {
            'question': '问题描述',
            'answer': '解答过程#### 答案'
        }
        """
        question = example['question']
        answer = example['answer']

        # GSM8K的answer包含推理过程和最终答案（用####分隔）
        if '####' in answer:
            reasoning, final_answer = answer.split('####')
        else:
            reasoning = answer
            final_answer = ''

        prompt = f"""请解决以下数学应用题。请提供详细的解题步骤和最终答案。

问题：{question}

请一步步推理并给出答案。"""

        response = f"""让我来一步步解决这个问题：

{reasoning.strip()}

因此，答案是：{final_answer.strip()}"""

        return {
            'prompt': prompt,
            'response': response,
            'metadata': json.dumps({
                'source': 'GSM8K',
                'subject': 'word_problems',
            })
        }

    def _format_scienceqa_example(self, example: Dict) -> Dict:
        """格式化ScienceQA数据集样本"""
        question = example.get('question', '')
        choices = example.get('choices', [])
        answer = example.get('answer', '')
        hint = example.get('hint', '')
        subject = example.get('subject', 'science')

        # 构建多选题格式
        choices_str = '\n'.join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])

        prompt = f"""请回答以下科学问题。请提供推理过程和最终答案。

问题：{question}

选项：
{choices_str}

请一步步分析并给出答案。"""

        # 构建回答
        if hint:
            response = f"""让我来分析这个问题：

提示：{hint}

基于以上分析，正确答案是：{answer}"""
        else:
            response = f"""正确答案是：{answer}"""

        return {
            'prompt': prompt,
            'response': response,
            'metadata': json.dumps({
                'source': 'ScienceQA',
                'subject': subject,
            })
        }

    def create_baseline_sft_data(self):
        """
        创建baseline SFT的训练数据

        将所有meta-training任务的support和query数据混合
        """
        print("\n=== Creating Baseline SFT Data ===")

        all_data = []
        meta_train_dir = self.output_dir / "meta_train"

        for file in meta_train_dir.glob("*_support.parquet"):
            df = pd.read_parquet(file)
            all_data.append(df)
            print(f"  Added {len(df)} samples from {file.name}")

        for file in meta_train_dir.glob("*_query.parquet"):
            df = pd.read_parquet(file)
            all_data.append(df)
            print(f"  Added {len(df)} samples from {file.name}")

        # 合并所有数据
        combined_df = pd.concat(all_data, ignore_index=True)

        # 打乱
        combined_df = combined_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        # 保存
        output_path = self.output_dir / "baseline_sft_all_mixed.parquet"
        combined_df.to_parquet(output_path, index=False)

        print(f"\nTotal samples: {len(combined_df)}")
        print(f"Saved to: {output_path}")

    def generate_config_files(self):
        """生成配置文件"""
        print("\n=== Generating Config Files ===")

        # 收集所有meta-training任务
        meta_train_dir = self.output_dir / "meta_train"
        tasks = []

        support_files = sorted(meta_train_dir.glob("*_support.parquet"))
        for support_file in support_files:
            task_name = support_file.stem.replace('_support', '')
            query_file = meta_train_dir / f"{task_name}_query.parquet"

            if query_file.exists():
                support_df = pd.read_parquet(support_file)
                query_df = pd.read_parquet(query_file)

                task_config = {
                    'name': task_name,
                    'support_files': [str(support_file.relative_to(self.output_dir.parent))],
                    'query_files': [str(query_file.relative_to(self.output_dir.parent))],
                    'support_max_samples': len(support_df),
                    'query_max_samples': len(query_df),
                }
                tasks.append(task_config)

        # 生成FOMAML-SFT配置
        fomaml_config = {
            'model': {
                'partial_pretrain': 'meta-llama/Llama-3.2-1B',
                'lora_rank': 16,
                'lora_alpha': 32,
                'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
                'use_fsdp': True,
                'trust_remote_code': True,
            },
            'meta': {
                'use_fomaml': True,
                'inner_lr': 5e-5,
                'num_inner_steps': 5,
                'inner_batch_size': 4,
                'outer_lr': 2e-5,
                'meta_batch_size': 3,
                'query_batch_size': 4,
                'tasks': tasks,
            },
            'data': {
                'max_length': 2048,
                'truncation': 'right',
                'prompt_key': 'prompt',
                'response_key': 'response',
            },
            'optim': {
                'optimizer_type': 'AdamW',
                'lr': 2e-5,
                'weight_decay': 0.01,
                'clip_grad': 1.0,
                'betas': [0.9, 0.999],
            },
            'trainer': {
                'device': 'cuda',
                'total_steps': 5000,
                'save_freq': 500,
                'test_freq': 100,
                'project_name': 'fomaml-math-science',
                'experiment_name': 'llama-3.2-1b-math-science',
                'logger': 'wandb',
                'default_local_dir': './checkpoints/fomaml_math_science',
            },
        }

        config_path = self.output_dir / "config_fomaml_math_science.yaml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(fomaml_config, f, default_flow_style=False)

        print(f"Generated FOMAML config: {config_path}")

        # 生成baseline SFT配置
        sft_config = {
            'model': fomaml_config['model'],
            'data': {
                **fomaml_config['data'],
                'train_files': [str((self.output_dir / "baseline_sft_all_mixed.parquet").relative_to(self.output_dir.parent))],
                'train_batch_size': 32,
            },
            'optim': fomaml_config['optim'],
            'trainer': {
                **fomaml_config['trainer'],
                'total_epochs': 3,
                'experiment_name': 'baseline-sft-mixed',
                'default_local_dir': './checkpoints/baseline_sft',
            },
        }

        sft_config_path = self.output_dir / "config_baseline_sft.yaml"
        with open(sft_config_path, 'w') as f:
            yaml.dump(sft_config, f, default_flow_style=False)

        print(f"Generated SFT config: {sft_config_path}")

    def print_summary(self):
        """打印数据集摘要"""
        print("\n" + "="*60)
        print("DATA PREPARATION SUMMARY")
        print("="*60)

        meta_train_dir = self.output_dir / "meta_train"
        few_shot_dir = self.output_dir / "few_shot_eval"

        print("\nMeta-Training Tasks:")
        tasks = set()
        for file in sorted(meta_train_dir.glob("*_support.parquet")):
            task_name = file.stem.replace('_support', '')
            tasks.add(task_name)

            support_df = pd.read_parquet(file)
            query_file = meta_train_dir / f"{task_name}_query.parquet"
            query_df = pd.read_parquet(query_file) if query_file.exists() else pd.DataFrame()

            print(f"  {task_name}:")
            print(f"    Support: {len(support_df)} samples")
            print(f"    Query:   {len(query_df)} samples")

        print(f"\nTotal meta-training tasks: {len(tasks)}")

        print("\nFew-Shot Evaluation Tasks:")
        test_files = sorted(few_shot_dir.glob("*_test.parquet"))
        for file in test_files:
            task_name = file.stem.replace('_test', '')
            df = pd.read_parquet(file)
            print(f"  {task_name}: {len(df)} test samples")

        baseline_file = self.output_dir / "baseline_sft_all_mixed.parquet"
        if baseline_file.exists():
            baseline_df = pd.read_parquet(baseline_file)
            print(f"\nBaseline SFT mixed data: {len(baseline_df)} samples")

        print("\n" + "="*60)
        print("Next steps:")
        print("1. Train FOMAML-SFT:")
        print("   python maml_sft_trainer.py --config-name config_fomaml_math_science")
        print("\n2. Train Baseline SFT:")
        print("   python verl/trainer/sft_trainer.py --config-name config_baseline_sft")
        print("\n3. Evaluate:")
        print("   python evaluate_few_shot.py --model-path checkpoints/fomaml_math_science")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare MATH & Science data for meta-learning")
    parser.add_argument("--output-dir", type=str, default="./data/math_science_meta",
                        help="Output directory for processed data")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--support-ratio", type=float, default=0.2,
                        help="Ratio of data for support set")
    parser.add_argument("--query-ratio", type=float, default=0.3,
                        help="Ratio of data for query set")
    parser.add_argument("--skip-math", action="store_true",
                        help="Skip MATH dataset preparation")
    parser.add_argument("--skip-gsm8k", action="store_true",
                        help="Skip GSM8K dataset preparation")
    parser.add_argument("--skip-scienceqa", action="store_true",
                        help="Skip ScienceQA dataset preparation")

    args = parser.parse_args()

    # 创建数据准备器
    preparator = MathScienceDataPreparator(
        output_dir=args.output_dir,
        seed=args.seed
    )

    # 准备各个数据集
    if not args.skip_math:
        preparator.prepare_math_dataset(
            support_ratio=args.support_ratio,
            query_ratio=args.query_ratio,
        )

    if not args.skip_gsm8k:
        preparator.prepare_gsm8k_dataset()

    if not args.skip_scienceqa:
        preparator.prepare_scienceqa_dataset()

    # 创建baseline数据
    preparator.create_baseline_sft_data()

    # 生成配置文件
    preparator.generate_config_files()

    # 打印摘要
    preparator.print_summary()


if __name__ == "__main__":
    main()
