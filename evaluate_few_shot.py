"""
Few-Shot学习评估脚本

评估FOMAML-SFT vs SFT在数学和科学推理任务上的few-shot学习能力

功能：
1. Few-shot学习曲线评估
2. 跨任务泛化评估
3. 适应速度评估
4. 统计显著性检验

使用方法：
    # 评估FOMAML模型
    python evaluate_few_shot.py \\
        --model-path checkpoints/fomaml_math_science/step_5000 \\
        --model-type fomaml \\
        --eval-tasks algebra geometry calculus \\
        --output-dir results/fomaml

    # 评估baseline SFT模型
    python evaluate_few_shot.py \\
        --model-path checkpoints/baseline_sft \\
        --model-type sft \\
        --eval-tasks algebra geometry calculus \\
        --output-dir results/sft
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import copy

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

# 设置样式
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class SFTDataset(Dataset):
    """简单的SFT数据集"""

    def __init__(self, data: List[Dict], tokenizer, max_length=2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        prompt = example['prompt']
        response = example['response']

        # 应用chat template
        prompt_chat = [{"role": "user", "content": prompt}]
        prompt_str = self.tokenizer.apply_chat_template(
            prompt_chat, add_generation_prompt=True, tokenize=False
        )
        full_str = prompt_str + response + self.tokenizer.eos_token

        # Tokenize
        encoding = self.tokenizer(
            full_str,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        # 创建loss mask（只对response部分计算loss）
        prompt_encoding = self.tokenizer(
            prompt_str,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        prompt_length = prompt_encoding['input_ids'].shape[1]

        loss_mask = encoding['attention_mask'].clone()
        loss_mask[0, :prompt_length] = 0  # mask掉prompt

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'loss_mask': loss_mask.squeeze(0),
            'prompt': prompt,
            'response': response,
        }


class FewShotEvaluator:
    """Few-Shot学习评估器"""

    def __init__(
        self,
        model,
        tokenizer,
        device='cuda',
        adaptation_lr=1e-4,
        adaptation_steps=100,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.adaptation_lr = adaptation_lr
        self.adaptation_steps = adaptation_steps

    def evaluate_zero_shot(self, test_data: List[Dict]) -> Dict:
        """Zero-shot评估"""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        dataset = SFTDataset(test_data, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Zero-shot evaluation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                loss_mask = batch['loss_mask'].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )

                # 计算masked loss
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                shift_mask = loss_mask[..., 1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                loss = (loss * shift_mask.view(-1)).sum() / (shift_mask.sum() + 1e-8)

                total_loss += loss.item()

                # 生成预测并检查正确性
                generated = self.generate_response(batch['prompt'][0])
                is_correct = self.check_correctness(
                    generated,
                    batch['response'][0]
                )
                if is_correct:
                    correct += 1
                total += 1

        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total if total > 0 else 0.0,
        }

    def adapt_and_evaluate(
        self,
        few_shot_data: List[Dict],
        test_data: List[Dict],
        n_runs: int = 1,
    ) -> Dict:
        """
        在few-shot数据上适应，然后在test数据上评估

        Args:
            few_shot_data: Few-shot样本
            test_data: 测试样本
            n_runs: 重复次数（减少随机性）

        Returns:
            评估结果字典
        """
        results = []

        for run in range(n_runs):
            # 克隆模型（避免修改原模型）
            adapted_model = copy.deepcopy(self.model)
            adapted_model.to(self.device)

            # 在few-shot数据上fine-tune
            if len(few_shot_data) > 0:
                print(f"Run {run+1}/{n_runs}: Adapting on {len(few_shot_data)} samples...")
                adapted_model = self._adapt(adapted_model, few_shot_data)

            # 在test数据上评估
            print(f"Run {run+1}/{n_runs}: Evaluating on {len(test_data)} samples...")
            eval_result = self._evaluate_model(adapted_model, test_data)
            results.append(eval_result)

            # 清理内存
            del adapted_model
            torch.cuda.empty_cache()

        # 聚合结果
        aggregated = {
            'loss_mean': np.mean([r['loss'] for r in results]),
            'loss_std': np.std([r['loss'] for r in results]),
            'accuracy_mean': np.mean([r['accuracy'] for r in results]),
            'accuracy_std': np.std([r['accuracy'] for r in results]),
            'runs': results,
        }

        return aggregated

    def _adapt(self, model, few_shot_data: List[Dict]):
        """在few-shot数据上fine-tune模型"""
        model.train()

        # 创建数据集和加载器
        dataset = SFTDataset(few_shot_data, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        # 优化器
        optimizer = AdamW(model.parameters(), lr=self.adaptation_lr)

        # Fine-tune
        for step in range(self.adaptation_steps):
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                loss_mask = batch['loss_mask'].to(self.device)

                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                # 计算masked loss
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                shift_mask = loss_mask[..., 1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                loss = (loss * shift_mask.view(-1)).sum() / (shift_mask.sum() + 1e-8)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        return model

    def _evaluate_model(self, model, test_data: List[Dict]) -> Dict:
        """评估模型在测试集上的性能"""
        model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        dataset = SFTDataset(test_data, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                loss_mask = batch['loss_mask'].to(self.device)

                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                # 计算loss
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                shift_mask = loss_mask[..., 1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                loss = (loss * shift_mask.view(-1)).sum() / (shift_mask.sum() + 1e-8)

                total_loss += loss.item()

                # 生成预测并检查正确性
                generated = self.generate_response(batch['prompt'][0])
                is_correct = self.check_correctness(
                    generated,
                    batch['response'][0]
                )
                if is_correct:
                    correct += 1
                total += 1

        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total if total > 0 else 0.0,
        }

    def generate_response(self, prompt: str, max_new_tokens=512) -> str:
        """生成模型的响应"""
        # 应用chat template
        messages = [{"role": "user", "content": prompt}]
        prompt_str = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # Tokenize
        inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding for consistency
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取response部分（去掉prompt）
        response = generated[len(prompt_str):].strip()

        return response

    def check_correctness(self, prediction: str, ground_truth: str) -> bool:
        """
        检查预测是否正确

        简化版本：提取最终答案并比较
        实际应用中可能需要更复杂的答案提取和比较逻辑
        """
        pred_answer = self.extract_answer(prediction)
        gt_answer = self.extract_answer(ground_truth)

        return self.normalize_answer(pred_answer) == self.normalize_answer(gt_answer)

    @staticmethod
    def extract_answer(text: str) -> str:
        """提取文本中的最终答案"""
        # 简单的答案提取逻辑
        # 查找常见的答案标记
        markers = ['答案是', '答案为', '因此', '所以', '答案：', '答案:']

        for marker in markers:
            if marker in text:
                answer = text.split(marker)[-1].strip()
                # 取第一句话或第一行
                answer = answer.split('\n')[0].split('。')[0].strip()
                return answer

        # 如果没有找到标记，返回最后一句话
        sentences = text.split('。')
        return sentences[-1].strip() if sentences else text

    @staticmethod
    def normalize_answer(answer: str) -> str:
        """规范化答案用于比较"""
        # 移除空格和标点
        import re
        answer = re.sub(r'[^\w\s]', '', answer)
        answer = answer.replace(' ', '').lower()
        return answer


def evaluate_learning_curves(
    evaluator: FewShotEvaluator,
    task_name: str,
    test_data: List[Dict],
    n_shots_list: List[int] = [0, 5, 10, 25, 50],
    n_runs: int = 3,
) -> Dict:
    """
    评估few-shot学习曲线

    Returns:
        {n_shots: {loss_mean, loss_std, accuracy_mean, accuracy_std}}
    """
    print(f"\n=== Evaluating Few-Shot Learning Curve: {task_name} ===")

    results = {}

    for n_shots in n_shots_list:
        print(f"\nEvaluating {n_shots}-shot...")

        if n_shots == 0:
            # Zero-shot
            result = evaluator.evaluate_zero_shot(test_data[:100])  # 限制测试集大小
            results[n_shots] = {
                'loss_mean': result['loss'],
                'loss_std': 0.0,
                'accuracy_mean': result['accuracy'],
                'accuracy_std': 0.0,
            }
        else:
            # Few-shot: 采样并适应
            # 为了减少随机性，重复n_runs次
            run_results = []

            for run in range(n_runs):
                # 从测试数据中采样n_shots个样本用于适应
                few_shot_indices = np.random.choice(len(test_data), size=n_shots, replace=False)
                few_shot_data = [test_data[i] for i in few_shot_indices]

                # 剩余数据用于评估
                eval_indices = [i for i in range(len(test_data)) if i not in few_shot_indices]
                eval_data = [test_data[i] for i in eval_indices[:100]]  # 限制评估集大小

                # 适应并评估
                result = evaluator.adapt_and_evaluate(
                    few_shot_data=few_shot_data,
                    test_data=eval_data,
                    n_runs=1,  # 内部只运行一次
                )

                run_results.append(result)

            # 聚合多次运行的结果
            results[n_shots] = {
                'loss_mean': np.mean([r['loss_mean'] for r in run_results]),
                'loss_std': np.std([r['loss_mean'] for r in run_results]),
                'accuracy_mean': np.mean([r['accuracy_mean'] for r in run_results]),
                'accuracy_std': np.std([r['accuracy_mean'] for r in run_results]),
            }

        print(f"  Loss: {results[n_shots]['loss_mean']:.4f} ± {results[n_shots]['loss_std']:.4f}")
        print(f"  Acc:  {results[n_shots]['accuracy_mean']:.2%} ± {results[n_shots]['accuracy_std']:.2%}")

    return results


def plot_learning_curves(
    results_dict: Dict[str, Dict],  # {model_name: {task_name: {n_shots: metrics}}}
    output_dir: Path,
):
    """绘制few-shot学习曲线"""
    print("\n=== Plotting Learning Curves ===")

    # 为每个任务创建一个图
    all_tasks = set()
    for model_results in results_dict.values():
        all_tasks.update(model_results.keys())

    for task in all_tasks:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for model_name, model_results in results_dict.items():
            if task not in model_results:
                continue

            task_results = model_results[task]
            n_shots_list = sorted(task_results.keys())

            # 提取数据
            losses = [task_results[n]['loss_mean'] for n in n_shots_list]
            loss_stds = [task_results[n]['loss_std'] for n in n_shots_list]
            accs = [task_results[n]['accuracy_mean'] * 100 for n in n_shots_list]  # 转换为百分比
            acc_stds = [task_results[n]['accuracy_std'] * 100 for n in n_shots_list]

            # 绘制Loss曲线
            ax1.plot(n_shots_list, losses, marker='o', label=model_name, linewidth=2)
            ax1.fill_between(
                n_shots_list,
                [l - s for l, s in zip(losses, loss_stds)],
                [l + s for l, s in zip(losses, loss_stds)],
                alpha=0.2
            )

            # 绘制Accuracy曲线
            ax2.plot(n_shots_list, accs, marker='o', label=model_name, linewidth=2)
            ax2.fill_between(
                n_shots_list,
                [a - s for a, s in zip(accs, acc_stds)],
                [a + s for a, s in zip(accs, acc_stds)],
                alpha=0.2
            )

        # 设置图表属性
        ax1.set_xlabel('Number of Few-Shot Examples', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title(f'{task} - Loss', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('Number of Few-Shot Examples', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title(f'{task} - Accuracy', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表
        output_path = output_dir / f"learning_curve_{task}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

        plt.close()


def compare_methods_statistically(
    fomaml_results: Dict,  # {task: {n_shots: metrics}}
    sft_results: Dict,
) -> Dict:
    """统计显著性检验"""
    print("\n=== Statistical Significance Testing ===")

    comparison = {}

    for task in fomaml_results.keys():
        if task not in sft_results:
            continue

        comparison[task] = {}

        for n_shots in fomaml_results[task].keys():
            if n_shots not in sft_results[task]:
                continue

            fomaml_acc = fomaml_results[task][n_shots]['accuracy_mean']
            sft_acc = sft_results[task][n_shots]['accuracy_mean']

            # 计算差异
            diff = fomaml_acc - sft_acc
            relative_improvement = (diff / sft_acc * 100) if sft_acc > 0 else 0

            comparison[task][n_shots] = {
                'fomaml_acc': fomaml_acc,
                'sft_acc': sft_acc,
                'difference': diff,
                'relative_improvement': relative_improvement,
            }

            print(f"\n{task} - {n_shots}-shot:")
            print(f"  FOMAML: {fomaml_acc:.2%}")
            print(f"  SFT:    {sft_acc:.2%}")
            print(f"  Diff:   {diff:.2%} ({relative_improvement:+.1f}%)")

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Few-Shot Learning Evaluation")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--model-type", type=str, choices=['fomaml', 'sft', 'base'],
                        required=True, help="Type of model")
    parser.add_argument("--tokenizer-path", type=str, default=None,
                        help="Path to tokenizer (default: same as model)")
    parser.add_argument("--data-dir", type=str, default="./data/math_science_meta/few_shot_eval",
                        help="Directory containing few-shot evaluation data")
    parser.add_argument("--eval-tasks", nargs='+', required=True,
                        help="Tasks to evaluate on")
    parser.add_argument("--n-shots", nargs='+', type=int, default=[0, 5, 10, 25, 50],
                        help="Few-shot sizes to evaluate")
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Number of runs for each few-shot size")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--adaptation-lr", type=float, default=1e-4,
                        help="Learning rate for few-shot adaptation")
    parser.add_argument("--adaptation-steps", type=int, default=100,
                        help="Number of steps for few-shot adaptation")

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型和tokenizer
    print(f"\n=== Loading {args.model_type.upper()} Model ===")
    print(f"Model path: {args.model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )

    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 创建评估器
    evaluator = FewShotEvaluator(
        model=model,
        tokenizer=tokenizer,
        adaptation_lr=args.adaptation_lr,
        adaptation_steps=args.adaptation_steps,
    )

    # 评估每个任务
    all_results = {}

    data_dir = Path(args.data_dir)

    for task_name in args.eval_tasks:
        # 加载测试数据
        test_file = data_dir / f"{task_name}_test.parquet"

        if not test_file.exists():
            print(f"\nWarning: Test file not found: {test_file}")
            continue

        test_df = pd.read_parquet(test_file)
        test_data = test_df.to_dict('records')

        print(f"\nLoaded {len(test_data)} test samples for {task_name}")

        # 评估学习曲线
        task_results = evaluate_learning_curves(
            evaluator=evaluator,
            task_name=task_name,
            test_data=test_data,
            n_shots_list=args.n_shots,
            n_runs=args.n_runs,
        )

        all_results[task_name] = task_results

    # 保存结果
    results_file = output_dir / f"{args.model_type}_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # 绘制学习曲线
    plot_learning_curves(
        results_dict={args.model_type: all_results},
        output_dir=output_dir,
    )

    print("\n=== Evaluation Complete ===")


if __name__ == "__main__":
    main()
