"""
META-LORA Trainer Implementation

基于论文: MeTA-LoRA: Data-Efficient Multi-Task Fine-Tuning for Large Language Models
arXiv: 2510.11598

核心思想:
1. Stage 1: 每个任务用少量样本(100个)快速适应，只更新LoRA参数
2. Stage 2: 聚合多任务梯度更新shared LoRA，促进知识转移

关键优势:
- Base model完全冻结，极低内存和计算成本
- 只优化LoRA参数(~0.1-1%模型参数)
- 100样本/任务即可达到全数据LoRA性能

与FOMAML的区别:
- FOMAML: 优化全模型参数
- META-LORA: 只优化LoRA参数，base model冻结
"""

import os
import logging
import time
from copy import deepcopy
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tensordict import TensorDict
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, PeftModel

logger = logging.getLogger(__file__)


class MetaLoRATrainer:
    """
    META-LORA Training: 两阶段优化框架

    只优化LoRA参数，base model完全冻结
    """

    def __init__(
        self,
        config,
        model: nn.Module,
        tokenizer,
        task_datasets: Dict[str, Dict[str, Dataset]],  # {task: {'train': ds, 'val': ds}}
        device_mesh=None,
    ):
        self.config = config
        self.base_model = model  # 将被冻结
        self.tokenizer = tokenizer
        self.task_datasets = task_datasets
        self.device_mesh = device_mesh

        # META-LORA超参数
        self.inner_lr = config.meta.inner_lr  # Stage 1学习率
        self.meta_lr = config.meta.meta_lr  # Stage 2学习率
        self.num_inner_steps = config.meta.num_inner_steps  # Stage 1适应步数
        self.inner_batch_size = config.meta.inner_batch_size
        self.meta_batch_size = config.meta.meta_batch_size  # 每次更新几个任务

        self.device_name = config.trainer.device

        # LoRA配置
        self.lora_config = LoraConfig(
            r=config.model.lora_rank,
            lora_alpha=config.model.lora_alpha,
            target_modules=config.model.target_modules,
            lora_dropout=config.model.get('lora_dropout', 0.0),
            bias="none",
            task_type="CAUSAL_LM",
        )

        # 初始化shared LoRA
        self._init_shared_lora()

        # 构建任务数据加载器
        self._build_task_dataloaders()

        # Meta-optimizer (用于更新shared LoRA)
        self.meta_optimizer = torch.optim.AdamW(
            self.get_lora_parameters(),
            lr=self.meta_lr,
            weight_decay=config.optim.weight_decay,
        )

        logger.info(f"Initialized META-LORA Trainer with {len(self.task_datasets)} tasks")
        logger.info(f"LoRA rank: {self.lora_config.r}, alpha: {self.lora_config.lora_alpha}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.get_lora_parameters()):,}")

    def _init_shared_lora(self):
        """
        初始化shared LoRA

        关键：base model完全冻结
        """
        # 冻结base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        logger.info("Base model frozen - will only train LoRA parameters")

        # 添加LoRA适配器
        self.shared_lora_model = get_peft_model(self.base_model, self.lora_config)

        # 验证只有LoRA参数可训练
        trainable_params = sum(p.numel() for p in self.shared_lora_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.shared_lora_model.parameters())

        logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} "
                    f"({100 * trainable_params / total_params:.2f}%)")

    def get_lora_parameters(self):
        """获取LoRA参数（用于优化）"""
        return [p for p in self.shared_lora_model.parameters() if p.requires_grad]

    def _build_task_dataloaders(self):
        """构建任务数据加载器"""
        self.task_loaders = {}

        for task_name, datasets in self.task_datasets.items():
            train_ds = datasets['train']
            val_ds = datasets['val']

            # 数据并行配置
            if self.device_mesh is not None:
                dp_rank = self.device_mesh.get_rank()
                dp_size = self.device_mesh.size()
            else:
                dp_rank = 0
                dp_size = 1

            # Train loader (用于Stage 1适应)
            train_sampler = DistributedSampler(
                train_ds, shuffle=True, num_replicas=dp_size, rank=dp_rank, drop_last=True
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=self.inner_batch_size,
                sampler=train_sampler,
                num_workers=4,
                pin_memory=True,
            )

            # Val loader (用于Stage 2元更新)
            val_sampler = DistributedSampler(
                val_ds, shuffle=False, num_replicas=dp_size, rank=dp_rank, drop_last=True
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=self.inner_batch_size,
                sampler=val_sampler,
                num_workers=4,
                pin_memory=True,
            )

            self.task_loaders[task_name] = {
                'train': train_loader,
                'val': val_loader,
                'train_sampler': train_sampler,
                'val_sampler': val_sampler,
            }

    def _compute_sft_loss(self, batch: TensorDict, model: nn.Module) -> torch.Tensor:
        """计算SFT损失（与verl兼容）"""
        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)
        position_ids = batch["position_ids"].to(self.device_name)
        loss_mask = batch["loss_mask"][:, 1:].reshape(-1).to(self.device_name)

        loss_fct = nn.CrossEntropyLoss(reduction='none')

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False
            )
            logits = output.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            shift_logits = shift_logits.view(-1, model.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)

            loss = loss_fct(shift_logits, shift_labels)
            loss = loss * loss_mask.to(loss.device)

            valid_tokens = loss_mask.sum() + 1e-8
            loss = loss.sum() / valid_tokens

        return loss

    def stage1_task_adaptation(
        self,
        task_name: str,
    ) -> Dict:
        """
        Stage 1: Task-Specific Adaptation

        在少量训练样本上快速适应LoRA参数

        Args:
            task_name: 任务名称

        Returns:
            adapted_lora_state: 适应后的LoRA state dict
            metrics: 训练指标
        """
        # 克隆shared LoRA参数作为起点
        adapted_model = deepcopy(self.shared_lora_model)
        adapted_model.train()

        # 创建task-specific优化器（只优化LoRA参数）
        optimizer = torch.optim.AdamW(
            [p for p in adapted_model.parameters() if p.requires_grad],
            lr=self.inner_lr,
        )

        # 获取任务的训练数据
        train_loader = self.task_loaders[task_name]['train']
        train_iter = iter(train_loader)

        losses = []

        # 执行K步适应
        for step in range(self.num_inner_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = TensorDict(batch, batch_size=len(batch['input_ids']))

            # Forward和backward
            optimizer.zero_grad()
            loss = self._compute_sft_loss(batch, adapted_model)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                [p for p in adapted_model.parameters() if p.requires_grad],
                max_norm=self.config.optim.clip_grad
            )

            optimizer.step()
            losses.append(loss.item())

        # 提取适应后的LoRA参数
        adapted_lora_state = self._extract_lora_state(adapted_model)

        return {
            'lora_state': adapted_lora_state,
            'avg_loss': sum(losses) / len(losses),
            'final_loss': losses[-1],
        }

    def stage2_shared_lora_update(
        self,
        task_batch: List[str],
        adapted_states: Dict[str, Dict],
    ) -> Dict[str, float]:
        """
        Stage 2: Shared LoRA Update

        通过聚合多任务梯度更新shared LoRA

        Args:
            task_batch: 任务列表
            adapted_states: 各任务的adapted LoRA states

        Returns:
            metrics: 更新指标
        """
        self.meta_optimizer.zero_grad()

        meta_loss = 0.0
        task_metrics = {}

        # 对每个任务计算meta gradient
        for task_name in task_batch:
            # 加载该任务的adapted LoRA
            self._load_lora_state(self.shared_lora_model, adapted_states[task_name]['lora_state'])

            # 在验证集上计算损失
            val_loader = self.task_loaders[task_name]['val']
            val_iter = iter(val_loader)

            try:
                val_batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                val_batch = next(val_iter)

            val_batch = TensorDict(val_batch, batch_size=len(val_batch['input_ids']))

            # 计算验证损失
            val_loss = self._compute_sft_loss(val_batch, self.shared_lora_model)

            # 累积meta loss
            meta_loss += val_loss / len(task_batch)

            task_metrics[f'{task_name}/val_loss'] = val_loss.item()
            task_metrics[f'{task_name}/train_loss'] = adapted_states[task_name]['final_loss']

        # 反向传播计算meta gradients
        meta_loss.backward()

        # 梯度裁剪
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.get_lora_parameters(),
            max_norm=self.config.optim.clip_grad
        )

        # 更新shared LoRA
        self.meta_optimizer.step()

        metrics = {
            'meta/loss': meta_loss.item(),
            'meta/grad_norm': grad_norm.item(),
            **task_metrics
        }

        return metrics

    def _extract_lora_state(self, model: PeftModel) -> Dict:
        """提取LoRA参数state dict"""
        lora_state = {}
        for name, param in model.named_parameters():
            if 'lora' in name.lower() and param.requires_grad:
                lora_state[name] = param.data.clone()
        return lora_state

    def _load_lora_state(self, model: PeftModel, lora_state: Dict):
        """加载LoRA参数"""
        for name, param in model.named_parameters():
            if name in lora_state:
                param.data = lora_state[name].clone()

    def meta_train_step(self, task_batch: List[str]) -> Dict[str, float]:
        """
        完整的元训练步骤：Stage 1 + Stage 2

        Args:
            task_batch: 本次迭代使用的任务列表

        Returns:
            metrics: 训练指标
        """
        # ===== Stage 1: Task-Specific Adaptation =====
        adapted_states = {}

        for task_name in task_batch:
            result = self.stage1_task_adaptation(task_name)
            adapted_states[task_name] = result

        # ===== Stage 2: Shared LoRA Update =====
        metrics = self.stage2_shared_lora_update(task_batch, adapted_states)

        return metrics

    def fit(self):
        """主训练循环"""
        rank = self.device_mesh.get_rank() if self.device_mesh else 0

        # 初始化tracking
        if rank == 0:
            from verl.utils.tracking import Tracking
            from omegaconf import OmegaConf

            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

        total_steps = self.config.trainer.total_steps
        task_names = list(self.task_datasets.keys())

        logger.info(f"Starting META-LORA training for {total_steps} steps")
        logger.info(f"Tasks: {task_names}")
        logger.info(f"Meta batch size: {self.meta_batch_size}")

        global_step = 0

        for step in tqdm(range(total_steps), desc="META-LORA training", disable=rank != 0):
            # 采样任务批次
            task_indices = torch.randperm(len(task_names))[:self.meta_batch_size].tolist()
            task_batch = [task_names[i] for i in task_indices]

            # 执行元训练步骤
            metrics = self.meta_train_step(task_batch)

            global_step += 1

            # 记录
            if rank == 0:
                tracking.log(data=metrics, step=global_step)

            # 保存checkpoint
            if global_step % self.config.trainer.save_freq == 0:
                self.save_checkpoint(step=global_step)

            # 评估
            if global_step % self.config.trainer.test_freq == 0:
                eval_metrics = self.evaluate()
                if rank == 0:
                    tracking.log(data=eval_metrics, step=global_step)

        logger.info("META-LORA training completed!")

    def evaluate(self) -> Dict[str, float]:
        """评估shared LoRA在所有任务上的性能"""
        self.shared_lora_model.eval()
        eval_metrics = {}

        with torch.no_grad():
            for task_name in self.task_datasets.keys():
                val_loader = self.task_loaders[task_name]['val']
                task_losses = []

                for i, batch in enumerate(val_loader):
                    if i >= 10:  # 限制评估样本数
                        break

                    batch = TensorDict(batch, batch_size=len(batch['input_ids']))
                    loss = self._compute_sft_loss(batch, self.shared_lora_model)
                    task_losses.append(loss.item())

                avg_loss = sum(task_losses) / len(task_losses) if task_losses else 0.0
                eval_metrics[f'eval/{task_name}/loss'] = avg_loss

        self.shared_lora_model.train()
        return eval_metrics

    def save_checkpoint(self, step: int):
        """保存checkpoint（只保存LoRA参数）"""
        if self.device_mesh is None or self.device_mesh.get_rank() == 0:
            checkpoint_path = os.path.join(
                self.config.trainer.default_local_dir,
                f"meta_lora_checkpoint_step_{step}.pt"
            )

            # 只保存LoRA参数（非常小！）
            lora_state = self._extract_lora_state(self.shared_lora_model)

            torch.save({
                'step': step,
                'lora_state_dict': lora_state,
                'meta_optimizer_state_dict': self.meta_optimizer.state_dict(),
                'config': self.config,
            }, checkpoint_path)

            logger.info(f"Saved META-LORA checkpoint to {checkpoint_path}")
            logger.info(f"Checkpoint size: {os.path.getsize(checkpoint_path) / 1024 / 1024:.2f} MB")


def create_meta_lora_task_datasets(config, tokenizer) -> Dict[str, Dict[str, Dataset]]:
    """
    创建META-LORA任务数据集

    每个任务需要：
    - train: 少量训练样本（e.g., 100个）用于Stage 1
    - val: 验证样本用于Stage 2

    Returns:
        {
            'task1': {'train': train_ds, 'val': val_ds},
            'task2': {'train': train_ds, 'val': val_ds},
            ...
        }
    """
    from verl.utils.dataset import SFTDataset

    task_datasets = {}

    for task_config in config.meta.tasks:
        task_name = task_config.name

        # Train dataset (Stage 1用，限制样本数)
        train_dataset = SFTDataset(
            parquet_files=task_config.train_files,
            tokenizer=tokenizer,
            config=config.data,
            max_samples=task_config.get('train_max_samples', 100),  # 默认100样本
        )

        # Val dataset (Stage 2用)
        val_dataset = SFTDataset(
            parquet_files=task_config.val_files,
            tokenizer=tokenizer,
            config=config.data,
            max_samples=task_config.get('val_max_samples', 200),
        )

        task_datasets[task_name] = {
            'train': train_dataset,
            'val': val_dataset,
        }

    return task_datasets


def run_meta_lora(config):
    """Main entry point for META-LORA training"""
    from verl.utils.distributed import initialize_global_process_group, destroy_global_process_group
    from verl.utils import hf_tokenizer
    from verl.utils.fs import copy_to_local
    from verl.utils.device import get_device_name
    from torch.distributed.device_mesh import init_device_mesh
    from transformers import AutoModelForCausalLM, AutoConfig

    # 初始化分布式
    local_rank, rank, world_size = initialize_global_process_group()
    device_name = get_device_name()
    device_mesh = init_device_mesh(
        device_type=device_name,
        mesh_shape=(world_size,),
        mesh_dim_names=("dp",)
    )

    # 加载tokenizer
    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)

    # 加载base model（将被冻结）
    model_config = AutoConfig.from_pretrained(
        local_model_path,
        trust_remote_code=config.model.trust_remote_code
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        config=model_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=config.model.trust_remote_code,
    ).to(device_name)

    # 创建任务数据集
    task_datasets = create_meta_lora_task_datasets(config, tokenizer)

    # 初始化META-LORA训练器
    trainer = MetaLoRATrainer(
        config=config,
        model=base_model,
        tokenizer=tokenizer,
        task_datasets=task_datasets,
        device_mesh=device_mesh,
    )

    # 训练
    trainer.fit()

    # 清理
    destroy_global_process_group()


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path="config", config_name="meta_lora_trainer", version_base=None)
    def main(config: DictConfig):
        run_meta_lora(config)

    main()
