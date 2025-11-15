"""
Reptile-SFT Trainer: A Simplified Alternative to MAML

Reptile is a simpler meta-learning algorithm that is easier to implement
and more memory-efficient than MAML, while achieving similar performance.

Key differences from MAML:
1. No separate support/query split needed
2. No meta-loss computation
3. Simple parameter interpolation instead of meta-gradients
4. Lower memory overhead

Algorithm:
    for each iteration:
        for each task:
            θ_old = θ
            Train on task for K steps → θ_task
            θ = θ + ε(θ_task - θ_old)  # Move towards task-adapted parameters
"""

import os
import logging
import time
from copy import deepcopy
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tensordict import TensorDict
from tqdm import tqdm

logger = logging.getLogger(__file__)


class ReptileSFTTrainer:
    """
    Reptile Meta-Learning SFT Trainer

    Simpler alternative to MAML with comparable performance.

    Args:
        config: Training configuration
        model: The base language model
        tokenizer: Tokenizer for the model
        task_datasets: Dictionary mapping task names to datasets
        device_mesh: FSDP device mesh (optional)
    """

    def __init__(
        self,
        config,
        model: nn.Module,
        tokenizer,
        task_datasets: Dict[str, Dataset],  # {task_name: dataset}
        device_mesh=None,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.task_datasets = task_datasets
        self.device_mesh = device_mesh

        # Reptile hyperparameters
        self.inner_lr = config.meta.inner_lr  # Learning rate for task adaptation
        self.outer_stepsize = config.meta.outer_stepsize  # ε for parameter interpolation
        self.num_inner_steps = config.meta.num_inner_steps  # K steps per task
        self.batch_size = config.meta.batch_size

        self.device_name = config.trainer.device

        # Build task dataloaders
        self._build_task_dataloaders()

        # Build inner optimizer (for task-specific training)
        self.inner_optimizer_class = torch.optim.AdamW
        self.inner_optimizer_kwargs = {
            'lr': self.inner_lr,
            'weight_decay': config.optim.weight_decay,
            'betas': config.optim.betas,
        }

        logger.info(f"Initialized Reptile-SFT Trainer with {len(self.task_datasets)} tasks")

    def _build_task_dataloaders(self):
        """Build dataloaders for each task"""
        self.task_loaders = {}

        for task_name, dataset in self.task_datasets.items():
            # Get data parallel rank and size
            if self.device_mesh is not None:
                dp_rank = self.device_mesh.get_rank()
                dp_size = self.device_mesh.size()
            else:
                dp_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                dp_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

            sampler = DistributedSampler(
                dataset, shuffle=True, num_replicas=dp_size, rank=dp_rank, drop_last=True
            )

            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=4,
                pin_memory=True,
            )

            self.task_loaders[task_name] = {
                'loader': loader,
                'sampler': sampler,
            }

    def _compute_sft_loss(self, batch: TensorDict, model: nn.Module) -> torch.Tensor:
        """Compute SFT loss for a batch"""
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

    def _adapt_to_task(self, task_name: str) -> Dict[str, torch.Tensor]:
        """
        Adapt model to a specific task for K steps

        Returns:
            Dictionary of adapted parameters
        """
        # Create task-specific optimizer
        inner_optimizer = self.inner_optimizer_class(
            self.model.parameters(),
            **self.inner_optimizer_kwargs
        )

        # Get task dataloader
        task_loader = self.task_loaders[task_name]['loader']
        task_iter = iter(task_loader)

        # Train for K steps on this task
        task_losses = []
        self.model.train()

        for step in range(self.num_inner_steps):
            try:
                batch = next(task_iter)
            except StopIteration:
                # Restart if we run out of data
                task_iter = iter(task_loader)
                batch = next(task_iter)

            batch = TensorDict(batch, batch_size=len(batch['input_ids']))

            # Zero gradients
            inner_optimizer.zero_grad()

            # Compute loss
            loss = self._compute_sft_loss(batch, self.model)
            task_losses.append(loss.item())

            # Backward and step
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.optim.clip_grad
            )

            inner_optimizer.step()

        # Return adapted parameters
        adapted_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }

        return {
            'params': adapted_params,
            'avg_loss': sum(task_losses) / len(task_losses),
        }

    def _reptile_update(self, task_name: str) -> Dict[str, float]:
        """
        Perform one Reptile update on a task

        Algorithm:
            1. Save current parameters θ_old
            2. Train on task for K steps → θ_task
            3. Update: θ = θ_old + ε(θ_task - θ_old)

        Returns:
            Dictionary of metrics
        """
        # Save current parameters
        old_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }

        # Adapt to task
        result = self._adapt_to_task(task_name)
        adapted_params = result['params']

        # Reptile update: interpolate between old and adapted parameters
        # θ = θ_old + ε(θ_task - θ_old)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                # Compute direction: θ_task - θ_old
                direction = adapted_params[name] - old_params[name]

                # Update parameter
                param.data = old_params[name] + self.outer_stepsize * direction

        metrics = {
            f'{task_name}/loss': result['avg_loss'],
            'meta/outer_stepsize': self.outer_stepsize,
        }

        return metrics

    def fit(self):
        """
        Main Reptile training loop

        For each iteration:
            1. Sample a task
            2. Adapt to task for K steps
            3. Interpolate parameters
        """
        rank = self.device_mesh.get_rank() if self.device_mesh else 0

        # Initialize tracking
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

        logger.info(f"Starting Reptile meta-training for {total_steps} steps")
        logger.info(f"Tasks: {task_names}")

        global_step = 0

        # Optionally decay outer stepsize over training
        initial_outer_stepsize = self.outer_stepsize

        for step in tqdm(range(total_steps), desc="Reptile training", disable=rank != 0):
            # Sample a task uniformly
            task_idx = torch.randint(0, len(task_names), (1,)).item()
            task_name = task_names[task_idx]

            # Perform Reptile update
            metrics = self._reptile_update(task_name)

            global_step += 1

            # Optional: decay outer stepsize
            if self.config.meta.get('decay_outer_stepsize', False):
                decay_rate = self.config.meta.get('stepsize_decay_rate', 0.99)
                self.outer_stepsize = initial_outer_stepsize * (decay_rate ** step)
                metrics['meta/outer_stepsize'] = self.outer_stepsize

            # Log metrics
            if rank == 0:
                tracking.log(data=metrics, step=global_step)

            # Save checkpoint
            if global_step % self.config.trainer.save_freq == 0:
                self.save_checkpoint(step=global_step)

            # Evaluate
            if global_step % self.config.trainer.test_freq == 0:
                eval_metrics = self.evaluate()
                if rank == 0:
                    tracking.log(data=eval_metrics, step=global_step)

        logger.info("Reptile meta-training completed!")

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate meta-learned model on all tasks

        For each task:
            Compute loss without adaptation (zero-shot)
        """
        self.model.eval()
        eval_metrics = {}

        with torch.no_grad():
            for task_name in self.task_datasets.keys():
                task_loader = self.task_loaders[task_name]['loader']
                task_losses = []

                # Evaluate on a few batches
                for i, batch in enumerate(task_loader):
                    if i >= 10:  # Evaluate on 10 batches
                        break

                    batch = TensorDict(batch, batch_size=len(batch['input_ids']))
                    loss = self._compute_sft_loss(batch, self.model)
                    task_losses.append(loss.item())

                avg_loss = sum(task_losses) / len(task_losses) if task_losses else 0.0
                eval_metrics[f'eval/{task_name}/loss'] = avg_loss

        self.model.train()
        return eval_metrics

    def save_checkpoint(self, step: int):
        """Save meta-model checkpoint"""
        if self.device_mesh is None or self.device_mesh.get_rank() == 0:
            checkpoint_path = os.path.join(
                self.config.trainer.default_local_dir,
                f"reptile_checkpoint_step_{step}.pt"
            )

            torch.save({
                'step': step,
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'outer_stepsize': self.outer_stepsize,
            }, checkpoint_path)

            logger.info(f"Saved checkpoint to {checkpoint_path}")


def create_task_datasets_for_reptile(config, tokenizer) -> Dict[str, Dataset]:
    """
    Create task-specific datasets for Reptile

    Unlike MAML, Reptile doesn't need support/query split.
    Each task just needs one dataset.

    Returns:
        Dictionary mapping task names to datasets
    """
    from verl.utils.dataset import SFTDataset

    task_datasets = {}

    for task_config in config.meta.tasks:
        task_name = task_config.name

        # Combine support and query files if they exist
        data_files = []
        if 'support_files' in task_config:
            data_files.extend(task_config.support_files)
        if 'query_files' in task_config:
            data_files.extend(task_config.query_files)
        if 'data_files' in task_config:
            data_files.extend(task_config.data_files)

        dataset = SFTDataset(
            parquet_files=data_files,
            tokenizer=tokenizer,
            config=config.data,
            max_samples=task_config.get('max_samples', -1)
        )

        task_datasets[task_name] = dataset

    return task_datasets


def run_reptile_sft(config):
    """Main entry point for Reptile-SFT training"""
    from verl.utils.distributed import initialize_global_process_group, destroy_global_process_group
    from verl.utils import hf_tokenizer
    from verl.utils.fs import copy_to_local
    from verl.utils.device import get_device_name
    from torch.distributed.device_mesh import init_device_mesh
    from transformers import AutoModelForCausalLM, AutoConfig

    # Initialize distributed
    local_rank, rank, world_size = initialize_global_process_group()
    device_name = get_device_name()
    device_mesh = init_device_mesh(
        device_type=device_name,
        mesh_shape=(world_size,),
        mesh_dim_names=("fsdp",)
    )

    # Load tokenizer
    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)

    # Load base model
    model_config = AutoConfig.from_pretrained(
        local_model_path,
        trust_remote_code=config.model.trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        config=model_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=config.model.trust_remote_code,
    )

    # Wrap with FSDP if needed
    if config.model.use_fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from verl.utils.fsdp_utils import get_fsdp_wrap_policy

        auto_wrap_policy = get_fsdp_wrap_policy(model, config=config.model.fsdp_config.wrap_policy)
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            device_mesh=device_mesh,
            use_orig_params=True,
        )

    # Create task datasets
    task_datasets = create_task_datasets_for_reptile(config, tokenizer)

    # Initialize trainer
    trainer = ReptileSFTTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        task_datasets=task_datasets,
        device_mesh=device_mesh,
    )

    # Train
    trainer.fit()

    # Cleanup
    destroy_global_process_group()


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path="config", config_name="reptile_sft_trainer", version_base=None)
    def main(config: DictConfig):
        run_reptile_sft(config)

    main()
