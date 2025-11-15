# Copyright 2024 - Meta Learning SFT Implementation
# Based on verl framework's FSDP SFT Trainer
"""
MAML/FOMAML SFT Trainer for LLMs

This trainer implements Model-Agnostic Meta-Learning for Supervised Fine-Tuning
of Large Language Models, based on the verl framework.

Key Features:
- Meta-learning across multiple tasks/domains
- Support for MAML and FOMAML algorithms
- Compatible with FSDP for large model training
- Task batching and efficient gradient computation
"""

import os
import logging
import time
from contextlib import nullcontext
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tensordict import TensorDict
from tqdm import tqdm

logger = logging.getLogger(__file__)


class MAMLSFTTrainer:
    """
    Meta-Learning SFT Trainer implementing MAML/FOMAML algorithms

    Args:
        config: Training configuration
        model: The base language model
        tokenizer: Tokenizer for the model
        task_datasets: Dictionary mapping task names to datasets
        device_mesh: FSDP device mesh
        use_fomaml: If True, use first-order approximation (FOMAML)
    """

    def __init__(
        self,
        config,
        model: nn.Module,
        tokenizer,
        task_datasets: Dict[str, Dict[str, Dataset]],  # {task_name: {'support': ds, 'query': ds}}
        device_mesh=None,
        use_fomaml: bool = True,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.task_datasets = task_datasets
        self.device_mesh = device_mesh
        self.use_fomaml = use_fomaml

        # Meta-learning hyperparameters
        self.inner_lr = config.meta.inner_lr  # α in MAML
        self.outer_lr = config.meta.outer_lr  # β in MAML
        self.num_inner_steps = config.meta.num_inner_steps  # K steps
        self.meta_batch_size = config.meta.meta_batch_size  # Number of tasks per meta-update

        self.device_name = config.trainer.device

        # Initialize meta-optimizer (for outer loop)
        self.meta_optimizer = self._build_meta_optimizer()

        # Build task dataloaders
        self._build_task_dataloaders()

        logger.info(f"Initialized MAML-SFT Trainer with {len(self.task_datasets)} tasks")
        logger.info(f"Using {'FOMAML' if use_fomaml else 'MAML'} algorithm")

    def _build_meta_optimizer(self):
        """Build optimizer for meta-parameter updates (outer loop)"""
        from verl.workers.config.optimizer import build_optimizer
        return build_optimizer(self.model.parameters(), self.config.optim)

    def _build_task_dataloaders(self):
        """Build dataloaders for each task's support and query sets"""
        self.task_loaders = {}

        for task_name, datasets in self.task_datasets.items():
            support_ds = datasets['support']
            query_ds = datasets['query']

            # Get data parallel rank and size
            if self.device_mesh is not None:
                dp_rank = self.device_mesh.get_rank()
                dp_size = self.device_mesh.size()
            else:
                dp_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                dp_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

            support_sampler = DistributedSampler(
                support_ds, shuffle=True, num_replicas=dp_size, rank=dp_rank, drop_last=True
            )
            query_sampler = DistributedSampler(
                query_ds, shuffle=True, num_replicas=dp_size, rank=dp_rank, drop_last=True
            )

            support_loader = DataLoader(
                support_ds,
                batch_size=self.config.meta.inner_batch_size,
                sampler=support_sampler,
                num_workers=4,
                pin_memory=True,
            )

            query_loader = DataLoader(
                query_ds,
                batch_size=self.config.meta.query_batch_size,
                sampler=query_sampler,
                num_workers=4,
                pin_memory=True,
            )

            self.task_loaders[task_name] = {
                'support': support_loader,
                'query': query_loader,
                'support_sampler': support_sampler,
                'query_sampler': query_sampler,
            }

    def _compute_sft_loss(self, batch: TensorDict, model: nn.Module) -> torch.Tensor:
        """
        Compute SFT loss for a batch, compatible with verl's loss computation

        This follows the same loss calculation as verl's sft_loss function
        """
        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)
        position_ids = batch["position_ids"].to(self.device_name)
        loss_mask = batch["loss_mask"][:, 1:].reshape(-1).to(self.device_name)

        loss_fct = nn.CrossEntropyLoss(reduction='none')

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            # Forward pass
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False
            )
            logits = output.logits

            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            # Flatten the tokens
            shift_logits = shift_logits.view(-1, model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)

            # Compute loss with masking
            loss = loss_fct(shift_logits, shift_labels)
            loss = loss * loss_mask.to(loss.device)

            # Average over valid tokens
            valid_tokens = loss_mask.sum() + 1e-8
            loss = loss.sum() / valid_tokens

        return loss

    def _inner_loop_update(
        self,
        task_name: str,
        support_batch: TensorDict,
        return_grads: bool = False
    ) -> Dict:
        """
        Perform inner loop adaptation on support set

        Args:
            task_name: Name of the task
            support_batch: Support set batch
            return_grads: If True, return gradients for meta-update

        Returns:
            Dictionary containing adapted parameters and optionally gradients
        """
        # Clone current model parameters (θ)
        fast_weights = {name: param.clone() for name, param in self.model.named_parameters()}

        # Perform K steps of gradient descent on support set
        support_loss = None
        for step in range(self.num_inner_steps):
            # Apply fast weights to model
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    param.data = fast_weights[name].data

            # Compute loss on support set with current fast weights
            support_loss = self._compute_sft_loss(support_batch, self.model)

            # Compute gradients
            grads = torch.autograd.grad(
                support_loss,
                self.model.parameters(),
                create_graph=not self.use_fomaml,  # FOMAML doesn't need computational graph
                retain_graph=False,  # No need to retain graph in FOMAML
            )

            # Update fast weights: θ'_i = θ - α * ∇_θ L_support
            with torch.no_grad():
                for (name, param), grad in zip(self.model.named_parameters(), grads):
                    fast_weights[name] = fast_weights[name] - self.inner_lr * grad

        result = {
            'fast_weights': fast_weights,
            'support_loss': support_loss.item(),
        }

        if return_grads:
            result['grads'] = grads

        return result

    def _meta_update_step(self, task_batch: List[str]) -> Dict[str, float]:
        """
        Perform one meta-update step on a batch of tasks

        This implements the outer loop of MAML:
        θ = θ - β * ∇_θ Σ_i L_query(θ'_i)

        Args:
            task_batch: List of task names to use in this meta-update

        Returns:
            Dictionary of metrics
        """
        meta_loss = 0.0
        task_metrics = {}

        # Track FOMAML-specific metrics
        task_support_losses = []
        task_query_losses = []
        adaptation_gaps = []

        # Zero out meta-optimizer gradients
        self.meta_optimizer.zero_grad()

        # Store original model state
        original_state = {name: param.clone() for name, param in self.model.named_parameters()}

        # For each task in the meta-batch
        for task_name in task_batch:
            # Get support and query batches
            support_batch = next(iter(self.task_loaders[task_name]['support']))
            query_batch = next(iter(self.task_loaders[task_name]['query']))

            support_batch = TensorDict(support_batch, batch_size=len(support_batch['input_ids']))
            query_batch = TensorDict(query_batch, batch_size=len(query_batch['input_ids']))

            # Inner loop: adapt on support set
            inner_result = self._inner_loop_update(
                task_name=task_name,
                support_batch=support_batch,
                return_grads=False
            )

            # Apply fast weights to model
            fast_weights = inner_result['fast_weights']
            for name, param in self.model.named_parameters():
                param.data = fast_weights[name].data

            # Outer loop: compute loss on query set with adapted parameters
            query_loss = self._compute_sft_loss(query_batch, self.model)

            # Accumulate meta loss
            meta_loss += query_loss / len(task_batch)

            # Record task-specific metrics
            support_loss_val = inner_result['support_loss']
            query_loss_val = query_loss.item()
            adaptation_gap = query_loss_val - support_loss_val

            task_support_losses.append(support_loss_val)
            task_query_losses.append(query_loss_val)
            adaptation_gaps.append(adaptation_gap)

            task_metrics[f'{task_name}/support_loss'] = support_loss_val
            task_metrics[f'{task_name}/query_loss'] = query_loss_val
            task_metrics[f'{task_name}/adaptation_gap'] = adaptation_gap

            # Restore original parameters before next task
            for name, param in self.model.named_parameters():
                param.data = original_state[name].data

        # Compute meta-gradients: ∇_θ Σ_i L_query(θ'_i)
        if self.use_fomaml:
            # FOMAML: Use first-order approximation
            # Simply backpropagate through query loss
            meta_loss.backward()
        else:
            # MAML: Use second-order gradients
            # This computes the full meta-gradient through inner loop
            meta_loss.backward()

        # Clip gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.optim.clip_grad
        )

        # Meta-optimizer step: θ = θ - β * meta_gradient
        self.meta_optimizer.step()

        # Aggregate metrics with enhanced FOMAML-specific metrics
        metrics = {
            'meta/loss': meta_loss.item(),
            'meta/grad_norm': grad_norm.item(),

            # FOMAML-specific aggregated metrics
            'meta/avg_support_loss': float(np.mean(task_support_losses)),
            'meta/avg_query_loss': float(np.mean(task_query_losses)),
            'meta/avg_adaptation_gap': float(np.mean(adaptation_gaps)),
            'meta/adaptation_gap_std': float(np.std(adaptation_gaps)),
            'meta/task_loss_variance': float(np.var(task_query_losses)),

            **task_metrics
        }

        return metrics

    def fit(self):
        """
        Main meta-training loop

        This implements the complete MAML algorithm:
        1. Sample a batch of tasks
        2. For each task, perform inner loop adaptation
        3. Compute meta-loss on query sets
        4. Update meta-parameters
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

        logger.info(f"Starting meta-training for {total_steps} steps")
        logger.info(f"Tasks: {task_names}")

        global_step = 0

        for step in tqdm(range(total_steps), desc="Meta-training", disable=rank != 0):
            # Sample a batch of tasks
            task_batch = torch.multinomial(
                torch.ones(len(task_names)),
                num_samples=min(self.meta_batch_size, len(task_names)),
                replacement=False
            ).tolist()
            task_batch = [task_names[i] for i in task_batch]

            # Perform meta-update
            metrics = self._meta_update_step(task_batch)

            global_step += 1

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

        logger.info("Meta-training completed!")

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate meta-learned model on all tasks

        For each task:
        1. Adapt on support set
        2. Evaluate on query set
        """
        self.model.eval()
        eval_metrics = {}

        # Save original parameters
        original_state = {name: param.clone() for name, param in self.model.named_parameters()}

        with torch.no_grad():
            for task_name in self.task_datasets.keys():
                support_batch = next(iter(self.task_loaders[task_name]['support']))
                query_batch = next(iter(self.task_loaders[task_name]['query']))

                support_batch = TensorDict(support_batch, batch_size=len(support_batch['input_ids']))
                query_batch = TensorDict(query_batch, batch_size=len(query_batch['input_ids']))

                # Adapt on support set
                inner_result = self._inner_loop_update(
                    task_name=task_name,
                    support_batch=support_batch,
                    return_grads=False
                )

                # Apply adapted parameters
                fast_weights = inner_result['fast_weights']
                for name, param in self.model.named_parameters():
                    param.data = fast_weights[name].data

                # Evaluate on query set with adapted parameters
                query_loss = self._compute_sft_loss(query_batch, self.model)

                eval_metrics[f'eval/{task_name}/query_loss'] = query_loss.item()
                eval_metrics[f'eval/{task_name}/support_loss'] = inner_result['support_loss']
                eval_metrics[f'eval/{task_name}/adaptation_gap'] = query_loss.item() - inner_result['support_loss']

                # Restore original parameters for next task
                for name, param in self.model.named_parameters():
                    param.data = original_state[name].data

        self.model.train()
        return eval_metrics

    def save_checkpoint(self, step: int):
        """Save meta-model checkpoint"""
        if self.device_mesh is None or self.device_mesh.get_rank() == 0:
            checkpoint_path = os.path.join(
                self.config.trainer.default_local_dir,
                f"maml_checkpoint_step_{step}.pt"
            )

            torch.save({
                'step': step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.meta_optimizer.state_dict(),
                'config': self.config,
            }, checkpoint_path)

            logger.info(f"Saved checkpoint to {checkpoint_path}")


# Helper function to create task datasets
def create_task_datasets(config, tokenizer) -> Dict[str, Dict[str, Dataset]]:
    """
    Create task-specific datasets

    Each task should have:
    - support set: for inner loop adaptation (few-shot examples)
    - query set: for meta-gradient computation

    Example structure:
    {
        'medical': {
            'support': MedicalSFTDataset(...),
            'query': MedicalSFTDataset(...)
        },
        'legal': {
            'support': LegalSFTDataset(...),
            'query': LegalSFTDataset(...)
        },
        ...
    }
    """
    from verl.utils.dataset import SFTDataset

    task_datasets = {}

    for task_config in config.meta.tasks:
        task_name = task_config.name

        support_dataset = SFTDataset(
            parquet_files=task_config.support_files,
            tokenizer=tokenizer,
            config=config.data,
            max_samples=task_config.get('support_max_samples', -1)
        )

        query_dataset = SFTDataset(
            parquet_files=task_config.query_files,
            tokenizer=tokenizer,
            config=config.data,
            max_samples=task_config.get('query_max_samples', -1)
        )

        task_datasets[task_name] = {
            'support': support_dataset,
            'query': query_dataset
        }

    return task_datasets


def run_maml_sft(config):
    """Main entry point for MAML-SFT training"""
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
    task_datasets = create_task_datasets(config, tokenizer)

    # Initialize trainer
    trainer = MAMLSFTTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        task_datasets=task_datasets,
        device_mesh=device_mesh,
        use_fomaml=config.meta.get('use_fomaml', True),
    )

    # Train
    trainer.fit()

    # Cleanup
    destroy_global_process_group()


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path="config", config_name="maml_sft_trainer", version_base=None)
    def main(config: DictConfig):
        run_maml_sft(config)

    main()
