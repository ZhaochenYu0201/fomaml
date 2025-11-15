"""
Data Preparation Script for MAML-SFT

This script helps prepare task-specific datasets for meta-learning.
Each task needs support (few-shot examples) and query (evaluation) sets.

Example usage:
    python prepare_maml_data.py --task medical --support-ratio 0.2
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json


def split_dataset_for_maml(
    input_file: str,
    output_dir: str,
    task_name: str,
    support_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[str, str]:
    """
    Split a dataset into support and query sets for MAML

    Args:
        input_file: Path to input parquet/jsonl file
        output_dir: Directory to save split datasets
        task_name: Name of the task (e.g., 'medical', 'legal')
        support_ratio: Ratio of data to use as support set (rest is query)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (support_file_path, query_file_path)
    """
    # Read input data
    if input_file.endswith('.parquet'):
        df = pd.read_parquet(input_file)
    elif input_file.endswith('.jsonl'):
        df = pd.read_json(input_file, lines=True)
    else:
        raise ValueError("Input file must be .parquet or .jsonl")

    print(f"Loaded {len(df)} samples from {input_file}")

    # Shuffle and split
    np.random.seed(seed)
    indices = np.random.permutation(len(df))
    split_idx = int(len(df) * support_ratio)

    support_indices = indices[:split_idx]
    query_indices = indices[split_idx:]

    support_df = df.iloc[support_indices].reset_index(drop=True)
    query_df = df.iloc[query_indices].reset_index(drop=True)

    # Create output directory
    output_path = Path(output_dir) / task_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Save split datasets
    support_file = output_path / "support.parquet"
    query_file = output_path / "query.parquet"

    support_df.to_parquet(support_file, index=False)
    query_df.to_parquet(query_file, index=False)

    print(f"Saved {len(support_df)} support samples to {support_file}")
    print(f"Saved {len(query_df)} query samples to {query_file}")

    return str(support_file), str(query_file)


def create_multi_domain_dataset(
    task_configs: Dict[str, Dict],
    output_dir: str,
    seed: int = 42
) -> Dict[str, Dict[str, str]]:
    """
    Create multi-domain dataset for MAML from multiple sources

    Args:
        task_configs: Dictionary mapping task names to config dicts
            Example:
            {
                'medical': {
                    'input_file': 'path/to/medical.parquet',
                    'support_ratio': 0.2,
                    'max_samples': 5000
                },
                ...
            }
        output_dir: Root directory for output
        seed: Random seed

    Returns:
        Dictionary mapping task names to support/query file paths
    """
    dataset_info = {}

    for task_name, config in task_configs.items():
        print(f"\n=== Processing task: {task_name} ===")

        input_file = config['input_file']
        support_ratio = config.get('support_ratio', 0.2)
        max_samples = config.get('max_samples', -1)

        # Load and optionally subsample
        if input_file.endswith('.parquet'):
            df = pd.read_parquet(input_file)
        elif input_file.endswith('.jsonl'):
            df = pd.read_json(input_file, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {input_file}")

        if max_samples > 0 and len(df) > max_samples:
            np.random.seed(seed)
            df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
            print(f"Subsampled to {max_samples} samples")

        # Save to temp file
        temp_file = Path(output_dir) / "temp" / f"{task_name}.parquet"
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(temp_file, index=False)

        # Split into support and query
        support_file, query_file = split_dataset_for_maml(
            input_file=str(temp_file),
            output_dir=output_dir,
            task_name=task_name,
            support_ratio=support_ratio,
            seed=seed
        )

        dataset_info[task_name] = {
            'support': support_file,
            'query': query_file
        }

        # Clean up temp file
        temp_file.unlink()

    # Save dataset info
    info_file = Path(output_dir) / "dataset_info.json"
    with open(info_file, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    print(f"\n=== Dataset info saved to {info_file} ===")

    return dataset_info


def balance_task_sizes(
    dataset_info: Dict[str, Dict[str, str]],
    target_support_size: int = 500,
    target_query_size: int = 1000,
    seed: int = 42
):
    """
    Balance the number of samples across tasks by subsampling

    This ensures each task has similar number of samples for fair meta-learning

    Args:
        dataset_info: Output from create_multi_domain_dataset
        target_support_size: Target number of support samples per task
        target_query_size: Target number of query samples per task
        seed: Random seed
    """
    np.random.seed(seed)

    for task_name, files in dataset_info.items():
        print(f"\n=== Balancing task: {task_name} ===")

        # Process support set
        support_df = pd.read_parquet(files['support'])
        if len(support_df) > target_support_size:
            support_df = support_df.sample(n=target_support_size, random_state=seed)
            support_df.to_parquet(files['support'], index=False)
            print(f"Reduced support set from {len(support_df)} to {target_support_size}")
        else:
            print(f"Support set size: {len(support_df)} (target: {target_support_size})")

        # Process query set
        query_df = pd.read_parquet(files['query'])
        if len(query_df) > target_query_size:
            query_df = query_df.sample(n=target_query_size, random_state=seed)
            query_df.to_parquet(files['query'], index=False)
            print(f"Reduced query set from {len(query_df)} to {target_query_size}")
        else:
            print(f"Query set size: {len(query_df)} (target: {target_query_size})")


def verify_dataset_format(file_path: str, required_keys: List[str] = ['prompt', 'response']):
    """
    Verify that a dataset has the required format for SFT

    Args:
        file_path: Path to parquet file
        required_keys: Required column names
    """
    df = pd.read_parquet(file_path)

    print(f"Dataset: {file_path}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {df.columns.tolist()}")

    missing_keys = [key for key in required_keys if key not in df.columns]
    if missing_keys:
        raise ValueError(f"Missing required columns: {missing_keys}")

    # Show sample
    print("\nSample data:")
    print(df.head(2).to_dict('records'))
    print()


def main():
    parser = argparse.ArgumentParser(description="Prepare data for MAML-SFT")
    parser.add_argument("--config", type=str, help="Path to task config JSON file")
    parser.add_argument("--output-dir", type=str, default="./data/maml", help="Output directory")
    parser.add_argument("--balance", action="store_true", help="Balance task sizes")
    parser.add_argument("--support-size", type=int, default=500, help="Target support set size")
    parser.add_argument("--query-size", type=int, default=1000, help="Target query set size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verify", action="store_true", help="Verify dataset formats")

    args = parser.parse_args()

    # Load task configs
    if args.config:
        with open(args.config, 'r') as f:
            task_configs = json.load(f)
    else:
        # Example configuration
        task_configs = {
            'medical': {
                'input_file': 'path/to/medical_data.parquet',
                'support_ratio': 0.2,
                'max_samples': 5000
            },
            'legal': {
                'input_file': 'path/to/legal_data.parquet',
                'support_ratio': 0.2,
                'max_samples': 5000
            },
            # Add more tasks...
        }
        print("No config provided, using example configuration")
        print("Please create a config JSON file with your task specifications")
        return

    # Create datasets
    dataset_info = create_multi_domain_dataset(
        task_configs=task_configs,
        output_dir=args.output_dir,
        seed=args.seed
    )

    # Balance if requested
    if args.balance:
        balance_task_sizes(
            dataset_info=dataset_info,
            target_support_size=args.support_size,
            target_query_size=args.query_size,
            seed=args.seed
        )

    # Verify if requested
    if args.verify:
        print("\n=== Verifying datasets ===")
        for task_name, files in dataset_info.items():
            print(f"\nTask: {task_name}")
            verify_dataset_format(files['support'])
            verify_dataset_format(files['query'])

    print("\n=== Data preparation complete! ===")
    print(f"Datasets saved to: {args.output_dir}")
    print(f"Dataset info: {Path(args.output_dir) / 'dataset_info.json'}")


if __name__ == "__main__":
    main()
