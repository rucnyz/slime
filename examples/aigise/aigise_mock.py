"""
AIgiSE Mock Data Generator for slime Training

Generates a JSONL file from an AIgiSE dataset that slime can use as
``--prompt-data``.  Each line contains:
    {"index": <int>, "metadata": <dataset_row_dict>}

Supports local JSON array files, local HF disk datasets, and remote
HuggingFace Hub datasets.

Usage:
    python aigise_mock.py --local_dir /root/aigise_data/ \\
                          --dataset_path /path/to/mock_test_dataset.json \\
                          --output_filename mock_tasks.jsonl
"""

import argparse
import json
import os
from typing import Any

import datasets


def _load_dataset_rows(dataset_path: str, dataset_split: str) -> list:
    if os.path.exists(dataset_path):
        if os.path.isdir(dataset_path):
            return list(datasets.load_from_disk(dataset_path))

        if dataset_path.endswith(".json"):
            with open(dataset_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data

        return list(datasets.load_dataset("json", data_files=dataset_path, split="train"))

    return list(datasets.load_dataset(dataset_path, split=dataset_split))


def main():
    parser = argparse.ArgumentParser(description="AIgiSE Mock Data Generator")
    parser.add_argument(
        "--local_dir",
        required=True,
        help="Directory to write the output JSONL file",
    )
    parser.add_argument(
        "--dataset_path",
        default="sunblaze-ucb/cybergym",
        help="HuggingFace dataset path or local path",
    )
    parser.add_argument(
        "--dataset_split",
        default="tasks",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--task_subset_file",
        default=None,
        help="Optional file with task_id list (one per line) to filter tasks",
    )
    parser.add_argument(
        "--output_filename",
        default="aigise_tasks.jsonl",
        help="Name of the output JSONL file (default: aigise_tasks.jsonl)",
    )
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)

    rows = _load_dataset_rows(args.dataset_path, args.dataset_split)

    if args.task_subset_file and os.path.exists(args.task_subset_file):
        with open(args.task_subset_file, "r") as f:
            task_list = {line.strip() for line in f if line.strip()}
        rows = [r for r in rows if r.get("task_id") in task_list]
        print(f"Filtered to {len(rows)} tasks using {args.task_subset_file}")

    output_path = os.path.join(args.local_dir, args.output_filename)
    with open(output_path, "w") as f:
        for i, row in enumerate(rows):
            row_dict = dict(row) if not isinstance(row, dict) else row
            record = {"index": i, "metadata": row_dict}
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(rows)} tasks to {output_path}")


if __name__ == "__main__":
    main()
