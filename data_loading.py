"""Lightweight datasets and DataLoaders used by model training."""

import json
import math
import numbers
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler


DEFAULT_TRAINING_DATASET_ROOT = (
    Path(__file__).resolve().parent / "training_files" / "target_training_datasets"
)


def normalize_binary_label(value, context="label"):
    """Return a canonical integer binary label or fail with a clear error."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            pass
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, numbers.Real) and math.isfinite(float(value)):
        numeric_value = float(value)
        if numeric_value in (0.0, 1.0):
            return int(numeric_value)
    raise ValueError(
        f"{context} must be a binary value (0 or 1); received {value!r}. "
        "The DEEPScreen training pipeline currently supports binary "
        "classification only."
    )


def normalize_binary_labels(values, context="labels"):
    return [
        normalize_binary_label(value, f"{context}[{index}]")
        for index, value in enumerate(values)
    ]


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


class DEEPScreenDataset(Dataset):
    def __init__(
        self,
        target_id,
        train_val_test,
        parent_path=DEFAULT_TRAINING_DATASET_ROOT,
    ):
        self.target_id = target_id
        self.train_val_test = train_val_test
        self.dataset_path = Path(parent_path) / target_id

        split_path = self.dataset_path / "train_val_test_dict.json"
        with split_path.open(encoding="utf-8") as split_file:
            self.train_val_test_folds = json.load(split_file)

        if train_val_test == "all":
            selected_folds = [
                item
                for split_name in ("training", "validation", "test")
                for item in self.train_val_test_folds.get(split_name, [])
            ]
        else:
            if train_val_test not in self.train_val_test_folds:
                raise ValueError(
                    f"Unknown split {train_val_test!r}; expected training, "
                    "validation, test, or all."
                )
            selected_folds = self.train_val_test_folds[train_val_test]

        malformed = [item for item in selected_folds if len(item) < 2]
        if malformed:
            raise ValueError(f"Malformed split entries in {split_path}: {malformed[:3]}")

        self.compid_list = [compid_label[0] for compid_label in selected_folds]
        self.label_list = normalize_binary_labels(
            [compid_label[1] for compid_label in selected_folds],
            context=f"{target_id}/{train_val_test}",
        )

    def __len__(self):
        return len(self.compid_list)

    def __getitem__(self, index):
        comp_id = self.compid_list[index]
        image_path = self.dataset_path / "imgs" / f"{comp_id}.png"

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found for compound ID: {comp_id}")

        with Image.open(image_path) as image:
            image_array = (
                np.asarray(image, dtype=np.float32) / np.float32(255.0)
            )

        if image_array.ndim != 3 or image_array.shape[2] != 3:
            raise ValueError(
                f"Expected an RGB image at {image_path}, got {image_array.shape}"
            )

        return image_array.transpose((2, 0, 1)), self.label_list[index], comp_id


def get_train_test_val_data_loaders(
    target_id,
    seed,
    batch_size=32,
    num_workers=4,
    parent_path=DEFAULT_TRAINING_DATASET_ROOT,
):
    """Build deterministic loaders while honoring a caller-provided dataset root."""
    num_workers = int(num_workers)
    if num_workers < 0:
        raise ValueError("num_workers must be greater than or equal to zero")

    loader_options = {
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        # Spawn prevents workers from inheriting the parent's CUDA handles.
        loader_options["multiprocessing_context"] = "spawn"

    datasets = (
        DEEPScreenDataset(target_id, "training", parent_path=parent_path),
        DEEPScreenDataset(target_id, "validation", parent_path=parent_path),
        DEEPScreenDataset(target_id, "test", parent_path=parent_path),
    )

    def make_loader(dataset, generator_seed):
        generator = make_generator(generator_seed)
        sampler = SubsetRandomSampler(range(len(dataset)), generator=generator)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            generator=generator,
            worker_init_fn=seed_worker,
            **loader_options,
        )

    return tuple(
        make_loader(dataset, int(seed) + offset)
        for offset, dataset in enumerate(datasets)
    )
