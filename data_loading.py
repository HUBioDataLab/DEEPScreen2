"""Lightweight datasets and DataLoaders used by model training."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler


DEFAULT_TRAINING_DATASET_ROOT = (
    Path(__file__).resolve().parent / "training_files" / "target_training_datasets"
)


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
            selected_folds = self.train_val_test_folds
        else:
            selected_folds = self.train_val_test_folds[train_val_test]

        self.compid_list = [compid_label[0] for compid_label in selected_folds]
        self.label_list = [compid_label[1] for compid_label in selected_folds]

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

        return image_array.transpose((2, 0, 1)), self.label_list[index], comp_id


def get_train_test_val_data_loaders(
    target_id,
    seed,
    batch_size=32,
    num_workers=4,
):
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

    training_dataset = DEEPScreenDataset(target_id, "training")
    validation_dataset = DEEPScreenDataset(target_id, "validation")
    test_dataset = DEEPScreenDataset(target_id, "test")
    generator = make_generator(seed)

    def make_loader(dataset):
        sampler = SubsetRandomSampler(
            range(len(dataset)),
            generator=generator,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            generator=generator,
            worker_init_fn=seed_worker,
            **loader_options,
        )

    return (
        make_loader(training_dataset),
        make_loader(validation_dataset),
        make_loader(test_dataset),
    )
