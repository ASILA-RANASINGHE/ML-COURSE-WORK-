"""
data.py - CIFAR-10 data pipeline for Exercise 1 (owner: Member A)

What this file gives the team:
    CLASSES               names of the 10 classes, in label order 0-9
    get_dataloaders()     train / validation / test DataLoaders (the main function)
    get_datasets()        the three Dataset objects (used for checks)
    split_indices()       the fixed 45,000 / 5,000 split of the training partition
    compute_mean_std()    per-channel mean and std of the 45k training images

Rules this file enforces:
    * The official test partition (10,000 images) is never mixed into training.
    * The validation set (5,000 images) is carved out of the 50,000 training images
      with a fixed seed, so every member gets exactly the same split.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Class names in the order of CIFAR-10's integer labels (0 = airplane ... 9 = truck).
CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")

# Standard per-channel (R, G, B) statistics of the CIFAR-10 training images.
# Step 7 checks that our own 45k training subset gives almost the same values.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_transform(mean=CIFAR10_MEAN, std=CIFAR10_STD):
    """Preprocessing applied to every image.

    ToTensor():  PIL image (32x32x3, values 0-255) -> float tensor [3, 32, 32], values 0-1
    Normalize(): (x - mean) / std per channel -> values roughly centred on 0 with std 1,
                 which helps gradient-based training converge faster.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def split_indices(n_total=50000, val_size=5000, seed=42):
    """Shuffle the indices 0..n_total-1 with a fixed seed and cut off a validation part.

    Returns (train_idx, val_idx). Same seed -> same split on every computer.
    """
    rng = np.random.RandomState(seed)          # private random generator, seeded
    shuffled = rng.permutation(n_total)        # e.g. [31245, 7, 40211, ...]
    val_idx = shuffled[:val_size]              # first 5,000 -> validation
    train_idx = shuffled[val_size:]            # remaining 45,000 -> training
    return train_idx, val_idx


def get_datasets(val_size=5000, seed=42, data_dir="./data", train_transform=None):
    """Build the train / validation / test Dataset objects.

    train_transform: optional extra transform for TRAINING images only
    (not used in Exercise 1; Exercise 3 passes data augmentation here).
    Validation and test images always get the plain transform.
    """
    eval_tf = get_transform()
    train_tf = train_transform if train_transform is not None else eval_tf

    # Two views of the same 50k training files: one with the training transform,
    # one with the plain transform. This keeps augmentation off the validation images.
    full_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    full_val = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=eval_tf)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=eval_tf)

    train_idx, val_idx = split_indices(len(full_train), val_size, seed)
    train_set = Subset(full_train, train_idx)
    val_set = Subset(full_val, val_idx)
    return train_set, val_set, test_set


def get_dataloaders(batch_size=128, val_size=5000, seed=42, data_dir="./data",
                    num_workers=0, train_transform=None):
    """Main function used by the whole team.

    Returns (train_loader, val_loader, test_loader).
        images: float32 tensors [B, 3, 32, 32], normalised
        labels: int64 tensors [B]
    Only the training loader shuffles. Its shuffle order is seeded, so runs repeat.
    num_workers=0 is the safe choice on Windows + Jupyter.
    """
    train_set, val_set, test_set = get_datasets(val_size, seed, data_dir, train_transform)

    g = torch.Generator()
    g.manual_seed(seed)                        # makes the batch order repeatable

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, generator=g)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, val_loader, test_loader


def compute_mean_std(data_dir="./data", val_size=5000, seed=42):
    """Per-channel mean and std of the 45k TRAINING images only (never val or test).

    Returns (mean, std), each a tuple of 3 floats (R, G, B).
    """
    raw = datasets.CIFAR10(root=data_dir, train=True, download=True)   # no transform
    train_idx, _ = split_indices(len(raw), val_size, seed)
    pixels = raw.data[train_idx].astype(np.float32) / 255.0             # shape (45000, 32, 32, 3)
    mean = pixels.mean(axis=(0, 1, 2))
    std = pixels.std(axis=(0, 1, 2))
    return tuple(float(m) for m in mean), tuple(float(s) for s in std)
