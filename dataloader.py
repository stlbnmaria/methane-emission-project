import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import KFold
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from typing import List, Tuple


def get_cv_dataloaders(
    augmented_data: datasets.ImageFolder,
    base_data: datasets.ImageFolder,
    folds: int = 5,
    cv_shuffle: bool = True,
    rands: int = 10,
) -> List[Tuple[torch.utils.data.DataLoader]]:
    """
    Divide torch dataset from folder into cv splits for training and validation.

    Args:
    :param augmented_data: augmented data for training
    :param base_data: data for validation with no augmentation - as for testing
    :param folds: number of folds for cross validation
    :param cv_shuffle: if the data should be shuffeled before the cv-split
    :param rands: random state for the cv-split

    Returns:
    :returns: list of tuples of training and validation torch data for cross-validation
    """

    num_images = len(augmented_data)
    indices = list(range(num_images))

    dataloaders = []

    # use sklearn kfold to split into random training/validation indices
    cv = KFold(n_splits=folds, random_state=rands, shuffle=cv_shuffle)
    for train_idx, valid_idx in cv.split(indices):
        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # create dataloaders using the cv indexes
        train_loader = torch.utils.data.DataLoader(
            augmented_data,
            batch_size=32,
            sampler=train_sampler,
        )
        # sample the validation dataset from a separate dataset
        # that doesn't include the image aug transformations.
        valid_loader = torch.utils.data.DataLoader(
            base_data,
            batch_size=32,
            sampler=valid_sampler,
        )

        dataloaders.append((train_loader, valid_loader))

    return dataloaders


def load_train_data(data_path: Path, folds: int = 5) -> datasets.ImageFolder:
    """
    Load data for training either with cross validation or train-val-split.

    Args:
    :param data_path: path to training images that are in folders per category
    :param folds: number of folds to split the training data to; if set to 1, a random
                  split to create one training dataset and one validation dataset will be performed

    Returns:
    :returns: list of tuples of training and validation torch data for cross-validation
    """

    # define transforms
    # in order to load uint16 some transformation with PIL is necessary
    transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda image: torch.from_numpy(
                    np.array(image).astype(np.float32) / 65535
                ).unsqueeze(0)
            )
        ]
    )
    # reduced transformations for validation dataset to not contain augmentation
    transform_infer = transforms.Compose(
        [
            transforms.Lambda(
                lambda image: torch.from_numpy(
                    np.array(image).astype(np.float32) / 65535
                ).unsqueeze(0)
            )
        ]
    )

    # create dataset from folder using torchvision
    augmented_dataset = datasets.ImageFolder(
        data_path, transform=transform, loader=lambda path: Image.open(path)
    )
    base_dataset = datasets.ImageFolder(
        data_path, transform=transform_infer, loader=lambda path: Image.open(path)
    )

    if folds > 1:
        # get cv splits to perform cross validation
        train_data = get_cv_dataloaders(augmented_dataset, base_dataset, folds=folds)
    else:
        # do a random split for train-val-data
        train_set, _ = torch.utils.data.random_split(
            augmented_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(10)
        )
        _, val_set = torch.utils.data.random_split(
            base_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(10)
        )
        train_data = [(train_set, val_set)]

    return train_data
