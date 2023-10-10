import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import KFold
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from typing import List


def get_cv_dataloaders(
    base_data: datasets.ImageFolder,
    folds: int = 5,
    cv_shuffle: bool = True,
    rands: int = 10,
) -> List[dict[str, torch.utils.data.DataLoader]]:
    """
    Divide torch dataset from folder into cv splits for training and validation.

    Args:
    :param base_data: data for training
    :param folds: number of folds for cross validation
    :param cv_shuffle: if the data should be shuffeled before the cv-split
    :param rands: random state for the cv-split

    Returns:
    :returns: list of dicts of training and validation torch data for cross-validation
    """

    num_images = len(base_data)
    indices = list(range(num_images))

    dataloaders = []

    # use sklearn kfold to split into random training/validation indices
    cv = KFold(n_splits=folds, random_state=rands, shuffle=cv_shuffle)
    for train_idx, valid_idx in cv.split(indices):
        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # create dataloaders using the cv indexes
        # sample the training dataset
        train_loader = torch.utils.data.DataLoader(
            base_data,
            batch_size=32,
            sampler=train_sampler,
        )
        # sample the validation dataset
        valid_loader = torch.utils.data.DataLoader(
            base_data,
            batch_size=32,
            sampler=valid_sampler,
        )

        dataloaders.append({"train": train_loader, "val": valid_loader})

    return dataloaders


def load_train_data(
    data_path: Path, folds: int = 5
) -> List[dict[str, torch.utils.data.DataLoader]]:
    """
    Load data for training either with cross validation or train-val-split.

    Args:
    :param data_path: path to training images that are in folders per category
    :param folds: number of folds to split the training data to; if set to 1, a random
                  split to create one training dataset and one validation dataset will be performed

    Returns:
    :returns: list of dicts of training and validation torch data for cross-validation / train-val-split
    """

    # define transforms
    # in order to load uint16 some transformation with PIL is necessary
    transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda image: torch.from_numpy(
                    np.array(image).astype(np.float32) / 65535
                ).repeat(3, 1, 1)
            ),
        ]
    )
    # create dataset from folder using torchvision
    base_dataset = datasets.ImageFolder(
        data_path, transform=transform, loader=lambda path: Image.open(path)
    )

    if folds > 1:
        # get cv splits to perform cross validation
        train_data = get_cv_dataloaders(base_dataset, folds=folds)
    else:
        # do a random split for train-val-data
        train_set, val_set = torch.utils.data.random_split(
            base_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(10)
        )
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=32)
        valid_loader = torch.utils.data.DataLoader(val_set, batch_size=32)

        train_data = [{"train": train_loader, "val": valid_loader}]

    return train_data