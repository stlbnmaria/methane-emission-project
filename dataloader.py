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


def load_train_data(data_path: Path) -> datasets.ImageFolder:
    """
    Load data for training either with cross validation or train-val-split.

    Args:
    :param data_path: path to training images that are in folders per category

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

    # create dataset from folder using torchvision
    train_dataset = datasets.ImageFolder(
        data_path, transform=transform, loader=lambda path: Image.open(path)
    )

    # get cv splits to perform cross validation
    train_data = get_cv_dataloaders(train_dataset, train_dataset)

    return train_data
