import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import KFold, train_test_split
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from typing import List, Tuple


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
            batch_size=16,
            sampler=train_sampler,
        )
        # sample the validation dataset
        valid_loader = torch.utils.data.DataLoader(
            base_data,
            batch_size=16,
            sampler=valid_sampler,
        )

        dataloaders.append({"train": train_loader, "val": valid_loader})

    return dataloaders


def load_train_data(
    data_path: Path = Path("./data/train_data/images/"), folds: int = 5
) -> List[dict[str, torch.utils.data.DataLoader]]:
    """
    Load data for training either with cross validation or train-val-split.

    Args:
    :param data_path: path to training images that are in folders per category
    :param folds: number of folds to split the training data to; if set to 1, a random
                  split to create one training dataset and one validation dataset will be performed;
                  if set to 0 the whole shuffled training data will be provided

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
    elif folds == 1:
        # do a random split for train-val-data
        train_set, val_set = torch.utils.data.random_split(
            base_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(10)
        )
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=16, shuffle=True
        )
        valid_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=True)

        train_data = [{"train": train_loader, "val": valid_loader}]
    elif folds == 0:
        # load the whole train data as one for final model training
        train_loader = torch.utils.data.DataLoader(
            base_dataset, batch_size=16, shuffle=True
        )
        train_data = [{"train": train_loader}]

    return train_data


def load_inference_data(
    data_path: Path = Path("./data/test_data/"),
) -> Tuple[torch.utils.data.DataLoader, List]:
    """
    Load data for inference incl. csv with filenames.

    Args:
    :param data_path: path to validation images

    Returns:
    :returns: tuple of validation data images in torch format and list of filenames
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
    # attention: this will produce labels, just ignore them later
    base_dataset = datasets.ImageFolder(
        data_path, transform=transform, loader=lambda path: Image.open(path)
    )
    # create torch dataloader with no shuffling
    val_loader = torch.utils.data.DataLoader(base_dataset, batch_size=1, shuffle=False)

    # get filenames from image folder
    filenames = [obj[0].split("/")[-1] for obj in base_dataset.imgs]

    return val_loader, filenames


def load_tabular_train_data(
    data_path: Path = Path("./data/train_data/metadata.csv"),
    folds: int = 5,
    rands: int = 42,
    cv_shuffle: bool = True,
) -> List[dict[str, pd.DataFrame]]:
    """This function loads and transforms the tabular data
    in such a way that it can easily be used for a ML pipeline
    Pay attention depenidng on number of folds the output data type will change.

    Args:
        data_path (Path, optional): Path of the csv. Defaults to Path("./data/train_data/metadata.csv").
        folds (int, optional): Number of folds wanted. Defaults to 5.
        rands (int, optional): Number used for random state if the data
        should be shuffeled before the cv-split. Defaults to 42.
        cv_shuffle (bool, optional): Whether to use cv_shuffle. Defaults to True.

    Returns:
        List[dict[str, pd.DataFrame]]]: list of dicts of training and validation tabular data
        for cross-validation / train-val-split
    """

    # specify the columns to include
    columns_to_needed = [
        "date",
        "plume",
        "lat",
        "lon",
        "id_coord",
        "coord_x",
        "coord_y",
    ]

    # Load the data into a DataFrame
    base_data = pd.read_csv(data_path, usecols=columns_to_needed)

    # drop duplicate rows based on id_coords
    base_data.drop_duplicates(
        subset=["id_coord", "date", "coord_x", "coord_y"], inplace=True
    )
    base_data.drop(["id_coord", "coord_x", "coord_y"], axis=1, inplace=True)

    # convert date column to datetime
    base_data["date"] = pd.to_datetime(
        base_data["date"], format="%Y%m%d", errors="coerce"
    )

    # adding month as column
    base_data["month"] = base_data["date"].dt.month
    # adding weekday as column
    base_data["weekday"] = base_data["date"].dt.weekday
    # droping date
    base_data = base_data.drop(labels="date", axis=1)

    # transforming plume from yes/no to 1/0
    yes_no_mapping = {"yes": 1, "no": 0}
    base_data["plume"] = base_data["plume"].map(yes_no_mapping)

    if folds > 1:
        # use sklearn kfold to split into random training/validation indices
        cv = KFold(n_splits=folds, random_state=rands, shuffle=cv_shuffle)

        cv_splits = []

        for train_idx, val_idx in cv.split(base_data.index):
            train_data = base_data.iloc[train_idx]
            val_data = base_data.iloc[val_idx]
            cv_splits.append({"train": train_data, "val": val_data})

    elif folds == 1:
        # use a singular train_testsplit
        train_data, val_data = train_test_split(
            base_data, test_size=0.2, random_state=rands
        )
        cv_splits = [{"train": train_data, "val": val_data}]

    elif folds == 0:
        # return all the transformed data
        train_data = base_data
        cv_splits = [{"train": train_data}]

    return cv_splits


def load_tabular_inference_data(
    data_path: Path = Path("./data/test_data"),
) -> List[dict[str, pd.DataFrame]]:
    """This function loads and transforms the tabular data for inference

    Args:
        data_path (Path, optional): Path of the parent folder of test data. Defaults to Path("./data/test_data").

    Returns:
        List[dict[str, pd.DataFrame]]]: list of dicts of test tabular data
    """

    # specify the columns to include
    columns_to_needed = ["date", "lat", "lon", "id_coord"]

    # Load the data into a DataFrame
    base_data = pd.read_csv(data_path / "metadata.csv", usecols=columns_to_needed)

    # drop duplicate rows based on id_coords
    filenames = [
        file for file in os.listdir(data_path / "images") if file != ".DS_Store"
    ]
    filenames = sorted(filenames)
    base_data = base_data.sort_values(by=["date", "id_coord"])
    base_data.drop_duplicates(subset=["id_coord", "date"], inplace=True)
    base_data.drop(["id_coord"], axis=1, inplace=True)

    # convert date column to datetime
    base_data["date"] = pd.to_datetime(
        base_data["date"], format="%Y%m%d", errors="coerce"
    )

    # adding month as column
    base_data["month"] = base_data["date"].dt.month
    # adding weekday as column
    base_data["weekday"] = base_data["date"].dt.weekday
    # droping date
    base_data = base_data.drop(labels="date", axis=1)

    infer_data = [{"test": base_data, "filenames": filenames}]
    return infer_data
