import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
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
            train_set, batch_size=32, shuffle=True
        )
        valid_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=True)

        train_data = [{"train": train_loader, "val": valid_loader}]
    elif folds == 0:
        # load the whole train data as one for final model training
        train_loader = torch.utils.data.DataLoader(
            base_dataset, batch_size=32, shuffle=True
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
        folds: int=5,
        rands: int=42,
        cv_shuffle: bool = True):
    """_summary_

    Args:
        data_path (Path, optional): _description_. Defaults to Path("./data/train_data/metadata.csv").
        folds (int, optional): _description_. Defaults to 5.
    """

    # specify the columns to include
    columns_to_needed = ["date", "plume", "lat", "lon"]

    # Load the data into a DataFrame
    base_data = pd.read_csv(data_path, usecols=columns_to_needed)

    # convert date column to datetime
    base_data["date"] = pd.to_datetime(base_data['date'],
                                       format="%Y%m%d",
                                       errors='coerce')
    
    #data = base_data[["lat", "lon", "plume"]]


    base_data["month"] = base_data["date"].dt.month
    base_data["weekday"] = base_data["date"].dt.weekday

    base_data = base_data.drop(labels= "date", axis=1)

    X = base_data.drop(columns=["plume"])
    y = base_data["plume"]

    if folds > 1:
        cv = KFold(n_splits=folds, random_state=rands, shuffle=cv_shuffle)

        cv_splits = []

        for train_idx, val_idx in cv.split(X,y):
            train_data = base_data.iloc[train_idx]
            val_data = base_data.iloc[val_idx]
            cv_splits.append({"train": train_data, "val": val_data})
            
        return cv_splits

    elif folds == 1:
        train_data, val_data = train_test_split(base_data, test_size=0.2, random_state=rands)
        
        return train_data, val_data

    elif folds == 0:
        train_data = base_data
        
        return train_data
