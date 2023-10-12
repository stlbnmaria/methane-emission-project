import torch
import torch.nn as nn
from torchvision import transforms
from typing import Tuple


def get_augmented_data(
    phase: str, inputs: torch.tensor, labels: torch.tensor, how: str
) -> Tuple[torch.tensor, torch.tensor]:
    """
    Get the right data augmentation depending on 

    Args:
    :param phase: phase in the training - either train or val
    :param inputs: batched images as tensors
    :param labels: batched labels for images as tensors
    :param how: baseline model or pretrained


    Returns:
    :returns: tuple of augmented / resized & normalized inputs + labels
    """
    # first define the transforms for the SimpleCNN model
    if how == "baseline":
        if phase == "train":
            data_aug = nn.Sequential(
                transforms.Resize(75),
                transforms.RandomCrop(64),
                transforms.RandomRotation(degrees=30),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomAdjustSharpness(2.0),
                transforms.RandomAutocontrast(p=0.5),
                transforms.Normalize([0.2315, 0.2315, 0.2315], [0.2268, 0.2268, 0.2268]),
            )

            data_aug_input = nn.Sequential(
                transforms.Normalize([0.2315, 0.2315, 0.2315], [0.2268, 0.2268, 0.2268]),
            )

        if phase == "val":
            data_aug = nn.Sequential(
                transforms.Normalize([0.2315, 0.2315, 0.2315], [0.2268, 0.2268, 0.2268]),
            )

    # define transforms for torch pretrained model
    elif how == "pretrained":
        resize = 232
        crop = 224
        if phase == "train":
            data_aug = nn.Sequential(
                transforms.Resize(resize),
                transforms.RandomCrop(crop),
                transforms.RandomRotation(degrees=30),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomAdjustSharpness(2.0),
                transforms.RandomAutocontrast(p=0.5),
                transforms.Normalize([0.2315, 0.2315, 0.2315], [0.2268, 0.2268, 0.2268]),
            )

            data_aug_input = nn.Sequential(
                transforms.Resize(resize),
                transforms.RandomCrop(crop),
                transforms.Normalize([0.2315, 0.2315, 0.2315], [0.2268, 0.2268, 0.2268]),
            )

        if phase == "val":
            data_aug = nn.Sequential(
                transforms.Resize(resize),
                transforms.RandomCrop(crop),
                transforms.Normalize([0.2315, 0.2315, 0.2315], [0.2268, 0.2268, 0.2268]),
            )

    inputs_aug = data_aug(inputs)  # augmented data
    if phase == "train":
        input_resize = data_aug_input(inputs)  # original resized data depending on model
        inputs_comb = torch.cat(
            (input_resize, inputs_aug), dim=0
        )  # Concatenate original and augmented data
        labels = torch.cat((labels, labels), dim=0)

        return (inputs_comb, labels)
    else: 
        return (inputs_aug, labels)
