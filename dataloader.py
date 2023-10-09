import numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms

# Define default PATH for data
PATH = "./data/train_data/images/"

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
    PATH, transform=transform, loader=lambda path: Image.open(path)
)

# define data loader from torch
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
