from pathlib import Path
from torchvision import transforms, datasets
import numpy as np
import torch
from PIL import Image

data_path = Path("./data/train_data/images/")
transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda image: torch.from_numpy(
                    # transforms tif to np.float32 
                    # necessary for pytorch
                    np.array(image).astype(np.float32) / 65535
                ).repeat(3, 1, 1)
            ),
        ]
    )
# create dataset from folder using torchvision
base_dataset = datasets.ImageFolder(
    data_path, transform=transform, loader=lambda path: Image.open(path)
)

dataloader = torch.utils.data.DataLoader(base_dataset, batch_size=428)
images, labels = next(iter(dataloader))

print("The mean of Sample data is:", torch.mean(images))
print("The std of Sample data is:", torch.std(images))