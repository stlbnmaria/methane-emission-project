import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torcheval.metrics import BinaryAUROC, BinaryAccuracy, Mean
import torch.nn.functional as F
from torchvision import models, transforms
from typing import Tuple
import time
import os
from tempfile import TemporaryDirectory
from dataloader import load_train_data


# create and configure logger
logging.basicConfig(filename="std.log", format="%(message)s", filemode="w")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class SimpleCNN(nn.Module):
    def __init__(self):
        # ancestor constructor call
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=2)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=2
        )
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=2
        )

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x))))
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x))))
        x = self.avg(x)
        x = x.view(-1, 64)
        x = self.fc(x)
        return x


def train_model(
    model: models,
    dataloaders: dict[str, torch.utils.data.DataLoader],
    criterion: nn,
    optimizer: optim,
    scheduler: lr_scheduler,
    device: torch.device,
    num_epochs: int = 10,
):
    """ """
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

        torch.save(model.state_dict(), best_model_params_path)
        best_auc = 0.0

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch}/{num_epochs - 1}")
            logger.info("-" * 10)

            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                auc = BinaryAUROC()
                acc = BinaryAccuracy(threshold=0.5)
                running_loss = Mean()

                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        # data augmentation HERE
                        if phase == "train":
                            data_aug = nn.Sequential(
                                # transforms.Resize(256),
                                # transforms.CenterCrop(224),
                                transforms.Normalize(
                                    [0.2315, 0.2315, 0.2315], [0.2268, 0.2268, 0.2268]
                                ),
                            )
                        if phase == "val":
                            data_aug = nn.Sequential(
                                # transforms.Resize(256),
                                # transforms.CenterCrop(224),
                                transforms.Normalize(
                                    [0.2315, 0.2315, 0.2315], [0.2268, 0.2268, 0.2268]
                                ),
                            )

                        inputs_aug = data_aug(inputs)
                        #######################

                        outputs = model(inputs_aug)
                        probs = nn.Softmax(dim=1)(outputs)[:, 1]
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss.update(loss.detach(), weight=len(inputs))
                    auc.update(probs, labels)
                    acc.update(probs, labels)
                if phase == "train":
                    scheduler.step()

                epoch_loss = running_loss.compute().item()
                epoch_acc = acc.compute().item()
                epoch_auc = auc.compute().item()

                logger.info(
                    f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUC: {epoch_auc:.4f}"
                )

                # deep copy the model
                if phase == "val" and epoch_auc > best_auc:
                    best_auc = epoch_auc
                    torch.save(model.state_dict(), best_model_params_path)

            logger.info("")

        time_elapsed = time.time() - since
        logger.info(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        logger.info(f"Best val AUC: {best_auc:4f}")

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model, best_auc


def fine_tune(
    device: torch.device, dataloaders: dict[str, torch.utils.data.DataLoader], how: str
):
    """ """
    if how == "baseline":
        model_ft = SimpleCNN().to(device)
        logger.info("Initialize baseline model")
    else:
        model_ft = models.resnet18(weights="IMAGENET1K_V1")
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)

        model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # run fine tuning on pretrained model
    model_ft, auc = train_model(
        model_ft,
        dataloaders,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        device,
        num_epochs=2,
    )
    return model_ft, auc


def main():
    folds = 3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_path = Path("./data/train_data/images/")
    train_data = load_train_data(data_path, folds=folds)
    i = 0
    aucs = []
    for dataloaders in train_data:
        i += 1
        logger.info("------------------")
        logger.info(f"Starting fold {i}")
        model_ft, fold_auc = fine_tune(device, dataloaders, how="baseline")
        aucs.append(fold_auc)

    logger.info(f"Average val AUC: {sum(aucs)/folds:.4f}")


if __name__ == "__main__":
    main()
