import click
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torcheval.metrics import BinaryAUROC, BinaryAccuracy, Mean
from torchvision import models, transforms
import time
import os
from tempfile import TemporaryDirectory

from models.baseline_cnn import SimpleCNN
from dataloader import load_train_data


# create and configure logger
logging.basicConfig(filename="models/modeling.log", format="%(message)s", filemode="w")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def train_model(
    model: models,
    dataloaders: dict[str, torch.utils.data.DataLoader],
    criterion: nn,
    optimizer: optim,
    scheduler: lr_scheduler,
    device: torch.device,
    num_epochs: int = 10,
    save: bool = False,
) -> float:
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
                # initialise metrics to track
                auc = BinaryAUROC()
                acc = BinaryAccuracy(threshold=0.5)
                running_loss = Mean()

                # set model to correct mode for the phase
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                # iterate over batches
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        # data augmentation HERE
                        # TODO: complete and maybe carve out as function
                        if phase == "train":
                            data_aug = nn.Sequential(
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.Normalize(
                                    [0.2315, 0.2315, 0.2315], [0.2268, 0.2268, 0.2268]
                                ),
                            )
                        if phase == "val":
                            data_aug = nn.Sequential(
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.Normalize(
                                    [0.2315, 0.2315, 0.2315], [0.2268, 0.2268, 0.2268]
                                ),
                            )

                        inputs_aug = data_aug(inputs)
                        #######################

                        # fine tune model - forward pass
                        outputs = model(inputs_aug)
                        # get probabilities with Softmax activation
                        probs = nn.Softmax(dim=1)(outputs)[:, 1]
                        # calculate loss
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # update epoch statistics - meaning add the preds / labels / loss
                    running_loss.update(loss.detach(), weight=len(inputs))
                    auc.update(probs, labels)
                    acc.update(probs, labels)

                # for train set learning rate scheduler
                if phase == "train":
                    scheduler.step()

                # compute epoch statistics
                epoch_loss = running_loss.compute().item()
                epoch_acc = acc.compute().item()
                epoch_auc = auc.compute().item()

                # log statistics for this epoch
                logger.info(
                    f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUC: {epoch_auc:.4f}"
                )

                # deep copy the model
                if phase == "val" and epoch_auc > best_auc:
                    best_auc = epoch_auc
                    torch.save(model.state_dict(), best_model_params_path)

            logger.info("")

        # log time for the fold and best validation AUC
        time_elapsed = time.time() - since
        logger.info(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        logger.info(f"Best val AUC: {best_auc:4f}")

        # save model to best.pt in models to be used for inference
        if save:
            # load best model weights
            model.load_state_dict(torch.load(best_model_params_path))
            torch.save(model.state_dict(), "./models/best.pt")

    return best_auc


def fine_tune(
    device: torch.device,
    dataloaders: dict[str, torch.utils.data.DataLoader],
    how: str,
    num_epochs: int,
    learning_rate: float,
    save: bool,
) -> float:
    """ """
    # if baseline is demanded load simple CNN from models / baseline_cnn.py
    if how == "baseline":
        model_ft = SimpleCNN().to(device)
        logger.info("Initialize baseline model")
    # if pre-trained load torch pretrained model and replace last layer
    else:
        model_ft = models.resnet18(weights="IMAGENET1K_V1")
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)
        model_ft = model_ft.to(device)

    # define loss criterion to be cross entropy
    criterion = nn.CrossEntropyLoss()

    # define optimizer with initial learning rate and momentum
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=0.9)

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
        num_epochs,
        save,
    )
    return auc


@click.command()
@click.option("--folds", "-f", default=5, type=int)
@click.option("--how", "-h", default="pretrained", type=str)
@click.option("--save", "-s", default=False, type=bool)
@click.option("--num_epochs", "-e", default=5, type=int)
@click.option("--learning_rate", "-l", default=0.001, type=float)
def main(folds, save, how, num_epochs, learning_rate):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_data = load_train_data(folds=folds)

    # count folds and track AUC per fold
    i = 0
    aucs = []
    # iterate over folds in train_data
    for dataloaders in train_data:
        i += 1
        logger.info("------------------")
        logger.info(f"Starting fold {i}")
        fold_auc = fine_tune(device, dataloaders, how, num_epochs, learning_rate, save)
        aucs.append(fold_auc)

    logger.info("------------------")
    logger.info(f"Average val AUC: {sum(aucs)/folds:.4f}")


if __name__ == "__main__":
    main()
