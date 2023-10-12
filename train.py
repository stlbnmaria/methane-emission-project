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

# TODO: make typing specific in the end - when final


# create and configure logger
logging.basicConfig(filename="models/modeling.log", format="%(message)s", filemode="w")
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def train_model(
    model: models,
    dataloaders: dict[str, torch.utils.data.DataLoader],
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.SGD,
    scheduler: lr_scheduler.StepLR,
    device: torch.device,
    num_epochs: int = 10,
    save: bool = False,
) -> float:
    """
    Train torch model on fold - incl. validation depending on data.

    Args:
    :param model: torch model - either pretrained or SimpleCNN as baseline
    :param dataloaders: data as dict from dataloader.py
    :param criterion: loss criterion
    :param optimizer: optimizer for backpropagation
    :param scheduler: learning rate scheduler
    :param device: device to get GPU / CPU
    :param num_epochs: number of epochs for training of the fold
    :param save: if the model of the best epoch (according to val AUC) should be saved


    Returns:
    :returns: validation AUC of the fold
    """
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
            for phase in list(dataloaders.keys()):
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
                        # data augmentation
                        # TODO: complete and maybe carve out as function
                        if phase == "train":
                            data_aug = nn.Sequential(
                                transforms.Resize(75),
                                transforms.RandomCrop(64),
                                transforms.RandomRotation(degrees=30),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.RandomAdjustSharpness(2.0),
                                transforms.RandomAutocontrast(p=0.5),
                                transforms.Normalize(
                                    [0.2315, 0.2315, 0.2315], [0.2268, 0.2268, 0.2268]
                                ),
                            )    

                            data_aug_input = nn.Sequential(
                                # transforms.Resize(256),
                                # transforms.RandomCrop(224),
                                transforms.Normalize(
                                    [0.2315, 0.2315, 0.2315], [0.2268, 0.2268, 0.2268]
                                ),
                            )

                        if phase == "val":
                            data_aug = nn.Sequential(
                                #transforms.Resize(256),
                                #transforms.RandomCrop(224),
                                transforms.Normalize(
                                    [0.2315, 0.2315, 0.2315], [0.2268, 0.2268, 0.2268]
                                ),
                            )

                        inputs_aug = data_aug(inputs) # augmented data
                        input_resize = data_aug_input(inputs) # original resized data depending on model
                        inputs_comb = torch.cat((input_resize, inputs_aug), dim=0)  # Concatenate original and augmented data
                        labels = torch.cat((labels, labels), dim=0)
                        #######################

                        # fine tune model - forward pass
                        outputs = model(inputs_comb)
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
    """
    Sets up model for finetuning / training for the fold and by
    using the train_model function completes the job.

    Args:
    :param device: device to get GPU / CPU
    :param dataloaders: data as dict from dataloader.py
    :param how: baseline model or pretrained
    :param num_epochs: number of epochs for training of the fold
    :param learning_rate: initial learning rate
    :param save: if the model of the best epoch (according to val AUC) should be saved

    Returns:
    :returns: validation AUC of the fold
    """
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
    auc = train_model(
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
    """
    This function performs the training.

    Args:
    :param folds: number of folds for cv (2+), train-val (1) or just train (0) data
    :param how: baseline model or pretrained
    :param save: if the model of the best epoch (according to val AUC) should be saved
    :param num_epochs: number of epochs for training of the fold
    :param learning_rate: initial learning rate

    Returns:
    :returns: None
    """
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

    if folds > 0:
        logger.info("------------------")
        logger.info(f"Average val AUC: {sum(aucs)/folds:.4f}")


if __name__ == "__main__":
    main()
