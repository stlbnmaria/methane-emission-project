from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torcheval.metrics import BinaryAUROC, BinaryAccuracy, Mean
from torchvision import models, transforms
import time
import os
from tempfile import TemporaryDirectory
from dataloader import load_train_data


def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch}/{num_epochs - 1}")
            print("-" * 10)

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
                                transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.RandomRotation(degrees=30),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.Normalize(
                                    [0.2315, 0.2315, 0.2315], [0.2268, 0.2268, 0.2268]
                                ),
                            )
                            data_aug_input = nn.Sequential(
                                transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.Normalize(
                                    [0.2315, 0.2315, 0.2315], [0.2268, 0.2268, 0.2268]
                                ),
                            )
                        if phase == "val":
                            data_aug = nn.Sequential(
                                transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.Normalize(
                                    [0.2315, 0.2315, 0.2315], [0.2268, 0.2268, 0.2268]
                                ),
                            )

                        inputs_aug = data_aug(inputs) # augmented data
                        input_resize = data_aug_input(inputs) # original resized data
                        inputs_com = torch.cat((input_resize, inputs_aug), dim=0)  # Concatenate original and augmented data
                        labels = torch.cat((labels, labels), dim=0)
                        #######################
                        

                        outputs = model(inputs_com)
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

                #if phase == "train":
                #    epoch_loss = running_loss / (dataset_sizes[phase] * 2)
                #    epoch_acc = running_corrects.double() / (dataset_sizes[phase] * 2)
                
                #else:
                #    epoch_loss = running_loss / dataset_sizes[phase]
                #    epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUC: {auc.compute().item():.4f}")

                # deep copy the model
                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        print(f"Best val Acc: {best_acc:4f}")

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model


def fine_tune(device, dataloaders):
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
    model_ft = train_model(
        model_ft, dataloaders, criterion, optimizer_ft, exp_lr_scheduler, device, num_epochs=10
    )


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_path = Path("./data/train_data/images/")
    train_data = load_train_data(data_path, folds=1)
    for dataloaders in train_data:
        fine_tune(device, dataloaders)