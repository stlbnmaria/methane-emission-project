import click
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms

from dataloader import load_inference_data


@click.command()
@click.option("--save", "-s", default=True, type=bool)
def torch_inference(save: bool) -> pd.DataFrame:
    """
    This function predicts on the test data and saves a csv with path and probabilities.

    Args: 
    :param save: if the predictions should be saved to csv

    Returns: 
    :returns: pandas dataframe with paths to img and probability predictions
    """
    # load inference data
    val_data, filenames = load_inference_data()
    preds = []

    # set up model
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load("./models/best.pt"))
    model.eval()

    # iterate over data
    for inputs, _ in val_data:
        # perform necessary transformations for inference data
        data_aug = nn.Sequential(
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.Normalize([0.2315, 0.2315, 0.2315], [0.2268, 0.2268, 0.2268]),
        )
        inputs_aug = data_aug(inputs)

        # predict with model
        outputs = model(inputs_aug)
        # get probabilities with Softmax activation
        probs = nn.Softmax(dim=1)(outputs)[:, 1]
        preds.append(probs.item())

    # save predictions as csv for hand in
    out_df = pd.DataFrame({"path": filenames, "label": preds})
    if save:
        out_df.to_csv("predictions/submission_test_file.csv", index=False)
    
    return out_df


if __name__ == "__main__":
    torch_inference()
