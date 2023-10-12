import click
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms

from dataloader import load_inference_data
from dataloader import load_tabular_inference_data
from xgboost import XGBClassifier


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

@click.command()
@click.option("--save", "-s", default=True, type=bool)
def tabular_inference(save: bool)-> pd.DataFrame:
    """ This function predicts on the tabular meta-test-data and saves a csv
        with path and probabilities.

    Args:
        save (bool): if the predictions should be saved to csv

    Returns:
        pd.DataFrame: pandas dataframe with paths to img and
        probability predictions
    """
    test_data = load_tabular_inference_data()
    test_df = test_data[0]["test"]
    filenames = test_data[0]["filenames"]
    loaded_model = XGBClassifier()
    loaded_model.load_model('models/best_xgb.json')

    y_pred = loaded_model.predict_proba(test_df)
    y_pred = y_pred[:, 1]

    out_df_tab = pd.DataFrame({"path":filenames, "label": y_pred})

    if save:
        out_df_tab.to_csv("predictions/submission_test_tabular.csv", index=False)
    return out_df_tab


@click.command()
@click.option("--save", "-s", default=True, type=bool)
def ensemble_method(save: bool)-> pd.DataFrame:
    """using weighted average to combine image predictions
       and tabular predictions to one unified predictions csv

    Args:
        save (bool): if the predictions should be saved to csv

    Returns:
        pd.DataFrame: pandas dataframe with paths to img and
        probability predictions
    """
    out_df = torch_inference(save=False)
    out_df_tab = tabular_inference(save=False)

    merged_df = pd.merge(out_df, out_df_tab, on="path")

    merged_df["label"] = merged_df["label_x"]*0.65 + merged_df["label_y"]*0.35
    
    merged_df.drop(["label_x", "label_y"], axis=1, inplace=True)

    if save:
        merged_df.to_csv("predictions/submission_ensemble.csv", index=False)
    return merged_df

if __name__ == "__main__":
    torch_inference()
    tabular_inference()
    ensemble_method()

