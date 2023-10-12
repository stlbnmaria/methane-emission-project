import pandas as pd
import xgboost as xgb


def train_model(
        train_data: any, # need to fix
        model: str = "xgb",
        model_params: dict[str, any] = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
        },
        eval_metric: str = "auc",
        num_epochs: int = 10
) -> any: # need to fix
    # if have one train df
    if isinstance(train_data: pd.DataFrame):
        X = train_data.drop(columns=["plume"])
        y = train_data["plume"]

        if model =="xgb":
            # Convert data to DMatrix format (needed for xgb)
            dtrain = xgb.DMatrix(X, label=y)
            model = model.train(model_params,
                                dtrain,
                                num_boost_round=num_epochs)
            
    # if have tulp hof train and test df        
    elif isinstance(train_data, tuple) and len(train_data) ==2:
        train_set, test_set = train_data
        X_train, y_train = train_set.drop(columns=["plume"]), train_set["plume"]
        X_test, y_test = test_set.drop(columns=["plume"]), test_set["plume"]

        if model =="xgb":
            # Convert data to DMatrix format (needed for xgb)
            dtrain = xgb.DMatrix(X_train, label=y_train)
            model = model.train(model_params,
                                dtrain,
                                num_boost_round=num_epochs)
            
    elif isinstance(train_data, list) and all(isinstance(item, dict) and "train" in item and "val" in item for item in train_data):
        # empty list so save models for each fold
        models = []
        for fold_data in train_data:
            train_set, val_set = fold_data["train"], fold_data["val"]
            X_train, y_train = train_set.drop(columns=["plume"]), train_set["plume"]
            dtrain = xgb.DMatrix(X_train, label=y_train)

            model = model = model.train(model_params,
                                        dtrain,
                                        num_boost_round=num_epochs)
    
    else:
        raise ValueError("Invalid train_data format")
    
    if isinstance(model, list):
        return models
    
    return model

def evaluate_model(model, data):
    # if multiple models were trained one per fold
    if isinstance(model, list):
        auc = []
        acc = []
        for fold_model in model:
            auc, acc = evaluate_single_model(fold_model, data)
            auc,append()

