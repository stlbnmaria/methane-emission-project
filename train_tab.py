from sklearn.metrics import accuracy_score, roc_auc_score
from dataloader import load_tabular_train_data
from xgboost import XGBClassifier

cv_splits = load_tabular_train_data()

auc_scores = []
acc_scores = []

for fold in cv_splits:

    train_data = fold["train"]
    val_data = fold["val"]

    X_train = train_data.drop(columns=["plume"])
    y_train = train_data["plume"]
    X_val = val_data.drop(columns=["plume"])
    y_val = val_data["plume"]

    model = XGBClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_val)
    print(y_pred)
    y_pred = y_pred[:, 1]

    auc = roc_auc_score(y_val, y_pred)
    acc = accuracy_score(y_val, y_pred>=0.5)

    auc_scores.append(auc)
    acc_scores.append(acc)

print(auc_scores)
print(acc_scores)
