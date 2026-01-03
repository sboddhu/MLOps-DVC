import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def main():
    params = yaml.safe_load(open("params.yaml"))
    p_data = params["data"]
    p_train = params["train"]

    df = pd.read_csv("data/dataset.csv")
    X = df.drop(columns=["label"]).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=p_data["test_size"],
        random_state=p_data["random_state"],
        stratify=y
    )

    model = LogisticRegression(
        C=float(p_train["C"]),
        max_iter=int(p_train["max_iter"]),
        solver=str(p_train["solver"]),
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba))

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/model.joblib")

    Path("models/train_meta.json").write_text(json.dumps({
        "auc_sanity": auc,
        "C": float(p_train["C"]),
        "max_iter": int(p_train["max_iter"]),
        "solver": str(p_train["solver"]),
    }, indent=2))

if __name__ == "__main__":
    main()
