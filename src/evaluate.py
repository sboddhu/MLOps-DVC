import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

def main():
    params = yaml.safe_load(open("params.yaml"))
    p_data = params["data"]
    p_eval = params["eval"]

    df = pd.read_csv("data/dataset.csv")
    X = df.drop(columns=["label"]).values
    y = df["label"].values

    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=p_data["test_size"],
        random_state=p_data["random_state"],
        stratify=y
    )

    model = joblib.load("models/model.joblib")
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    auc = float(roc_auc_score(y_test, proba))
    acc = float(accuracy_score(y_test, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)

    metrics = {
        "roc_auc": auc,
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "n_test": int(len(y_test))
    }
    Path("metrics.json").write_text(json.dumps(metrics, indent=2))

    thresholds = int(p_eval.get("thresholds", 50))
    thr = np.linspace(0.0, 1.0, thresholds)
    tpr = []
    fpr = []

    y_pos = (y_test == 1)
    n_pos = max(1, int(y_pos.sum()))
    n_neg = max(1, int((~y_pos).sum()))

    for t in thr:
        pred = proba >= t
        tp = int((pred & y_pos).sum())
        fp = int((pred & ~y_pos).sum())
        tpr.append(tp / n_pos)
        fpr.append(fp / n_neg)

    out = Path("reports")
    out.mkdir(exist_ok=True)
    pd.DataFrame({"threshold": thr, "tpr": tpr, "fpr": fpr}).to_csv(out / "roc.csv", index=False)

if __name__ == "__main__":
    main()
