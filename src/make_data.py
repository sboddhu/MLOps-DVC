import json
from pathlib import Path
import pandas as pd
import yaml
from sklearn.datasets import make_classification

def main():
    params = yaml.safe_load(open("params.yaml"))
    p = params["data"]

    X, y = make_classification(
        n_samples=p["n_samples"],
        n_features=p["n_features"],
        n_informative=p["n_informative"],
        n_redundant=max(0, p["n_features"] - p["n_informative"] - 5),
        n_clusters_per_class=2,
        class_sep=p["class_sep"],
        random_state=p["random_state"],
    )

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["label"] = y

    out = Path("data")
    out.mkdir(exist_ok=True)
    df.to_csv(out / "dataset.csv", index=False)

    meta = {
        "n_samples": int(p["n_samples"]),
        "n_features": int(p["n_features"]),
        "n_informative": int(p["n_informative"]),
        "class_sep": float(p["class_sep"]),
        "random_state": int(p["random_state"]),
    }
    Path("data/meta.json").write_text(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
