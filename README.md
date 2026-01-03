# DVC Collaboration Mini-Tutorial (In‑Class Activity)

This repo is a **minimal, classroom-friendly** DVC example you can run in ~20–30 minutes.
It trains a tiny model on a synthetic dataset, tracks artifacts + metrics with DVC, and
shows a simple collaboration workflow (two “students” making different parameter changes).

## What you’ll practice
- `dvc init`, `dvc repro` (pipeline execution)
- `dvc metrics show/diff`, `dvc plots show`
- `dvc exp run` and `dvc exp show` (quick experiments)
- A collaboration flow via **branches + PR** (or pairs on one repo)

---

## 0) Prereqs
- Python 3.10+
- `git`
- `dvc` (pip)
- Recommended: a GitHub repo created from this folder

Install:
```bash
pip install -r requirements.txt
```

---

## 1) Initialize Git + DVC (first time only)
```bash
git init
dvc init
git add . 
git commit -m "init: dvc pipeline tutorial"
```

> If you already pushed to GitHub, add remote + push:
```bash
git remote add origin <YOUR_REPO_URL>
git push -u origin main
```

---

## 2) Run the pipeline
This will:
1) generate data
2) train a model
3) evaluate and write metrics

```bash
dvc repro
```

See metrics:
```bash
dvc metrics show
```

See a quick plot (ROC curve data):
```bash
dvc plots show reports/roc.csv --x fpr --y tpr
```

---

## 3) In‑class collaboration activity (pairs)
### Option A: Branch + PR (recommended)
**Student A**:
```bash
git checkout -b exp-higher-C
python -c "import yaml; p=yaml.safe_load(open('params.yaml')); p['train']['C']=2.0; open('params.yaml','w').write(yaml.safe_dump(p, sort_keys=False))"
dvc repro
git add params.yaml dvc.lock metrics.json reports/roc.csv
git commit -m "exp: increase C to 2.0"
git push -u origin exp-higher-C
```

**Student B**:
```bash
git checkout -b exp-more-features
python -c "import yaml; p=yaml.safe_load(open('params.yaml')); p['data']['n_features']=30; p['data']['n_informative']=15; open('params.yaml','w').write(yaml.safe_dump(p, sort_keys=False))"
dvc repro
git add params.yaml dvc.lock metrics.json reports/roc.csv
git commit -m "exp: n_features=30"
git push -u origin exp-more-features
```

Then open PRs and compare metrics in GitHub, and also locally:
```bash
git checkout main
git pull
dvc metrics diff
```

### Option B: Use DVC experiments (fast, no commits needed)
```bash
dvc exp run -S train.C=0.2
dvc exp run -S train.C=2.0
dvc exp show
```

---

## 4) Optional: Add a DVC remote (for larger artifacts)
For classroom use, a **local** remote works:
```bash
mkdir -p .dvcstore
dvc remote add -d localstore .dvcstore
git add .dvc/config
git commit -m "chore: add local dvc remote"
dvc push
```

For Colab: see `notebooks/DVC_Collab_Tutorial.ipynb`.

---

## Repo structure
- `src/` — pipeline code
- `dvc.yaml` — stages
- `params.yaml` — editable parameters
- `metrics.json` — DVC-tracked metrics
- `reports/roc.csv` — DVC plot data
- `data/` — synthetic dataset (DVC-tracked)
- `models/` — trained model artifact (DVC-tracked)
