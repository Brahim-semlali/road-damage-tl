# Road Damage Classification via Transfer Learning

**Master ISI S3 — Deep Learning project (FSSM, Université Cadi Ayyad), 2025-2026**

We compare a CNN trained from scratch with four ImageNet-pretrained architectures
(**AlexNet, ConvNeXtV2, NFNet-F1, ViT-Base/16**) for road-damage image classification,
under three transfer-learning strategies:

| Strategy | What is trained |
|---|---|
| Feature extraction | Backbone frozen, only the classification head is trained |
| Partial fine-tuning | Last blocks of the backbone + head |
| Full fine-tuning | Whole network (lower learning rate on the backbone) |

Models are compared on accuracy, precision, recall, F1-score, ROC/AUC, number of parameters,
training time and inference latency. Predictions are explained with Grad-CAM.

> **Status:** project setup — data pipeline in progress. Results will be added once experiments are run.

## Dataset

Road Damage Classification and Assessment (Kaggle):
<https://www.kaggle.com/datasets/prudhvignv/road-damage-classification-and-assessment>

Images are **not** stored in this repository. Only the split files (`data/splits/*.csv`)
are versioned, so that every team member uses exactly the same train/validation/test images.

## Repository structure

```
configs/          experiment configurations (YAML) — shared training conditions in base.yaml
data/raw/         raw images (git-ignored)
data/splits/      reproducible train/val/test CSVs (versioned)
notebooks/        exploration notebooks
src/data/         make_splits.py (cleaning + split), eda.py (figures)
src/models/       model factory and freezing strategies
src/explain/      Grad-CAM / ViT explainability
src/utils/        reproducibility helpers
results/          figures and tables used in the paper
paper/            scientific paper (English)
docs/             team organisation and planning
```

## Installation

```bash
git clone https://github.com/Brahim-semlali/road-damage-tl.git
cd road-damage-tl
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Running on Google Colab (recommended)

All experiments are run on Google Colab (free T4 GPU).

1. Open [`notebooks/00_colab_setup.ipynb`](notebooks/00_colab_setup.ipynb) in Colab
   (*File > Open notebook > GitHub*, paste the repository URL).
2. Add the Colab Secrets `KAGGLE_USERNAME` and `KAGGLE_KEY` (and `GITHUB_TOKEN` if the repo is private).
3. Run all cells: the notebook mounts Google Drive, clones this repository, downloads the dataset
   (cached on Drive), creates the split and the EDA figures.

Persistent files (dataset archive, checkpoints, MLflow runs, results) are stored on Google Drive in
`MyDrive/road-damage-tl/`; images are unzipped to the local Colab disk at each session for speed.

## Local usage

```bash
# 1. Download the dataset (requires a Kaggle API token in ~/.kaggle/kaggle.json)
kaggle datasets download -d prudhvignv/road-damage-classification-and-assessment -p data/raw --unzip

# 2. Clean the data and create the reproducible stratified split (70/15/15, seed 42)
python -m src.data.make_splits --raw data/raw

# 3. Exploratory data analysis (figures in results/eda/)
python -m src.data.eda --raw data/raw
```

Training and evaluation commands will be documented here as they are implemented.

## Reproducibility

- Fixed seed (42) for splitting and training.
- Corrupted images and exact duplicates (MD5) are removed **before** splitting, to avoid train/test leakage.
- All models share the same training conditions (`configs/base.yaml`).
- Experiments are tracked with MLflow.

## Team

| Member | Main responsibilities |
|---|---|
| Student 1 | Training pipeline, ConvNeXtV2, transfer-learning strategies, MLflow, coordination |
| Student 2 | EDA, cleaning, preprocessing & augmentation, CNN from scratch, AlexNet |
| Student 3 | ViT-Base/16, NFNet-F1, computational cost measurements, hyperparameter study |
| Student 4 | Evaluation metrics, Grad-CAM, figures, comparative analysis, paper integration |

See [`docs/team_and_planning.md`](docs/team_and_planning.md) for the detailed plan.
