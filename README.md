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

> **Status:** experiments in progress — 4 of 5 models trained (scratch CNN, AlexNet, ConvNeXtV2, ViT-B/16); NFNet-F1 running. Current results: [`results/global_results.md`](results/global_results.md).

## Dataset

Road Damage Classification and Assessment (Kaggle):
<https://www.kaggle.com/datasets/prudhvignv/road-damage-classification-and-assessment>

After cleaning: **2,066 images, 4 classes** (road condition levels), stratified split 70/15/15:

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| good | 592 | 126 | 127 | 845 |
| satisfactory | 358 | 77 | 77 | 512 |
| poor | 275 | 59 | 59 | 393 |
| very_poor | 221 | 48 | 47 | 316 |
| **Total** | **1,446** | **310** | **310** | **2,066** |

> ⚠️ **Dataset bias:** each class comes from a different image source (`good`: 1929×1430 dashcam
> images; `poor`/`satisfactory`: 600×600 street images; `very_poor`: web photos of various sizes).
> Models may exploit these source cues instead of the road surface; we check this with Grad-CAM
> and by analysing the `poor` vs `satisfactory` confusion (same source).

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
2. No setup needed. If Kaggle asks for authentication, log in in the box shown in the cell output.
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

### Training

One script trains every model with every strategy under the same conditions (`configs/base.yaml`):

```bash
# Baseline CNN from scratch
python -m src.train --model scratch_cnn --data-root data/raw --out runs

# Pretrained models: alexnet | convnextv2 | nfnet_f1 | vit_b16
# Strategies: feature_extraction | partial_finetune | full_finetune
python -m src.train --model convnextv2 --strategy feature_extraction --data-root data/raw --out runs

# Quick sanity check (1 epoch, 2 batches)
python -m src.train --model alexnet --strategy full_finetune --epochs 1 --max-batches 2
```

On Colab, use [`notebooks/01_train_colab.ipynb`](notebooks/01_train_colab.ipynb) (choose `MODEL` in the first cell;
its 3 strategies are trained one after the other). Each run saves its learning curves, confusion matrix, test metrics
(accuracy, macro precision/recall/F1, AUC), parameter count, training time, peak GPU memory,
inference latency and test predictions, and logs them to MLflow.

| Strategy | Trainable part (see `src/models/factory.py`) |
|---|---|
| Feature extraction | new classification layer only |
| Partial fine-tuning | last backbone stage + head (AlexNet: last 2 conv + FC layers; ViT: last 4 blocks) |
| Full fine-tuning | all layers (backbone LR = 1e-5, head LR = 1e-3) |

### Comparison and explainability

```bash
# Global results table, ROC curves, confusion matrices, learning curves, performance vs cost
python -m src.compare --runs runs --out results

# Grad-CAM (all 5 architectures, incl. ViT) + road-focus ratio, for one run
python -m src.explain.gradcam --run runs/<model>__<strategy>__<timestamp> --data-root data/raw
```

On Colab: [`notebooks/02_analysis_colab.ipynb`](notebooks/02_analysis_colab.ipynb).

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
