# Organisation de l'équipe et planning

> Remplacez « Étudiant 1…4 » par vos noms. Ce fichier sert de base à la réunion de suivi
> et au tableau « Contributions » exigé à la fin du papier.

## 1. Rôles (chacun code ET rédige)

| Étudiant | Code | Expériences | Sections du papier |
|---|---|---|---|
| **1 — Pipeline & coordination** | `train.py` générique, configs YAML, gel/dégel des couches, MLflow | ConvNeXtV2 × 3 stratégies | III.1 Workflow, III.6 Paramètres, VII Conclusion |
| **2 — Données & baseline** | `make_splits.py`, `eda.py`, Dataset PyTorch, augmentations Albumentations, CNN from scratch | CNN scratch, AlexNet × 3 stratégies, validation croisée du baseline | III.2 Dataset, III.3 Prétraitement, III.4 Augmentation |
| **3 — Modèles avancés & coût** | Chargement timm, AMP, accumulation de gradients, mesures (temps, mémoire GPU, paramètres, latence) | ViT-B/16 et NFNet-F1 × 3 stratégies, étude des hyperparamètres | III.5 Modèles, IV.1 Matériel/logiciel, IV.7 Analyse coût/performance |
| **4 — Évaluation & rédaction** | `evaluate.py` (métriques, ROC/AUC, matrices de confusion), Grad-CAM, explicabilité ViT, figures | Validation croisée du meilleur modèle, analyse des erreurs | Abstract, I Introduction, II État de l'art, V Interprétabilité, intégration finale |

Tout le monde : **2-3 références** pour l'état de l'art, relecture croisée du papier, commits réguliers.

## 2. Conditions d'entraînement communes (exigence du PDF)

Même split (`data/splits/`), même seed (42), même résolution (224×224), même normalisation
ImageNet, mêmes epochs, même optimiseur (AdamW), même scheduler (cosine) et même early stopping.
Tout est défini dans `configs/base.yaml`.

**Validation croisée** — 12 configurations × 5 folds est hors de portée de nos GPU.
Proposition (à valider avec l'encadrant) : un split fixe pour toute la comparaison, puis une
validation croisée stratifiée en 5 folds pour le CNN baseline et pour la meilleure configuration.

**NFNet-F1** — entraîné en 224 px au lieu de sa résolution native (plus élevée) pour respecter
les conditions communes et nos limites GPU. Ce compromis sera mentionné dans le papier.

## 3. Planning

| Semaine | Jalon | Livrable vérifiable |
|---|---|---|
| **S1 (→ lundi)** | Lancement | Dépôt GitHub, EDA, split figé, rôles, plan du papier, références |
| S2 | Pipeline + baseline | `train.py` validé sur le CNN scratch, AlexNet entraîné, MLflow opérationnel |
| S3 | Feature extraction | Les 4 modèles pré-entraînés en feature extraction |
| S4 | Fine-tuning | Partial et full fine-tuning sur les 4 modèles |
| S5 | Évaluation | Tableau global, ROC/AUC, Grad-CAM, validation croisée, analyse d'erreurs |
| S6 | Finalisation | Papier complet en anglais, README final, vérification contre le PDF |

Règle : **aucun modèle lourd n'est entraîné avant que le pipeline et le baseline soient validés (S2).**

## 4. Checklist pour la réunion de lundi

- [ ] Dépôt GitHub créé, 4 membres invités, premiers commits de chacun
- [ ] Dataset téléchargé, `make_splits.py` exécuté (nombre d'images, classes, doublons)
- [ ] Figures EDA dans `results/eda/`
- [ ] Rôles validés (ce fichier, avec vos noms)
- [ ] Plan du papier (`paper/outline.md`) + références vérifiées
- [ ] Questions pour l'encadrant (ci-dessous)
- [ ] Bonus : premier entraînement court du CNN baseline

## 5. Questions à poser à l'encadrant

1. La validation croisée doit-elle porter sur toutes les configurations, ou le split fixe + CV sur
   le baseline et le meilleur modèle suffit-il ?
2. Le CNN from scratch doit-il figurer dans le tableau global des résultats ?
3. Peut-on utiliser une variante réduite de ConvNeXtV2 (Tiny) si la Base est trop lourde ?
4. Quel format pour le papier (template IEEE/Springer, nombre de pages) ?
5. Date exacte de la présentation et du dépôt final ?
