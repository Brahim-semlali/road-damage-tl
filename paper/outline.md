# Paper outline (English) — follows the professor's required structure

**Title (draft):** Road Damage Classification via Transfer Learning: A Comparative Study of
Feature Extraction, Partial and Full Fine-Tuning across CNN and Transformer Architectures

**Abstract** — objective, methodology, data, results (written last). *(Student 4)*
**Keywords (5-6):** road damage classification; transfer learning; fine-tuning; convolutional neural networks; vision transformer; Grad-CAM
**GitHub:** https://github.com/Brahim-semlali/road-damage-tl

## I. Introduction *(Student 4)*
1. Context — importance of road maintenance (safety, economy, transport); CV/DL for damage detection.
2. Problem statement — manual inspection is slow, costly, subjective, dangerous; road-specific CV
   challenges (lighting, shadows, occlusion, texture variety); CNNs need large data; annotated data is limited;
   can transfer learning help?
3. Contributions — comparison of pretrained architectures; impact of fine-tuning depth;
   transfer-learning analysis for **road damage** classification; Grad-CAM interpretability.
4. Paper organisation.

## II. Related Work *(Student 4, references from everyone)*
1. Road damage detection: classical image processing → classical ML → deep learning.
2. Recent works table:

| Reference | Year | Dataset | Model | Result (Accuracy) | Limitations |
|---|---|---|---|---|---|
| | | | | | |

3. Positioning of our work.

## III. Methodology
1. Proposed workflow — diagram. *(Student 1)*
2. Dataset description — source, #images, #classes, images per class, split. *(Student 2, from `results/eda/dataset_table.md`)*
3. Preprocessing — why, resizing, normalisation. *(Student 2)*
4. Data augmentation — why, procedure. *(Student 2)*
5. Models — scratch CNN, AlexNet, ConvNeXtV2, NFNet-F1, ViT-B/16; justification of the choice
   (different families, sizes and eras). *(Student 3)*
6. Training parameters — optimizer, LR, batch size, epochs, cross-validation. *(Student 1)*
7. Evaluation metrics — definitions and formulas. *(Student 4)*

## IV. Results
1. Hardware & software. *(Student 3)*
2. Learning curves (train/val accuracy and loss).
3. Confusion matrices.
4. ROC curves.
5. AUC.
6. Global results table:

| Model | Strategy | Accuracy | Precision | Recall | F1-score | AUC | #Params | Training time |
|---|---|---|---|---|---|---|---|---|
| CNN (scratch) | — | | | | | | | |
| AlexNet | FE / PFT / FFT | | | | | | | |
| ConvNeXtV2 | FE / PFT / FFT | | | | | | | |
| NFNet-F1 | FE / PFT / FFT | | | | | | | |
| ViT-Base/16 | FE / PFT / FFT | | | | | | | |

7. Comparative analysis — most accurate? lightest? most robust?

## V. Interpretability *(Student 4)*
Grad-CAM figures: damaged areas, detected cues; attention-based maps for ViT.

## VI. Discussion
Strengths · Limitations (controlled dataset, real-world generalisation, class imbalance) ·
Perspectives (multimodal learning, smartphone deployment, real-time drone detection).

## VII. Conclusion *(Student 1)*

## Contributions table *(mandatory)*

| Student | Contribution |
|---|---|
| Student 1 | |
| Student 2 | |
| Student 3 | |
| Student 4 | |

## References — starting list

> ⚠️ Each reference must be **checked on Google Scholar** (authors, venue, year) and actually read
> before being cited. Add the BibTeX to `paper/references.bib`.

**Models & methods**
1. A. Krizhevsky, I. Sutskever, G. E. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks," NeurIPS, 2012.
2. S. Woo et al., "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders," CVPR, 2023.
3. A. Brock, S. De, S. L. Smith, K. Simonyan, "High-Performance Large-Scale Image Recognition Without Normalization," ICML, 2021.
4. A. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," ICLR, 2021.
5. R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," ICCV, 2017.
6. S. Abnar, W. Zuidema, "Quantifying Attention Flow in Transformers," ACL, 2020. *(attention rollout for ViT)*

**Transfer learning**
7. S. J. Pan, Q. Yang, "A Survey on Transfer Learning," IEEE TKDE, 2010.
8. J. Yosinski, J. Clune, Y. Bengio, H. Lipson, "How transferable are features in deep neural networks?," NeurIPS, 2014.
9. S. Kornblith, J. Shlens, Q. V. Le, "Do Better ImageNet Models Transfer Better?," CVPR, 2019.

**Data augmentation**
10. C. Shorten, T. M. Khoshgoftaar, "A survey on Image Data Augmentation for Deep Learning," Journal of Big Data, 2019.
11. A. Buslaev et al., "Albumentations: Fast and Flexible Image Augmentations," Information, 2020.

**Road damage**
12. H. Maeda et al., "Road Damage Detection and Classification Using Deep Neural Networks with Smartphone Images," Computer-Aided Civil and Infrastructure Engineering, 2018.
13. L. Zhang, F. Yang, Y. D. Zhang, Y. J. Zhu, "Road crack detection using deep convolutional neural network," IEEE ICIP, 2016.
14. D. Arya et al., "RDD2022: A multi-national image dataset for automatic Road Damage Detection," arXiv, 2022.
15. C. Koch et al., "A review on computer vision based defect detection and condition assessment of concrete and asphalt civil infrastructure," Advanced Engineering Informatics, 2015.

**To find (Student 4 + everyone):** recent (2021-2025) papers on road damage *classification* with
transfer learning, ideally using the same Kaggle dataset, to fill the Related Work table.
