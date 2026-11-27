| Model | Strategy | Accuracy | Precision | Recall | F1-score | AUC | Params (M) | Trainable (M) | Training time (min) | Peak GPU mem (MB) | Latency (ms) | Best epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CNN (scratch) | From scratch | 0.9290 | 0.9244 | 0.9116 | 0.9160 | 0.9851 | 4.72 | 4.72 | 11.09 | 1128 | 2.53 | 28 |
| AlexNet | Feature extraction | 0.9484 | 0.9500 | 0.9342 | 0.9382 | 0.9811 | 57.02 | 0.02 | 9.81 | 344 | 2.12 | 21 |
| AlexNet | Partial fine-tuning | 0.9548 | 0.9551 | 0.9427 | 0.9457 | 0.9840 | 57.02 | 56.03 | 12.18 | 1110 | 2.11 | 30 |
| AlexNet | Full fine-tuning | 0.9581 | 0.9607 | 0.9459 | **0.9499** | 0.9840 | 57.02 | 57.02 | 10.56 | 1125 | 2.13 | 19 |
