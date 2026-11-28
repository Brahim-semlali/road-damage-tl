| Model | Strategy | Runs | Accuracy | Precision | Recall | F1-score | AUC | Params (M) | Trainable (M) | Training time (min) | Peak GPU mem (MB) | Latency (ms) | Best epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CNN (scratch) | From scratch | 1 | 0.9290 | 0.9244 | 0.9116 | 0.9160 | 0.9851 | 4.72 | 4.72 | 11.09 | 1128 | 2.53 | 28 |
| AlexNet | Feature extraction | 3 | 0.9473 | 0.9485 | 0.9331 | 0.9369 ± 0.0018 | 0.9793 | 57.02 | 0.02 | 9.02 | 344 | 2.05 | 11 |
| AlexNet | Partial fine-tuning | 3 | 0.9548 | 0.9572 | 0.9420 | 0.9458 ± 0.0041 | 0.9829 | 57.02 | 56.03 | 11.10 | 1110 | 2.11 | 30 |
| AlexNet | Full fine-tuning | 3 | 0.9527 | 0.9545 | 0.9395 | 0.9432 ± 0.0062 | 0.9867 | 57.02 | 57.02 | 10.01 | 1125 | 2.07 | 18 |
| ConvNeXtV2-T | Feature extraction | 1 | 0.9290 | 0.9221 | 0.9150 | 0.9168 | 0.9866 | 27.87 | 0.00 | 10.01 | 641 | 8.61 | 20 |
| ConvNeXtV2-T | Partial fine-tuning | 1 | 0.9387 | 0.9303 | 0.9287 | 0.9292 | 0.9895 | 27.87 | 15.49 | 11.48 | 760 | 8.54 | 24 |
| ConvNeXtV2-T | Full fine-tuning | 1 | 0.9516 | 0.9464 | 0.9414 | 0.9432 | 0.9949 | 27.87 | 27.87 | 12.11 | 3803 | 12.30 | 19 |
| NFNet-F1 | Feature extraction | 1 | 0.9387 | 0.9343 | 0.9244 | 0.9273 | 0.9874 | 129.57 | 0.01 | 14.53 | 699 | 42.87 | 30 |
| NFNet-F1 | Partial fine-tuning | 1 | 0.9516 | 0.9501 | 0.9394 | 0.9426 | 0.9851 | 129.57 | 46.09 | 16.03 | 1320 | 30.05 | 25 |
| NFNet-F1 | Full fine-tuning | 1 | 0.9581 | 0.9558 | 0.9479 | 0.9505 | 0.9881 | 129.57 | 129.57 | 14.60 | 4145 | 31.44 | 13 |
| ViT-B/16 | Feature extraction | 1 | 0.9452 | 0.9402 | 0.9329 | 0.9353 | 0.9891 | 85.80 | 0.00 | 12.20 | 529 | 17.86 | 27 |
| ViT-B/16 | Partial fine-tuning | 1 | 0.9516 | 0.9501 | 0.9394 | 0.9426 | 0.9887 | 85.80 | 28.36 | 9.14 | 1394 | 14.59 | 14 |
| ViT-B/16 | Full fine-tuning | 1 | 0.9613 | 0.9633 | 0.9501 | **0.9539** | 0.9917 | 85.80 | 85.80 | 12.55 | 3292 | 15.47 | 19 |

Metrics are the mean over the repeated runs (Runs > 1: mean ± std of the macro F1); figures use the median-F1 run of each model/strategy.
