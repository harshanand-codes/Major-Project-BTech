# Results — ViT-Base/16 full pipeline (original `/root/project`)

Full pipeline: ViT-Base/16 encoder + Prompt-Guided Attention + Cross-Frame
Correspondence + BioMed CLIP text encoder, with all four loss terms (L1: Dice+BCE,
L2: VL-alignment, L3: feature-correspondence, L4: temporal). Configuration:
`configs/config.yaml`. Trained for the full 100 epochs (no early stopping triggered).

## Dataset: Image Segmentation + Video Segmentation Dataset

| Split | Image seg | Video seg  |
|-------|-----------|------------|
| Train | 800       | 934 pairs  |
| Val   | 100       | 259 pairs  |
| Test  | 100       | 1009 pairs |

**Trainable params:** 171,040,809 / **Total:** 366,943,530
(Difference is the frozen BioMed CLIP text encoder.)

## Test Results

### Image Test (100 images)

| Metric | Value |
|--------|-------|
| Dice | 0.9128 |
| IoU | 0.8504 |
| F1 | 0.9128 |
| Hausdorff Distance | 17.64 |
| Loss | 0.9432 |

### Video Test (1009 pairs)

| Metric | Value |
|--------|-------|
| Dice | 0.7541 |
| IoU | 0.6919 |
| F1 | 0.7541 |
| Hausdorff Distance | 43.38 |
| Loss | 2.6543 |

**Best validation Dice:** 0.9283 (epoch 85)



## Qualitative Predictions — 5 image samples + 5 video samples

5 held-out test samples spanning small to large polyp coverage from each dataset.
Predictions use the best model checkpoint, threshold 0.5, green overlay.

### Image Test samples (5)

Picked at the 10th / 30th / 50th / 70th / 90th percentile of GT polyp area
to span small to large polyps.

| # | Filename | Prompt | GT polyp area | Input | Ground Truth | Prediction (overlay) |
|---|----------|--------|--------------:|-------|--------------|----------------------|
| 1 | `cju5ddda9bkkt0850enzwatb1.jpg` | small round polyp | 3.4% | ![input](samples/image_1_input.png) | ![gt](samples/image_1_gt_mask.png) | ![pred](samples/image_1_pred_overlay.png) |
| 2 | `cjyzu9th0qt4r0a46pyl4zik0.jpg` | medium round polyp | 6.6% | ![input](samples/image_2_input.png) | ![gt](samples/image_2_gt_mask.png) | ![pred](samples/image_2_pred_overlay.png) |
| 3 | `cju5yjq1pmlgc0801z0t24bly.jpg` | medium round polyp | 12.4% | ![input](samples/image_3_input.png) | ![gt](samples/image_3_gt_mask.png) | ![pred](samples/image_3_pred_overlay.png) |
| 4 | `cju320gyvbch60801v2amdi2g.jpg` | large irregular polyp | 20.8% | ![input](samples/image_4_input.png) | ![gt](samples/image_4_gt_mask.png) | ![pred](samples/image_4_pred_overlay.png) |
| 5 | `cju0qx73cjw570799j4n5cjze.jpg` | large irregular polyp | 31.9% | ![input](samples/image_5_input.png) | ![gt](samples/image_5_gt_mask.png) | ![pred](samples/image_5_pred_overlay.png) |

### Video Test samples (5)

One pair from each of the 5 distinct test sequences with a representative polyp.

| # | Sequence | Frame pair | Prompt | GT polyp area | Frame 1 | Frame 2 (correspondence) | Ground Truth | Prediction (overlay) |
|---|----------|------------|--------|--------------:|---------|--------------------------|--------------|----------------------|
| 1 | seq16 | 2.jpg & 3.jpg | large round polyp | 36.8% | ![f1](samples/video_1_input_frame1.png) | ![f2](samples/video_1_input_frame2.png) | ![gt](samples/video_1_gt_mask.png) | ![pred](samples/video_1_pred_overlay.png) |
| 2 | seq14 | 0.jpg & 1.jpg | medium round polyp | 13.2% | ![f1](samples/video_2_input_frame1.png) | ![f2](samples/video_2_input_frame2.png) | ![gt](samples/video_2_gt_mask.png) | ![pred](samples/video_2_pred_overlay.png) |
| 3 | seq5 | 0.jpg & 1.jpg | medium round polyp | 9.9% | ![f1](samples/video_3_input_frame1.png) | ![f2](samples/video_3_input_frame2.png) | ![gt](samples/video_3_gt_mask.png) | ![pred](samples/video_3_pred_overlay.png) |
| 4 | seq6 | 24.jpg & 25.jpg | medium irregular polyp | 12.6% | ![f1](samples/video_4_input_frame1.png) | ![f2](samples/video_4_input_frame2.png) | ![gt](samples/video_4_gt_mask.png) | ![pred](samples/video_4_pred_overlay.png) |
| 5 | seq13 | 1.jpg & 2.jpg | small irregular polyp | 2.4% | ![f1](samples/video_5_input_frame1.png) | ![f2](samples/video_5_input_frame2.png) | ![gt](samples/video_5_gt_mask.png) | ![pred](samples/video_5_pred_overlay.png) |

## Training Curves

Overview of the full 100-epoch run (best image-val Dice marked at epoch 85):

![Training summary](plots/summary.png)

Individual plots:

![Loss curves (train / img_val / vid_val, log scale)](plots/losses.png)

![Dice over epochs](plots/dice.png)

![IoU over epochs](plots/iou.png)

![F1 over epochs](plots/f1.png)

![Hausdorff Distance over epochs](plots/hausdorff.png)

![Train sub-loss decomposition (L1-L4, log scale)](plots/sublosses.png)

![Learning-rate schedule (cosine)](plots/lr.png)

## Training Log

```
Epoch 001/100 (56.1s) | LR: 1.00e-04 | Train Loss: 6.1755 (L1: 2.4083, L2: 0.6479, L3: 0.2949, L4: 3.5178)
  Img Val  | Loss: 2.2514 | Dice: 0.7704 | IoU: 0.6460 | F1: 0.7704 | HD: 42.94
  Vid Val  | Loss: 4.5158 | Dice: 0.6722 | IoU: 0.6013 | F1: 0.6722 | HD: 52.21
  -> New best model saved (Dice: 0.7704)
Epoch 002/100 (56.3s) | LR: 9.99e-05 | Train Loss: 4.5092 (L1: 2.0861, L2: 0.2191, L3: 0.0982, L4: 2.3091)
  Img Val  | Loss: 2.0523 | Dice: 0.7137 | IoU: 0.5757 | F1: 0.7137 | HD: 47.35
  Vid Val  | Loss: 4.4428 | Dice: 0.6909 | IoU: 0.6288 | F1: 0.6909 | HD: 48.18
Epoch 003/100 (50.2s) | LR: 9.98e-05 | Train Loss: 3.8697 (L1: 1.9626, L2: 0.1437, L3: 0.0747, L4: 1.7568)
  Img Val  | Loss: 1.7177 | Dice: 0.8145 | IoU: 0.7101 | F1: 0.8145 | HD: 31.63
  Vid Val  | Loss: 4.1703 | Dice: 0.7446 | IoU: 0.6701 | F1: 0.7446 | HD: 47.57
  -> New best model saved (Dice: 0.8145)
Epoch 004/100 (45.5s) | LR: 9.96e-05 | Train Loss: 3.3925 (L1: 1.7767, L2: 0.0927, L3: 0.0488, L4: 1.4872)
  Img Val  | Loss: 1.6236 | Dice: 0.8399 | IoU: 0.7424 | F1: 0.8399 | HD: 32.50
  Vid Val  | Loss: 4.0014 | Dice: 0.8271 | IoU: 0.7531 | F1: 0.8271 | HD: 29.66
  -> New best model saved (Dice: 0.8399)
Epoch 005/100 (49.0s) | LR: 9.94e-05 | Train Loss: 3.0725 (L1: 1.6576, L2: 0.0596, L3: 0.0345, L4: 1.2955)
  Img Val  | Loss: 1.6164 | Dice: 0.8596 | IoU: 0.7682 | F1: 0.8596 | HD: 27.10
  Vid Val  | Loss: 3.9123 | Dice: 0.8078 | IoU: 0.7390 | F1: 0.8078 | HD: 30.24
  -> New best model saved (Dice: 0.8596)
Epoch 006/100 (45.6s) | LR: 9.91e-05 | Train Loss: 2.9350 (L1: 1.5367, L2: 0.0568, L3: 0.0304, L4: 1.3092)
  Img Val  | Loss: 1.4916 | Dice: 0.8730 | IoU: 0.7875 | F1: 0.8730 | HD: 27.11
  Vid Val  | Loss: 3.8248 | Dice: 0.8373 | IoU: 0.7593 | F1: 0.8373 | HD: 28.21
  -> New best model saved (Dice: 0.8730)
Epoch 007/100 (48.3s) | LR: 9.88e-05 | Train Loss: 2.7166 (L1: 1.4296, L2: 0.0494, L3: 0.0276, L4: 1.2032)
  Img Val  | Loss: 1.5078 | Dice: 0.8666 | IoU: 0.7817 | F1: 0.8666 | HD: 28.84
  Vid Val  | Loss: 3.8200 | Dice: 0.7978 | IoU: 0.7145 | F1: 0.7978 | HD: 31.52
Epoch 008/100 (45.4s) | LR: 9.84e-05 | Train Loss: 2.5360 (L1: 1.3290, L2: 0.0480, L3: 0.0272, L4: 1.1295)
  Img Val  | Loss: 1.4154 | Dice: 0.8872 | IoU: 0.8111 | F1: 0.8872 | HD: 24.43
  Vid Val  | Loss: 3.7070 | Dice: 0.8598 | IoU: 0.7945 | F1: 0.8598 | HD: 21.38
  -> New best model saved (Dice: 0.8872)
Epoch 009/100 (45.2s) | LR: 9.80e-05 | Train Loss: 2.3732 (L1: 1.2285, L2: 0.0369, L3: 0.0218, L4: 1.0870)
  Img Val  | Loss: 1.3425 | Dice: 0.8945 | IoU: 0.8200 | F1: 0.8945 | HD: 22.50
  Vid Val  | Loss: 3.6316 | Dice: 0.8546 | IoU: 0.7890 | F1: 0.8546 | HD: 23.68
  -> New best model saved (Dice: 0.8945)
Epoch 010/100 (45.1s) | LR: 9.76e-05 | Train Loss: 2.2536 (L1: 1.1593, L2: 0.0292, L3: 0.0198, L4: 1.0474)
  Img Val  | Loss: 1.2472 | Dice: 0.9026 | IoU: 0.8330 | F1: 0.9026 | HD: 19.33
  Vid Val  | Loss: 3.5890 | Dice: 0.8644 | IoU: 0.7965 | F1: 0.8644 | HD: 20.14
  -> New best model saved (Dice: 0.9026)
Epoch 011/100 (45.3s) | LR: 9.70e-05 | Train Loss: 2.1043 (L1: 1.0694, L2: 0.0351, L3: 0.0234, L4: 0.9898)
  Img Val  | Loss: 1.2681 | Dice: 0.8856 | IoU: 0.8112 | F1: 0.8856 | HD: 22.01
  Vid Val  | Loss: 3.6527 | Dice: 0.8468 | IoU: 0.7730 | F1: 0.8468 | HD: 23.94
Epoch 012/100 (45.1s) | LR: 9.65e-05 | Train Loss: 1.9206 (L1: 1.0043, L2: 0.0357, L3: 0.0240, L4: 0.8571)
  Img Val  | Loss: 1.2175 | Dice: 0.9051 | IoU: 0.8361 | F1: 0.9051 | HD: 19.99
  Vid Val  | Loss: 3.5600 | Dice: 0.8165 | IoU: 0.7462 | F1: 0.8165 | HD: 36.71
  -> New best model saved (Dice: 0.9051)
Epoch 013/100 (44.6s) | LR: 9.59e-05 | Train Loss: 1.8183 (L1: 0.9523, L2: 0.0327, L3: 0.0228, L4: 0.8098)
  Img Val  | Loss: 1.1777 | Dice: 0.9102 | IoU: 0.8429 | F1: 0.9102 | HD: 19.56
  Vid Val  | Loss: 3.5703 | Dice: 0.7744 | IoU: 0.7095 | F1: 0.7744 | HD: 41.36
  -> New best model saved (Dice: 0.9102)
Epoch 014/100 (43.8s) | LR: 9.52e-05 | Train Loss: 1.6935 (L1: 0.8758, L2: 0.0264, L3: 0.0179, L4: 0.7756)
  Img Val  | Loss: 1.1897 | Dice: 0.9008 | IoU: 0.8317 | F1: 0.9008 | HD: 22.92
  Vid Val  | Loss: 3.5604 | Dice: 0.7550 | IoU: 0.6963 | F1: 0.7550 | HD: 50.31
Epoch 015/100 (46.1s) | LR: 9.46e-05 | Train Loss: 1.5961 (L1: 0.8196, L2: 0.0330, L3: 0.0225, L4: 0.7311)
  Img Val  | Loss: 1.2173 | Dice: 0.8890 | IoU: 0.8138 | F1: 0.8890 | HD: 25.27
  Vid Val  | Loss: 3.4518 | Dice: 0.8337 | IoU: 0.7741 | F1: 0.8337 | HD: 28.69
Epoch 016/100 (45.2s) | LR: 9.38e-05 | Train Loss: 1.5170 (L1: 0.7948, L2: 0.0261, L3: 0.0183, L4: 0.6764)
  Img Val  | Loss: 1.2184 | Dice: 0.8752 | IoU: 0.7970 | F1: 0.8752 | HD: 27.00
  Vid Val  | Loss: 3.4024 | Dice: 0.8109 | IoU: 0.7402 | F1: 0.8109 | HD: 38.00
Epoch 017/100 (44.5s) | LR: 9.30e-05 | Train Loss: 1.4636 (L1: 0.7497, L2: 0.0239, L3: 0.0173, L4: 0.6793)
  Img Val  | Loss: 1.1511 | Dice: 0.8995 | IoU: 0.8254 | F1: 0.8995 | HD: 21.22
  Vid Val  | Loss: 3.4324 | Dice: 0.8401 | IoU: 0.7776 | F1: 0.8401 | HD: 24.86
Epoch 018/100 (45.8s) | LR: 9.22e-05 | Train Loss: 1.4546 (L1: 0.7524, L2: 0.0350, L3: 0.0248, L4: 0.6523)
  Img Val  | Loss: 1.1454 | Dice: 0.8925 | IoU: 0.8213 | F1: 0.8925 | HD: 21.94
  Vid Val  | Loss: 3.3515 | Dice: 0.7906 | IoU: 0.7214 | F1: 0.7906 | HD: 36.37
Epoch 019/100 (44.6s) | LR: 9.14e-05 | Train Loss: 1.3102 (L1: 0.6977, L2: 0.0250, L3: 0.0186, L4: 0.5640)
  Img Val  | Loss: 1.1275 | Dice: 0.9087 | IoU: 0.8431 | F1: 0.9087 | HD: 19.02
  Vid Val  | Loss: 3.3777 | Dice: 0.7871 | IoU: 0.7224 | F1: 0.7871 | HD: 39.77
Epoch 020/100 (44.7s) | LR: 9.05e-05 | Train Loss: 1.2486 (L1: 0.6465, L2: 0.0227, L3: 0.0170, L4: 0.5661)
  Img Val  | Loss: 1.1370 | Dice: 0.9005 | IoU: 0.8326 | F1: 0.9005 | HD: 20.48
  Vid Val  | Loss: 3.3813 | Dice: 0.8436 | IoU: 0.7829 | F1: 0.8436 | HD: 23.77
Epoch 021/100 (44.6s) | LR: 8.95e-05 | Train Loss: 1.2541 (L1: 0.6318, L2: 0.0216, L3: 0.0168, L4: 0.5960)
  Img Val  | Loss: 1.1089 | Dice: 0.8995 | IoU: 0.8332 | F1: 0.8995 | HD: 20.00
  Vid Val  | Loss: 3.3360 | Dice: 0.8395 | IoU: 0.7800 | F1: 0.8395 | HD: 20.98
Epoch 022/100 (44.6s) | LR: 8.85e-05 | Train Loss: 1.1511 (L1: 0.5954, L2: 0.0167, L3: 0.0111, L4: 0.5283)
  Img Val  | Loss: 1.1037 | Dice: 0.8994 | IoU: 0.8345 | F1: 0.8994 | HD: 19.85
  Vid Val  | Loss: 3.2743 | Dice: 0.8581 | IoU: 0.7983 | F1: 0.8581 | HD: 23.40
Epoch 023/100 (45.1s) | LR: 8.75e-05 | Train Loss: 1.1140 (L1: 0.5922, L2: 0.0202, L3: 0.0136, L4: 0.4832)
  Img Val  | Loss: 1.1780 | Dice: 0.8809 | IoU: 0.8098 | F1: 0.8809 | HD: 22.04
  Vid Val  | Loss: 3.3379 | Dice: 0.8296 | IoU: 0.7695 | F1: 0.8296 | HD: 33.22
Epoch 024/100 (44.0s) | LR: 8.64e-05 | Train Loss: 1.0998 (L1: 0.5852, L2: 0.0190, L3: 0.0134, L4: 0.4767)
  Img Val  | Loss: 1.1236 | Dice: 0.8938 | IoU: 0.8246 | F1: 0.8938 | HD: 20.34
  Vid Val  | Loss: 3.3065 | Dice: 0.8269 | IoU: 0.7674 | F1: 0.8269 | HD: 21.88
Epoch 025/100 (44.5s) | LR: 8.54e-05 | Train Loss: 1.0711 (L1: 0.5489, L2: 0.0209, L3: 0.0135, L4: 0.4940)
  Img Val  | Loss: 1.1023 | Dice: 0.8989 | IoU: 0.8325 | F1: 0.8989 | HD: 20.36
  Vid Val  | Loss: 3.2671 | Dice: 0.8389 | IoU: 0.7759 | F1: 0.8389 | HD: 21.32
Epoch 026/100 (44.5s) | LR: 8.42e-05 | Train Loss: 1.0566 (L1: 0.5504, L2: 0.0214, L3: 0.0161, L4: 0.4718)
  Img Val  | Loss: 1.1068 | Dice: 0.9021 | IoU: 0.8358 | F1: 0.9021 | HD: 18.60
  Vid Val  | Loss: 3.2895 | Dice: 0.8419 | IoU: 0.7800 | F1: 0.8419 | HD: 22.59
Epoch 027/100 (45.0s) | LR: 8.31e-05 | Train Loss: 0.9807 (L1: 0.5108, L2: 0.0174, L3: 0.0137, L4: 0.4403)
  Img Val  | Loss: 1.1108 | Dice: 0.8986 | IoU: 0.8308 | F1: 0.8986 | HD: 19.02
  Vid Val  | Loss: 3.2430 | Dice: 0.8632 | IoU: 0.8042 | F1: 0.8632 | HD: 21.00
Epoch 028/100 (44.3s) | LR: 8.19e-05 | Train Loss: 0.9723 (L1: 0.5153, L2: 0.0155, L3: 0.0106, L4: 0.4262)
  Img Val  | Loss: 1.0802 | Dice: 0.9139 | IoU: 0.8537 | F1: 0.9139 | HD: 18.02
  Vid Val  | Loss: 3.1978 | Dice: 0.8612 | IoU: 0.7987 | F1: 0.8612 | HD: 20.54
  -> New best model saved (Dice: 0.9139)
Epoch 029/100 (45.0s) | LR: 8.06e-05 | Train Loss: 0.9574 (L1: 0.5004, L2: 0.0142, L3: 0.0136, L4: 0.4288)
  Img Val  | Loss: 1.1372 | Dice: 0.8880 | IoU: 0.8206 | F1: 0.8880 | HD: 20.79
  Vid Val  | Loss: 3.2476 | Dice: 0.8252 | IoU: 0.7685 | F1: 0.8252 | HD: 30.82
Epoch 030/100 (45.0s) | LR: 7.94e-05 | Train Loss: 0.9978 (L1: 0.5262, L2: 0.0253, L3: 0.0167, L4: 0.4317)
  Img Val  | Loss: 1.1291 | Dice: 0.8922 | IoU: 0.8202 | F1: 0.8922 | HD: 25.21
  Vid Val  | Loss: 3.2295 | Dice: 0.8074 | IoU: 0.7452 | F1: 0.8074 | HD: 34.13
Epoch 031/100 (44.4s) | LR: 7.81e-05 | Train Loss: 0.9480 (L1: 0.4975, L2: 0.0157, L3: 0.0151, L4: 0.4195)
  Img Val  | Loss: 1.1085 | Dice: 0.8974 | IoU: 0.8323 | F1: 0.8974 | HD: 20.80
  Vid Val  | Loss: 3.2880 | Dice: 0.8062 | IoU: 0.7418 | F1: 0.8062 | HD: 33.37
Epoch 032/100 (44.8s) | LR: 7.68e-05 | Train Loss: 0.8779 (L1: 0.4763, L2: 0.0107, L3: 0.0079, L4: 0.3713)
  Img Val  | Loss: 1.0700 | Dice: 0.9105 | IoU: 0.8469 | F1: 0.9105 | HD: 20.13
  Vid Val  | Loss: 3.2336 | Dice: 0.8434 | IoU: 0.7813 | F1: 0.8434 | HD: 22.45
Epoch 033/100 (45.3s) | LR: 7.55e-05 | Train Loss: 0.9037 (L1: 0.4625, L2: 0.0147, L3: 0.0104, L4: 0.4202)
  Img Val  | Loss: 1.0931 | Dice: 0.9032 | IoU: 0.8390 | F1: 0.9032 | HD: 20.44
  Vid Val  | Loss: 3.1701 | Dice: 0.8785 | IoU: 0.8191 | F1: 0.8785 | HD: 17.95
Epoch 034/100 (44.9s) | LR: 7.41e-05 | Train Loss: 0.8486 (L1: 0.4467, L2: 0.0117, L3: 0.0096, L4: 0.3775)
  Img Val  | Loss: 1.0583 | Dice: 0.9104 | IoU: 0.8464 | F1: 0.9104 | HD: 19.80
  Vid Val  | Loss: 3.1370 | Dice: 0.8396 | IoU: 0.7742 | F1: 0.8396 | HD: 34.13
Epoch 035/100 (45.0s) | LR: 7.27e-05 | Train Loss: 0.8588 (L1: 0.4513, L2: 0.0108, L3: 0.0094, L4: 0.3839)
  Img Val  | Loss: 1.0416 | Dice: 0.9184 | IoU: 0.8581 | F1: 0.9184 | HD: 18.88
  Vid Val  | Loss: 3.1038 | Dice: 0.8678 | IoU: 0.8043 | F1: 0.8678 | HD: 23.35
  -> New best model saved (Dice: 0.9184)
Epoch 036/100 (45.3s) | LR: 7.13e-05 | Train Loss: 0.8592 (L1: 0.4534, L2: 0.0095, L3: 0.0076, L4: 0.3832)
  Img Val  | Loss: 1.1000 | Dice: 0.9096 | IoU: 0.8455 | F1: 0.9096 | HD: 18.96
  Vid Val  | Loss: 3.1745 | Dice: 0.8459 | IoU: 0.7838 | F1: 0.8459 | HD: 23.66
Epoch 037/100 (44.0s) | LR: 6.99e-05 | Train Loss: 0.8544 (L1: 0.4515, L2: 0.0121, L3: 0.0100, L4: 0.3769)
  Img Val  | Loss: 1.0803 | Dice: 0.9007 | IoU: 0.8374 | F1: 0.9007 | HD: 20.78
  Vid Val  | Loss: 3.1293 | Dice: 0.8506 | IoU: 0.7881 | F1: 0.8506 | HD: 25.13
Epoch 038/100 (44.8s) | LR: 6.84e-05 | Train Loss: 0.8060 (L1: 0.4298, L2: 0.0096, L3: 0.0072, L4: 0.3523)
  Img Val  | Loss: 1.0906 | Dice: 0.9017 | IoU: 0.8384 | F1: 0.9017 | HD: 20.60
  Vid Val  | Loss: 3.1506 | Dice: 0.8698 | IoU: 0.8058 | F1: 0.8698 | HD: 21.63
Epoch 039/100 (44.3s) | LR: 6.69e-05 | Train Loss: 0.8010 (L1: 0.4150, L2: 0.0144, L3: 0.0083, L4: 0.3646)
  Img Val  | Loss: 1.0447 | Dice: 0.9161 | IoU: 0.8547 | F1: 0.9161 | HD: 18.19
  Vid Val  | Loss: 3.1495 | Dice: 0.8666 | IoU: 0.8079 | F1: 0.8666 | HD: 22.16
Epoch 040/100 (44.7s) | LR: 6.55e-05 | Train Loss: 0.7745 (L1: 0.4152, L2: 0.0090, L3: 0.0086, L4: 0.3343)
  Img Val  | Loss: 1.0177 | Dice: 0.9268 | IoU: 0.8691 | F1: 0.9268 | HD: 16.54
  Vid Val  | Loss: 3.1084 | Dice: 0.8726 | IoU: 0.8151 | F1: 0.8726 | HD: 17.53
  -> New best model saved (Dice: 0.9268)
Epoch 041/100 (44.3s) | LR: 6.39e-05 | Train Loss: 0.7294 (L1: 0.3964, L2: 0.0044, L3: 0.0032, L4: 0.3124)
  Img Val  | Loss: 1.0179 | Dice: 0.9280 | IoU: 0.8720 | F1: 0.9280 | HD: 16.12
  Vid Val  | Loss: 3.1097 | Dice: 0.8704 | IoU: 0.8105 | F1: 0.8704 | HD: 19.28
  -> New best model saved (Dice: 0.9280)
Epoch 042/100 (44.8s) | LR: 6.24e-05 | Train Loss: 0.7374 (L1: 0.3891, L2: 0.0061, L3: 0.0045, L4: 0.3315)
  Img Val  | Loss: 1.0283 | Dice: 0.9230 | IoU: 0.8646 | F1: 0.9230 | HD: 16.45
  Vid Val  | Loss: 3.1333 | Dice: 0.8646 | IoU: 0.8054 | F1: 0.8646 | HD: 19.91
Epoch 043/100 (44.9s) | LR: 6.09e-05 | Train Loss: 0.7440 (L1: 0.4035, L2: 0.0061, L3: 0.0054, L4: 0.3175)
  Img Val  | Loss: 1.0765 | Dice: 0.9090 | IoU: 0.8434 | F1: 0.9090 | HD: 20.36
  Vid Val  | Loss: 3.1377 | Dice: 0.8686 | IoU: 0.8093 | F1: 0.8686 | HD: 18.95
Epoch 044/100 (44.7s) | LR: 5.94e-05 | Train Loss: 0.8033 (L1: 0.4142, L2: 0.0114, L3: 0.0084, L4: 0.3705)
  Img Val  | Loss: 1.1049 | Dice: 0.8987 | IoU: 0.8330 | F1: 0.8987 | HD: 20.80
  Vid Val  | Loss: 3.1612 | Dice: 0.8507 | IoU: 0.7935 | F1: 0.8507 | HD: 22.21
Epoch 045/100 (44.5s) | LR: 5.78e-05 | Train Loss: 0.7253 (L1: 0.3995, L2: 0.0058, L3: 0.0042, L4: 0.3011)
  Img Val  | Loss: 1.0931 | Dice: 0.9005 | IoU: 0.8349 | F1: 0.9005 | HD: 20.31
  Vid Val  | Loss: 3.1384 | Dice: 0.8608 | IoU: 0.7980 | F1: 0.8608 | HD: 22.88
Epoch 046/100 (44.2s) | LR: 5.63e-05 | Train Loss: 0.6972 (L1: 0.3802, L2: 0.0058, L3: 0.0060, L4: 0.2938)
  Img Val  | Loss: 1.0561 | Dice: 0.9147 | IoU: 0.8531 | F1: 0.9147 | HD: 17.97
  Vid Val  | Loss: 3.1334 | Dice: 0.8596 | IoU: 0.7995 | F1: 0.8596 | HD: 20.07
Epoch 047/100 (44.5s) | LR: 5.47e-05 | Train Loss: 0.7208 (L1: 0.3756, L2: 0.0075, L3: 0.0050, L4: 0.3298)
  Img Val  | Loss: 1.0522 | Dice: 0.9193 | IoU: 0.8615 | F1: 0.9193 | HD: 17.70
  Vid Val  | Loss: 3.0611 | Dice: 0.8621 | IoU: 0.8015 | F1: 0.8621 | HD: 20.58
Epoch 048/100 (44.9s) | LR: 5.31e-05 | Train Loss: 0.7012 (L1: 0.3731, L2: 0.0103, L3: 0.0069, L4: 0.3061)
  Img Val  | Loss: 1.0427 | Dice: 0.9205 | IoU: 0.8640 | F1: 0.9205 | HD: 16.30
  Vid Val  | Loss: 3.0858 | Dice: 0.8723 | IoU: 0.8121 | F1: 0.8723 | HD: 17.99
Epoch 049/100 (45.0s) | LR: 5.16e-05 | Train Loss: 0.6903 (L1: 0.3706, L2: 0.0057, L3: 0.0044, L4: 0.3007)
  Img Val  | Loss: 1.0665 | Dice: 0.9148 | IoU: 0.8515 | F1: 0.9148 | HD: 19.06
  Vid Val  | Loss: 3.1325 | Dice: 0.8672 | IoU: 0.8123 | F1: 0.8672 | HD: 18.26
Epoch 050/100 (44.7s) | LR: 5.00e-05 | Train Loss: 0.6729 (L1: 0.3672, L2: 0.0061, L3: 0.0047, L4: 0.2836)
  Img Val  | Loss: 1.0428 | Dice: 0.9237 | IoU: 0.8650 | F1: 0.9237 | HD: 17.46
  Vid Val  | Loss: 3.0163 | Dice: 0.8699 | IoU: 0.8108 | F1: 0.8699 | HD: 19.53
Epoch 051/100 (44.1s) | LR: 4.84e-05 | Train Loss: 0.6306 (L1: 0.3233, L2: 0.0075, L3: 0.0084, L4: 0.2935)
  Img Val  | Loss: 1.0644 | Dice: 0.9151 | IoU: 0.8546 | F1: 0.9151 | HD: 17.53
  Vid Val  | Loss: 2.6818 | Dice: 0.8704 | IoU: 0.8116 | F1: 0.8704 | HD: 20.37
Epoch 052/100 (45.1s) | LR: 4.69e-05 | Train Loss: 0.5828 (L1: 0.2801, L2: 0.0067, L3: 0.0062, L4: 0.3002)
  Img Val  | Loss: 1.0976 | Dice: 0.9098 | IoU: 0.8473 | F1: 0.9098 | HD: 18.97
  Vid Val  | Loss: 2.6755 | Dice: 0.8591 | IoU: 0.8038 | F1: 0.8591 | HD: 21.55
Epoch 053/100 (45.3s) | LR: 4.53e-05 | Train Loss: 0.5348 (L1: 0.2504, L2: 0.0051, L3: 0.0049, L4: 0.2865)
  Img Val  | Loss: 1.0898 | Dice: 0.9103 | IoU: 0.8482 | F1: 0.9103 | HD: 18.58
  Vid Val  | Loss: 2.6183 | Dice: 0.8624 | IoU: 0.8005 | F1: 0.8624 | HD: 22.50
Epoch 054/100 (45.2s) | LR: 4.37e-05 | Train Loss: 0.5111 (L1: 0.2263, L2: 0.0064, L3: 0.0056, L4: 0.2920)
  Img Val  | Loss: 1.0533 | Dice: 0.9192 | IoU: 0.8604 | F1: 0.9192 | HD: 17.89
  Vid Val  | Loss: 2.7325 | Dice: 0.8466 | IoU: 0.7832 | F1: 0.8466 | HD: 24.95
Epoch 055/100 (44.8s) | LR: 4.22e-05 | Train Loss: 0.4644 (L1: 0.2104, L2: 0.0041, L3: 0.0036, L4: 0.2600)
  Img Val  | Loss: 1.0573 | Dice: 0.9162 | IoU: 0.8560 | F1: 0.9162 | HD: 18.39
  Vid Val  | Loss: 2.6194 | Dice: 0.8574 | IoU: 0.7989 | F1: 0.8574 | HD: 23.34
Epoch 056/100 (45.0s) | LR: 4.06e-05 | Train Loss: 0.4922 (L1: 0.2151, L2: 0.0048, L3: 0.0036, L4: 0.2873)
  Img Val  | Loss: 1.0711 | Dice: 0.9166 | IoU: 0.8570 | F1: 0.9166 | HD: 18.30
  Vid Val  | Loss: 2.6619 | Dice: 0.8553 | IoU: 0.7949 | F1: 0.8553 | HD: 27.94
Epoch 057/100 (44.3s) | LR: 3.91e-05 | Train Loss: 0.4788 (L1: 0.2078, L2: 0.0058, L3: 0.0037, L4: 0.2810)
  Img Val  | Loss: 1.0640 | Dice: 0.9217 | IoU: 0.8645 | F1: 0.9217 | HD: 17.62
  Vid Val  | Loss: 2.6412 | Dice: 0.8583 | IoU: 0.8023 | F1: 0.8583 | HD: 20.57
Epoch 058/100 (44.8s) | LR: 3.76e-05 | Train Loss: 0.4450 (L1: 0.1901, L2: 0.0034, L3: 0.0028, L4: 0.2672)
  Img Val  | Loss: 1.0651 | Dice: 0.9177 | IoU: 0.8591 | F1: 0.9177 | HD: 18.43
  Vid Val  | Loss: 2.6339 | Dice: 0.8629 | IoU: 0.8072 | F1: 0.8629 | HD: 19.16
Epoch 059/100 (44.4s) | LR: 3.61e-05 | Train Loss: 0.4324 (L1: 0.1851, L2: 0.0053, L3: 0.0052, L4: 0.2563)
  Img Val  | Loss: 1.0620 | Dice: 0.9174 | IoU: 0.8583 | F1: 0.9174 | HD: 17.82
  Vid Val  | Loss: 2.6116 | Dice: 0.8737 | IoU: 0.8163 | F1: 0.8737 | HD: 20.84
Epoch 060/100 (44.4s) | LR: 3.45e-05 | Train Loss: 0.4095 (L1: 0.1678, L2: 0.0031, L3: 0.0042, L4: 0.2556)
  Img Val  | Loss: 1.0760 | Dice: 0.9149 | IoU: 0.8545 | F1: 0.9149 | HD: 18.71
  Vid Val  | Loss: 2.6339 | Dice: 0.8690 | IoU: 0.8110 | F1: 0.8690 | HD: 20.88
Epoch 061/100 (44.7s) | LR: 3.31e-05 | Train Loss: 0.4328 (L1: 0.1819, L2: 0.0029, L3: 0.0032, L4: 0.2643)
  Img Val  | Loss: 1.0508 | Dice: 0.9205 | IoU: 0.8631 | F1: 0.9205 | HD: 17.67
  Vid Val  | Loss: 2.6250 | Dice: 0.8734 | IoU: 0.8121 | F1: 0.8734 | HD: 20.64
Epoch 062/100 (45.0s) | LR: 3.16e-05 | Train Loss: 0.4356 (L1: 0.1789, L2: 0.0052, L3: 0.0049, L4: 0.2699)
  Img Val  | Loss: 1.0627 | Dice: 0.9202 | IoU: 0.8621 | F1: 0.9202 | HD: 17.78
  Vid Val  | Loss: 2.5688 | Dice: 0.8712 | IoU: 0.8129 | F1: 0.8712 | HD: 19.81
Epoch 063/100 (44.3s) | LR: 3.01e-05 | Train Loss: 0.3880 (L1: 0.1649, L2: 0.0039, L3: 0.0030, L4: 0.2333)
  Img Val  | Loss: 1.0520 | Dice: 0.9223 | IoU: 0.8661 | F1: 0.9223 | HD: 16.81
  Vid Val  | Loss: 2.6022 | Dice: 0.8624 | IoU: 0.8093 | F1: 0.8624 | HD: 17.96
Epoch 064/100 (45.0s) | LR: 2.87e-05 | Train Loss: 0.3875 (L1: 0.1571, L2: 0.0022, L3: 0.0028, L4: 0.2455)
  Img Val  | Loss: 1.0587 | Dice: 0.9217 | IoU: 0.8648 | F1: 0.9217 | HD: 17.61
  Vid Val  | Loss: 2.6120 | Dice: 0.8631 | IoU: 0.8051 | F1: 0.8631 | HD: 19.99
Epoch 065/100 (45.2s) | LR: 2.73e-05 | Train Loss: 0.3791 (L1: 0.1540, L2: 0.0056, L3: 0.0034, L4: 0.2373)
  Img Val  | Loss: 1.0528 | Dice: 0.9218 | IoU: 0.8626 | F1: 0.9218 | HD: 17.51
  Vid Val  | Loss: 2.6145 | Dice: 0.8576 | IoU: 0.8018 | F1: 0.8576 | HD: 20.67
Epoch 066/100 (44.7s) | LR: 2.59e-05 | Train Loss: 0.3722 (L1: 0.1525, L2: 0.0034, L3: 0.0019, L4: 0.2332)
  Img Val  | Loss: 1.0509 | Dice: 0.9208 | IoU: 0.8636 | F1: 0.9208 | HD: 17.03
  Vid Val  | Loss: 2.6242 | Dice: 0.8603 | IoU: 0.8042 | F1: 0.8603 | HD: 20.54
Epoch 067/100 (45.0s) | LR: 2.45e-05 | Train Loss: 0.3914 (L1: 0.1543, L2: 0.0023, L3: 0.0024, L4: 0.2550)
  Img Val  | Loss: 1.0663 | Dice: 0.9208 | IoU: 0.8637 | F1: 0.9208 | HD: 16.78
  Vid Val  | Loss: 2.6338 | Dice: 0.8618 | IoU: 0.8029 | F1: 0.8618 | HD: 21.95
Epoch 068/100 (45.9s) | LR: 2.32e-05 | Train Loss: 0.3991 (L1: 0.1512, L2: 0.0026, L3: 0.0042, L4: 0.2677)
  Img Val  | Loss: 1.0542 | Dice: 0.9212 | IoU: 0.8631 | F1: 0.9212 | HD: 16.90
  Vid Val  | Loss: 2.5837 | Dice: 0.8780 | IoU: 0.8202 | F1: 0.8780 | HD: 20.67
Epoch 069/100 (45.0s) | LR: 2.19e-05 | Train Loss: 0.3608 (L1: 0.1387, L2: 0.0068, L3: 0.0070, L4: 0.2343)
  Img Val  | Loss: 1.0595 | Dice: 0.9217 | IoU: 0.8654 | F1: 0.9217 | HD: 17.30
  Vid Val  | Loss: 2.5357 | Dice: 0.8742 | IoU: 0.8183 | F1: 0.8742 | HD: 19.56
Epoch 070/100 (44.9s) | LR: 2.06e-05 | Train Loss: 0.3795 (L1: 0.1406, L2: 0.0034, L3: 0.0017, L4: 0.2604)
  Img Val  | Loss: 1.0481 | Dice: 0.9232 | IoU: 0.8672 | F1: 0.9232 | HD: 16.32
  Vid Val  | Loss: 2.5224 | Dice: 0.8824 | IoU: 0.8279 | F1: 0.8824 | HD: 18.77
Epoch 071/100 (44.7s) | LR: 1.94e-05 | Train Loss: 0.3696 (L1: 0.1415, L2: 0.0047, L3: 0.0027, L4: 0.2450)
  Img Val  | Loss: 1.0434 | Dice: 0.9265 | IoU: 0.8709 | F1: 0.9265 | HD: 16.62
  Vid Val  | Loss: 2.5282 | Dice: 0.8754 | IoU: 0.8207 | F1: 0.8754 | HD: 20.90
Epoch 072/100 (44.6s) | LR: 1.81e-05 | Train Loss: 0.3625 (L1: 0.1376, L2: 0.0014, L3: 0.0010, L4: 0.2453)
  Img Val  | Loss: 1.0484 | Dice: 0.9229 | IoU: 0.8669 | F1: 0.9229 | HD: 16.70
  Vid Val  | Loss: 2.5487 | Dice: 0.8741 | IoU: 0.8171 | F1: 0.8741 | HD: 22.29
Epoch 073/100 (45.0s) | LR: 1.69e-05 | Train Loss: 0.3680 (L1: 0.1374, L2: 0.0025, L3: 0.0024, L4: 0.2508)
  Img Val  | Loss: 1.0469 | Dice: 0.9253 | IoU: 0.8696 | F1: 0.9253 | HD: 16.59
  Vid Val  | Loss: 2.5426 | Dice: 0.8704 | IoU: 0.8126 | F1: 0.8704 | HD: 22.38
Epoch 074/100 (44.7s) | LR: 1.58e-05 | Train Loss: 0.3382 (L1: 0.1317, L2: 0.0022, L3: 0.0018, L4: 0.2227)
  Img Val  | Loss: 1.0452 | Dice: 0.9251 | IoU: 0.8698 | F1: 0.9251 | HD: 16.81
  Vid Val  | Loss: 2.5590 | Dice: 0.8725 | IoU: 0.8152 | F1: 0.8725 | HD: 21.06
Epoch 075/100 (44.9s) | LR: 1.46e-05 | Train Loss: 0.3474 (L1: 0.1322, L2: 0.0028, L3: 0.0021, L4: 0.2328)
  Img Val  | Loss: 1.0400 | Dice: 0.9278 | IoU: 0.8722 | F1: 0.9278 | HD: 16.64
  Vid Val  | Loss: 2.5679 | Dice: 0.8693 | IoU: 0.8122 | F1: 0.8693 | HD: 20.60
Epoch 076/100 (45.1s) | LR: 1.36e-05 | Train Loss: 0.3597 (L1: 0.1351, L2: 0.0032, L3: 0.0027, L4: 0.2433)
  Img Val  | Loss: 1.0405 | Dice: 0.9273 | IoU: 0.8712 | F1: 0.9273 | HD: 16.79
  Vid Val  | Loss: 2.5766 | Dice: 0.8655 | IoU: 0.8077 | F1: 0.8655 | HD: 20.32
Epoch 077/100 (45.0s) | LR: 1.25e-05 | Train Loss: 0.3310 (L1: 0.1238, L2: 0.0012, L3: 0.0017, L4: 0.2262)
  Img Val  | Loss: 1.0499 | Dice: 0.9230 | IoU: 0.8672 | F1: 0.9230 | HD: 16.98
  Vid Val  | Loss: 2.5521 | Dice: 0.8680 | IoU: 0.8090 | F1: 0.8680 | HD: 21.75
Epoch 078/100 (44.7s) | LR: 1.15e-05 | Train Loss: 0.3464 (L1: 0.1220, L2: 0.0025, L3: 0.0020, L4: 0.2470)
  Img Val  | Loss: 1.0560 | Dice: 0.9225 | IoU: 0.8667 | F1: 0.9225 | HD: 16.94
  Vid Val  | Loss: 2.5616 | Dice: 0.8674 | IoU: 0.8102 | F1: 0.8674 | HD: 20.39
Epoch 079/100 (44.8s) | LR: 1.05e-05 | Train Loss: 0.3432 (L1: 0.1215, L2: 0.0025, L3: 0.0026, L4: 0.2435)
  Img Val  | Loss: 1.0484 | Dice: 0.9258 | IoU: 0.8708 | F1: 0.9258 | HD: 16.60
  Vid Val  | Loss: 2.5682 | Dice: 0.8709 | IoU: 0.8126 | F1: 0.8709 | HD: 20.47
Epoch 080/100 (44.7s) | LR: 9.55e-06 | Train Loss: 0.3277 (L1: 0.1214, L2: 0.0033, L3: 0.0022, L4: 0.2242)
  Img Val  | Loss: 1.0491 | Dice: 0.9248 | IoU: 0.8704 | F1: 0.9248 | HD: 16.69
  Vid Val  | Loss: 2.5374 | Dice: 0.8747 | IoU: 0.8181 | F1: 0.8747 | HD: 19.37
Epoch 081/100 (45.0s) | LR: 8.65e-06 | Train Loss: 0.3140 (L1: 0.1173, L2: 0.0020, L3: 0.0017, L4: 0.2143)
  Img Val  | Loss: 1.0502 | Dice: 0.9244 | IoU: 0.8697 | F1: 0.9244 | HD: 16.78
  Vid Val  | Loss: 2.5376 | Dice: 0.8739 | IoU: 0.8165 | F1: 0.8739 | HD: 19.97
Epoch 082/100 (44.8s) | LR: 7.78e-06 | Train Loss: 0.3477 (L1: 0.1199, L2: 0.0018, L3: 0.0017, L4: 0.2526)
  Img Val  | Loss: 1.0469 | Dice: 0.9259 | IoU: 0.8704 | F1: 0.9259 | HD: 16.71
  Vid Val  | Loss: 2.5394 | Dice: 0.8730 | IoU: 0.8159 | F1: 0.8730 | HD: 19.48
Epoch 083/100 (45.0s) | LR: 6.96e-06 | Train Loss: 0.3231 (L1: 0.1137, L2: 0.0024, L3: 0.0020, L4: 0.2305)
  Img Val  | Loss: 1.0503 | Dice: 0.9274 | IoU: 0.8719 | F1: 0.9274 | HD: 16.85
  Vid Val  | Loss: 2.5380 | Dice: 0.8693 | IoU: 0.8104 | F1: 0.8693 | HD: 20.02
Epoch 084/100 (45.0s) | LR: 6.18e-06 | Train Loss: 0.3026 (L1: 0.1110, L2: 0.0018, L3: 0.0020, L4: 0.2094)
  Img Val  | Loss: 1.0519 | Dice: 0.9272 | IoU: 0.8717 | F1: 0.9272 | HD: 16.89
  Vid Val  | Loss: 2.5313 | Dice: 0.8676 | IoU: 0.8086 | F1: 0.8676 | HD: 20.19
Epoch 085/100 (45.3s) | LR: 5.45e-06 | Train Loss: 0.3577 (L1: 0.1246, L2: 0.0017, L3: 0.0025, L4: 0.2576)
  Img Val  | Loss: 1.0411 | Dice: 0.9283 | IoU: 0.8731 | F1: 0.9283 | HD: 16.60
  Vid Val  | Loss: 2.5460 | Dice: 0.8722 | IoU: 0.8143 | F1: 0.8722 | HD: 19.80
  -> New best model saved (Dice: 0.9283)
Epoch 086/100 (45.0s) | LR: 4.76e-06 | Train Loss: 0.3342 (L1: 0.1202, L2: 0.0026, L3: 0.0018, L4: 0.2348)
  Img Val  | Loss: 1.0439 | Dice: 0.9276 | IoU: 0.8726 | F1: 0.9276 | HD: 16.64
  Vid Val  | Loss: 2.5338 | Dice: 0.8727 | IoU: 0.8151 | F1: 0.8727 | HD: 19.35
Epoch 087/100 (44.9s) | LR: 4.11e-06 | Train Loss: 0.3327 (L1: 0.1185, L2: 0.0016, L3: 0.0014, L4: 0.2363)
  Img Val  | Loss: 1.0472 | Dice: 0.9277 | IoU: 0.8724 | F1: 0.9277 | HD: 16.75
  Vid Val  | Loss: 2.5170 | Dice: 0.8733 | IoU: 0.8159 | F1: 0.8733 | HD: 19.57
Epoch 088/100 (45.1s) | LR: 3.51e-06 | Train Loss: 0.3216 (L1: 0.1097, L2: 0.0022, L3: 0.0020, L4: 0.2349)
  Img Val  | Loss: 1.0487 | Dice: 0.9260 | IoU: 0.8711 | F1: 0.9260 | HD: 16.79
  Vid Val  | Loss: 2.5166 | Dice: 0.8744 | IoU: 0.8172 | F1: 0.8744 | HD: 18.53
Epoch 089/100 (44.8s) | LR: 2.96e-06 | Train Loss: 0.3037 (L1: 0.1094, L2: 0.0030, L3: 0.0024, L4: 0.2122)
  Img Val  | Loss: 1.0508 | Dice: 0.9269 | IoU: 0.8714 | F1: 0.9269 | HD: 16.89
  Vid Val  | Loss: 2.5078 | Dice: 0.8766 | IoU: 0.8189 | F1: 0.8766 | HD: 19.54
Epoch 090/100 (45.1s) | LR: 2.45e-06 | Train Loss: 0.3104 (L1: 0.1081, L2: 0.0014, L3: 0.0013, L4: 0.2243)
  Img Val  | Loss: 1.0510 | Dice: 0.9265 | IoU: 0.8711 | F1: 0.9265 | HD: 16.91
  Vid Val  | Loss: 2.5140 | Dice: 0.8754 | IoU: 0.8179 | F1: 0.8754 | HD: 19.63
Epoch 091/100 (44.9s) | LR: 1.99e-06 | Train Loss: 0.3242 (L1: 0.1152, L2: 0.0033, L3: 0.0023, L4: 0.2289)
  Img Val  | Loss: 1.0484 | Dice: 0.9263 | IoU: 0.8710 | F1: 0.9263 | HD: 16.87
  Vid Val  | Loss: 2.5199 | Dice: 0.8749 | IoU: 0.8173 | F1: 0.8749 | HD: 18.79
Epoch 092/100 (44.8s) | LR: 1.57e-06 | Train Loss: 0.3094 (L1: 0.1076, L2: 0.0017, L3: 0.0010, L4: 0.2236)
  Img Val  | Loss: 1.0501 | Dice: 0.9255 | IoU: 0.8701 | F1: 0.9255 | HD: 16.96
  Vid Val  | Loss: 2.5192 | Dice: 0.8743 | IoU: 0.8168 | F1: 0.8743 | HD: 19.20
Epoch 093/100 (45.0s) | LR: 1.20e-06 | Train Loss: 0.3080 (L1: 0.1069, L2: 0.0020, L3: 0.0016, L4: 0.2224)
  Img Val  | Loss: 1.0508 | Dice: 0.9263 | IoU: 0.8716 | F1: 0.9263 | HD: 16.83
  Vid Val  | Loss: 2.5190 | Dice: 0.8745 | IoU: 0.8166 | F1: 0.8745 | HD: 18.75
Epoch 094/100 (44.9s) | LR: 8.86e-07 | Train Loss: 0.3112 (L1: 0.1052, L2: 0.0013, L3: 0.0012, L4: 0.2297)
  Img Val  | Loss: 1.0477 | Dice: 0.9264 | IoU: 0.8716 | F1: 0.9264 | HD: 16.73
  Vid Val  | Loss: 2.5177 | Dice: 0.8743 | IoU: 0.8160 | F1: 0.8743 | HD: 19.62
Epoch 095/100 (45.0s) | LR: 6.16e-07 | Train Loss: 0.3254 (L1: 0.1109, L2: 0.0031, L3: 0.0022, L4: 0.2371)
  Img Val  | Loss: 1.0523 | Dice: 0.9265 | IoU: 0.8713 | F1: 0.9265 | HD: 16.72
  Vid Val  | Loss: 2.5158 | Dice: 0.8743 | IoU: 0.8162 | F1: 0.8743 | HD: 19.16
Epoch 096/100 (45.4s) | LR: 3.94e-07 | Train Loss: 0.3431 (L1: 0.1111, L2: 0.0028, L3: 0.0030, L4: 0.2585)
  Img Val  | Loss: 1.0517 | Dice: 0.9267 | IoU: 0.8716 | F1: 0.9267 | HD: 16.70
  Vid Val  | Loss: 2.5159 | Dice: 0.8742 | IoU: 0.8159 | F1: 0.8742 | HD: 19.00
Epoch 097/100 (45.4s) | LR: 2.22e-07 | Train Loss: 0.3088 (L1: 0.1103, L2: 0.0015, L3: 0.0014, L4: 0.2188)
  Img Val  | Loss: 1.0502 | Dice: 0.9269 | IoU: 0.8718 | F1: 0.9269 | HD: 16.69
  Vid Val  | Loss: 2.5146 | Dice: 0.8747 | IoU: 0.8166 | F1: 0.8747 | HD: 18.96
Epoch 098/100 (44.9s) | LR: 9.87e-08 | Train Loss: 0.3144 (L1: 0.1106, L2: 0.0016, L3: 0.0018, L4: 0.2250)
  Img Val  | Loss: 1.0490 | Dice: 0.9269 | IoU: 0.8720 | F1: 0.9269 | HD: 16.79
  Vid Val  | Loss: 2.5149 | Dice: 0.8749 | IoU: 0.8168 | F1: 0.8749 | HD: 18.95
Epoch 099/100 (45.2s) | LR: 2.47e-08 | Train Loss: 0.3098 (L1: 0.1105, L2: 0.0026, L3: 0.0038, L4: 0.2176)
  Img Val  | Loss: 1.0492 | Dice: 0.9270 | IoU: 0.8719 | F1: 0.9270 | HD: 16.68
  Vid Val  | Loss: 2.5151 | Dice: 0.8750 | IoU: 0.8169 | F1: 0.8750 | HD: 18.96
Epoch 100/100 (45.5s) | LR: 0.00e+00 | Train Loss: 0.3046 (L1: 0.1099, L2: 0.0022, L3: 0.0018, L4: 0.2134)
  Img Val  | Loss: 1.0525 | Dice: 0.9272 | IoU: 0.8721 | F1: 0.9272 | HD: 16.69
  Vid Val  | Loss: 2.5154 | Dice: 0.8749 | IoU: 0.8168 | F1: 0.8749 | HD: 19.21

Training complete. Best validation Dice: 0.9283```
