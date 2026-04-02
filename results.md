# Results

## Dataset: KVasir-SEG + LDPolyp Video Dataset

| Split | Samples |
|-------|---------|
| Train | 800 seg + 800 video frames |
| Val   | 100 |
| Test  | 100 |

**Trainable params:** 171,040,809 / **Total:** 366,943,530

## Training Log

```
Epoch 001/10 (57.0s) | LR: 9.76e-05 | Train Loss: 0.9823 (L1: 0.9057, L2: 0.0436, L3: 0.0242, L4: 0.0209) | Val Loss: 1.0071 | Dice: 0.6831 | IoU: 0.5481 | F1: 0.6831 | HD: 55.80
  -> New best model saved (Dice: 0.6831)
Epoch 002/10 (53.8s) | LR: 9.05e-05 | Train Loss: 0.7447 (L1: 0.7261, L2: 0.0114, L3: 0.0078, L4: 0.0033) | Val Loss: 0.7633 | Dice: 0.7720 | IoU: 0.6554 | F1: 0.7720 | HD: 42.48
  -> New best model saved (Dice: 0.7720)
Epoch 003/10 (52.2s) | LR: 7.94e-05 | Train Loss: 0.6184 (L1: 0.6056, L2: 0.0082, L3: 0.0049, L4: 0.0022) | Val Loss: 0.6488 | Dice: 0.8018 | IoU: 0.6898 | F1: 0.8018 | HD: 42.76
  -> New best model saved (Dice: 0.8018)
Epoch 004/10 (45.5s) | LR: 6.55e-05 | Train Loss: 0.5274 (L1: 0.5172, L2: 0.0067, L3: 0.0036, L4: 0.0017) | Val Loss: 0.5113 | Dice: 0.8338 | IoU: 0.7301 | F1: 0.8338 | HD: 31.78
  -> New best model saved (Dice: 0.8338)
Epoch 005/10 (45.3s) | LR: 5.00e-05 | Train Loss: 0.4737 (L1: 0.4639, L2: 0.0068, L3: 0.0033, L4: 0.0013) | Val Loss: 0.3643 | Dice: 0.8819 | IoU: 0.7961 | F1: 0.8819 | HD: 24.10
  -> New best model saved (Dice: 0.8819)
Epoch 006/10 (46.2s) | LR: 3.45e-05 | Train Loss: 0.3814 (L1: 0.3772, L2: 0.0026, L3: 0.0010, L4: 0.0011) | Val Loss: 0.3815 | Dice: 0.8871 | IoU: 0.8084 | F1: 0.8871 | HD: 23.81
  -> New best model saved (Dice: 0.8871)
Epoch 007/10 (44.5s) | LR: 2.06e-05 | Train Loss: 0.3358 (L1: 0.3334, L2: 0.0012, L3: 0.0004, L4: 0.0010) | Val Loss: 0.3329 | Dice: 0.8871 | IoU: 0.8078 | F1: 0.8871 | HD: 21.87
Epoch 008/10 (43.7s) | LR: 9.55e-06 | Train Loss: 0.3061 (L1: 0.3037, L2: 0.0013, L3: 0.0003, L4: 0.0009) | Val Loss: 0.2981 | Dice: 0.8965 | IoU: 0.8222 | F1: 0.8965 | HD: 18.96
  -> New best model saved (Dice: 0.8965)
Epoch 009/10 (44.1s) | LR: 2.45e-06 | Train Loss: 0.2888 (L1: 0.2867, L2: 0.0012, L3: 0.0002, L4: 0.0008) | Val Loss: 0.2909 | Dice: 0.9101 | IoU: 0.8425 | F1: 0.9101 | HD: 19.14
  -> New best model saved (Dice: 0.9101)
Epoch 010/10 (45.0s) | LR: 0.00e+00 | Train Loss: 0.2723 (L1: 0.2703, L2: 0.0011, L3: 0.0002, L4: 0.0008) | Val Loss: 0.2738 | Dice: 0.9114 | IoU: 0.8451 | F1: 0.9114 | HD: 17.99
  -> New best model saved (Dice: 0.9114)
```

**Best validation Dice:** 0.9114

## Test Results (100 images)

| Metric | Value |
|--------|-------|
| Dice | 0.8777 |
| IoU | 0.8015 |
| F1 | 0.8777 |
| Hausdorff Distance | 23.79 |
| Loss | 0.3397 |
