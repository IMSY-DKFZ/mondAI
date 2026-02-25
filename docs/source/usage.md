# Usage

## Simple computation of metric scores

```python
from mondAI import MAE

mae = MAE()
image = torch.random.rand((256,256))
score = mae(image, image)  # score = 0.0

```

## Comparison of metric implementations

Sometimes you would like to compare the implementations of the same metric from different libraries. This can be done easily with mondAI:

```python
import torch
from skimage.data import shepp_logan_phantom

from mondAI.metrics.full_reference.mae import MAE

img1 = torch.from_numpy(shepp_logan_phantom()).unsqueeze(0).unsqueeze(0).float() / 255.0
img2 = torch.flip(img1.clone(), dims=[2])  # Using the same image for simplicity

mae = MAE()
scores = mae(img1, img2, dims = ("B", "C", "H", "W"), compare_implementations=True)

# print as table
print(f"{'Implementation':<20} {'MAE Score':<10}")
print("-" * 30)
for impl, score in scores.items():
    print(f"{impl:<20} {score.item():.6f}")

```

Output:
```
Implementation       MAE Score
------------------------------
sklearn              0.000153
tensorflow           0.000153
monai                0.000153
mondAI               0.000153
```
