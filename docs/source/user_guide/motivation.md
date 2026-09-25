# Motivation

This metric library is motivated by good scientific practice. Researchers publish their algorithms every day and try to perform some benchmarking by computing metric scores, however, these are often not comparable, even though they might be called the same. In this example code you can find four computations of SSIM from torchmetrics, piq, seqar and monai packages with their corresponding default configurations. Unfortunately, when computing metric scores on an example image they yield four different scores. This might be due to different metric configurations, but could also be due to different implementations. Hence, to avoid such pitfalls, having a standardized metric library which provides validated implementations of all metrics would be useful.

# Example Code

```python
import torch
from monai.metrics.regression import SSIMMetric as SSIM_monai
from piq import ssim as SSIM_piq
from sewar.full_ref import ssim as SSIM_sewar
from skimage.data import shepp_logan_phantom
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM_torchmetrics


def compute_ssim_torchmetrics(img1, img2):
    ssim_metric = SSIM_torchmetrics(data_range=1.0)
    ssim_value = ssim_metric(img1, img2)
    return ssim_value.item()


def compute_ssim_piq(img1, img2):
    ssim_value = SSIM_piq(img1, img2)
    return ssim_value.item()


def compute_ssim_sewar(img1, img2):
    # Convert tensors to numpy arrays and remove batch and channel dimensions
    img1_np = img1.squeeze().cpu().numpy()
    img2_np = img2.squeeze().cpu().numpy()

    ssim_value = SSIM_sewar(img1_np, img2_np, MAX=1.0)
    return ssim_value[0]


def compute_ssim_monai(img1, img2):
    ssim_metric = SSIM_monai(spatial_dims=2)
    ssim_value = ssim_metric(img1, img2)
    return ssim_value.item()


# Example usage
if __name__ == "__main__":
    img1 = torch.from_numpy(shepp_logan_phantom()).unsqueeze(0).unsqueeze(0).float() / 255.0
    img2 = torch.flip(img1.clone(), dims=[2])  # Using the same image for simplicity

    ssim_torchmetrics = compute_ssim_torchmetrics(img1, img2)
    ssim_piq = compute_ssim_piq(img1, img2)
    ssim_sewar = compute_ssim_sewar(img1, img2)
    ssim_monai = compute_ssim_monai(img1, img2)

    print(f"SSIM (torchmetrics): \t {ssim_torchmetrics:.6f}")
    print(f"SSIM (piq): \t\t {ssim_piq:.6f}")
    print(f"SSIM (sewar): \t\t {ssim_sewar:.6f}")
    print(f"SSIM (monai): \t\t {ssim_monai:.6f}")

```

## Output

| Library | Score |
| ---- | ---- |
| SSIM (torchmetrics)| 0.997998|
| SSIM (piq)|          0.998295|
| SSIM (sewar)|        0.998396|
| SSIM (monai)|        0.997894|
