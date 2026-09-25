# Image Shapes

Within mondAI image can be any of the following:
* two dimensional single-channel/grayscale images: (H, W)
* RGB images: (3, H, W) or (H, W, 3)
* Multispectral images: (C, H, W) or (H, W, C)
* three dimensional image volumes: (D, H, W) or (H, W, D)
* batches of images or volumes: (B, H, W), (B, D, H, W)
* or any combination but at maximum a five dimensional input (B, C, D, H, W)

By default, mondAI expects only images to only have dimensions (H, W). Thus you can simply compute a metric score like this
```python
score = metric(image, reference)   # image and reference have shape (H, W)
```

If your images have other shapes, you need to specify the dimensions in the metric call so that mondAI knows which dimensions they correspond to, e.g.
```python
score = metric(image, reference, dims=("D", "H", "W"))   # image and reference have shape (D, H, W)
```

Another possibility is to change the default dimension package wide, so that you don't have to specify it on every metric call:
```python
from mondAI.settings import settings
settings.default_dims = ("D", "H", "W")
score = metric(image, reference)   # image and reference have shape (D, H, W)
```

Depending on the `expected_dims` of each metric, scores along the left over dimensions are computed and returned, e.g. a metric expecting 2D image shapes (H, W) which is given inputs of shape (B, C, H, W) will return scores in the shape (B, C).
