# Usage Guide

```{toctree}
:caption: Getting started
:hidden:
user_guide/motivation
user_guide/installation
user_guide/limitations
```

```{toctree}
:caption: Features
:hidden:
user_guide/image_shapes
user_guide/comparison
user_guide/additional_returns
user_guide/logger
```


## Simple computation of metric scores

```python
from mondAI.metrics import MAE

mae = MAE()

# image values need to be in [0,1] value range
image = torch.rand((256,256))
score = mae(image, image)  # score = 0.0

```
