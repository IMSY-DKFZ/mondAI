# mondAI

 > Metrics on n-dimension artificial images

This package provides image quality metric implementations which are:
* correct (in the sense of the original definition)
* usable
* reproducible and reportable
* tested
* documented

## Installation

To install the package in editable development mode, clone the repository, activate a virtual environment and run:
> pip install -e .[dev]

In the long run mondAI will also be installable via PyPI:
> pip install mondAI

## Usage

You can simply create metric objects and call them on your images/references:
```python
from mondAI import PSNR, SSIM

psnr = PSNR()
ssim = SSIM()

image = torch.rand((256, 128))
reference = torch.rand((256, 128))

psnr_score = psnr(image, reference)
ssim_score = ssim(image, reference)
```

You can also create a list of metrics and compute all of them by simply calling the list once:

```python
from mondAI import MetricList

validation_metrics = MetricList(name='validation', metrics=[PSNR(), SSIM()])

validation_scores = validation_metrics(image, reference)
```
For further information on usage, please read the [documentation](https://imsy.pages.dkfz.de/ispai/mondai/).

## Contributing

Any contributions to mondAI are appreciated. In particular, adding and reviewing implementations of metrics would be helpful. To reduce the burden of implementing a new metric, you can simply copy the `metric_template.py` file and follow the #TODO comments.

Please read the [contribution guidelines](CONTRIBUTING.md) once before you start contributing.

## License
This project is licensed under Apache 2.0
