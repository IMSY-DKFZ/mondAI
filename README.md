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
from mondAI.metrics import PSNR, SSIM

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

## Tests

Run tests with `pytest` and compute the coverage with

> pytest --cov=mondAI --cov-report=term-missing

In addition to tests checking the correctness of the code, there are also regression tests defined in `test_regression.py` which ensure that the computed metrics scores don't change, e.g. when updating dependencies. These scores are computed on the first run of the test (which then fails and creates a reference file under `test_regression`) and then used for consecutive runs for testing. Use `pytest --force-regen` or `pytest --regen-all` to update a single or all test scores. The test scores under `test_regression` need to be committed.

## License
This project is licensed under Apache 2.0
