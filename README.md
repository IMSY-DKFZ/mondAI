# mondAI

 > Metrics on n-dimension artificial images

This package provides image quality metric implementations which are:
* correct (in the sense of the original definition)
* usable
* reproducible and reportable
* tested
* documented

## Installation

To install the package in editable development mode with development dependencies, clone the repository, activate a virtual environment and run:
> pip install -e .[dev]

To be able to also call metric implementations from other packages and compare the scores, install additional dependencies:
> pip install -e .[dev,comparison]

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

## Implemented Metrics

|         | Expected dimensions | Input ranges | Output ranges | Symmetric | Scholar Citations | Original publication                                                                                                                                                     | Original implementation                                                                                            | Original implementation language |
|---------|---------------------|--------------|---------------|-----------|-------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|----------------------------------|
| CW-SSIM | H,W                 | 0-255        | 0-1           | yes       |               868 | [Link](https://ieeexplore.ieee.org/document/5109651)                                                                                                                            | [Link](https://de.mathworks.com/matlabcentral/fileexchange/53708-dct-subband-similarity-index-for-measuring-image-quality) | MATLAB                           |
| DISTS   | (B), 3, H, W        | 0-1          | 0-1           | yes       |              1651 | [Link](https://arxiv.org/pdf/2004.07728)                                                                                                                                         | [Link](https://github.com/dingkeyan93/DISTS)                                                                               | Python                           |
| DSS     | H,W                 | 0-255        | 0-1           | yes       |                71 | [Link](https://ieeexplore.ieee.org/document/7351172)                                                                                                                             | [Link](https://de.mathworks.com/matlabcentral/fileexchange/53708-dct-subband-similarity-index-for-measuring-image-quality) | MATLAB                           |
| FSIM    | (C=1 or 3), H,W     | 0-255        | 0-1           | yes       |              6359 | [Link](https://ieeexplore.ieee.org/document/5705575)                                                                                                                             | [Link](https://www4.comp.polyu.edu.hk/~cslzhang/IQA/FSIM/Files/FeatureSIM.m)                                               | MATLAB                           |
| GMSD    | H,W                 | 0-255        | 0-inf         | yes       |              2026 | [Link](https://ieeexplore.ieee.org/document/6678238)                                                                                                                             | [Link](https://www4.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm)                                                         | MATLAB                           |
| HaarPSI | (C=1 or 3), H,W     | 0-255        | 0-1           | yes       |               434 | [Link](https://www.sciencedirect.com/science/article/pii/S0923596517302187?casa_token=OVoC7jszE_0AAAAA:3yQpx49E9Zx2g0g27bG_khQ1VzIBmFldruHkicn97VwJOE_2RWuZQ8vgcn6wg2lp9_vurVs1) | [Link](https://github.com/rgcda/haarpsi)                                                                                   | MATLAB                           |
| IW-SSIM | H,W                 | 0-255        | 0-1           | no        |              1650 | [Link](https://ieeexplore.ieee.org/document/5635337)                                                                                                                             | [Link](https://ece.uwaterloo.ca/~z70wang/research/iwssim/)                                                                 | MATLAB                           |
| LPIPS   | (B), 3, H, W        | -1 - 1 <br>(in this package 0-1)      | 0-1           | yes       |             21764 | [Link](https://arxiv.org/pdf/1801.03924)                                                                                                                                         | [Link](https://github.com/richzhang/PerceptualSimilarity)                                                                  | Python                           |
| MAE     | H,W                 | any          | 0-inf         | yes       | -                 | -                                                                                                                                                                        | -                                                                                                                  | -                                |
| MDSI    | 3, H,W              | 0-255        | 0-inf         | yes       |               237 | [Link](https://arxiv.org/pdf/1608.07433)                                                                                                                                         | [Link](https://de.mathworks.com/matlabcentral/fileexchange/59809-mdsi-ref-dist-combmethod/files/MDSI.m)                    | MATLAB                           |
| MS-SSIM | H,W                 | 0-255        | 0-1           | yes       |              9669 | [Link](https://ieeexplore.ieee.org/document/1292216)                                                                                                                             | [Link](https://ece.uwaterloo.ca/~z70wang/research/iwssim/msssim.zip)                                                      | MATLAB                           |
| MSE     | H,W                 | any          | 0-inf         | yes       | -                 | -                                                                                                                                                                        | -                                                                                                                  | -                                |
| NIQE    | 3, H,W              |   0-255           | 0-inf         | -         |              8063 | [Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6353522)                                                                                                         | [Link](http://live.ece.utexas.edu/research/quality/niqe_release.zip)                                                       | MATLAB                           |
| NMSE    | H,W                 | any          | 0-inf         | yes       | -                 | -                                                                                                                                                                        | -                                                                                                                  | -                                |
| PaQ2PiQ | (B),3,H,W           | 0-1          | 0-1           | -         |               555 | [Link](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ying_From_Patches_to_Pictures_PaQ-2-PiQ_Mapping_the_Perceptual_Space_of_CVPR_2020_paper.pdf)                       | [Link](https://github.com/baidut/paq2piq/)                                                                                 | Python                           |
| PSNR    | H,W                 | any          | 0-inf         | yes       |      -             | -                                                                                                                                                                        | -                                                                                                                  | -                                |
| RMSE    | H,W                 | any          | 0-inf         | yes       | -                 | -                                                                                                                                                                        | -                                                                                                                  | -                                |
| SSIM    | H,W                 | 0-255        | 0-1           | yes       |             71473 | [Link](https://ieeexplore.ieee.org/document/1284395)                                                                                                                             | [Link](https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m)                                                       | MATLAB                           |
| VIF     | H,W                 | 0-255        | 0-1           | no        |              5379 | [Link](https://ieeexplore.ieee.org/document/1576816)                                                                                                                             | [Link](https://live.ece.utexas.edu/research/Quality/VIF.htm)                                                               | MATLAB                           |
| VSI     | 3, H,W              | 0-255        | 0-1           | yes       |              1213 | [Link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6873260)                                                                                                         | not accessible anymore                                                                                             | MATLAB                           |

## Contributing

Any contributions to mondAI are appreciated. In particular, adding and reviewing implementations of metrics would be helpful. To reduce the burden of implementing a new metric, you can simply copy the `.metric_template.py` file into `full_reference` or `no_reference` respectively and follow the #TODO comments.

Please read the [contribution guidelines](CONTRIBUTING.md) once before you start contributing.

## Tests

Run tests with `pytest` and compute the coverage with

> pytest -m "not slow" --cov=mondAI --cov-report=term-missing

In addition to tests checking the correctness of the code, there are also regression tests defined in `test_regression.py` which ensure that the computed metrics scores don't change, e.g. when updating dependencies. These scores are computed on the first run of the test (which then fails and creates a reference file under `test_regression`) and then used for consecutive runs for testing. Use `pytest --force-regen` or `pytest --regen-all` to update a single or all test scores. The test scores under `test_regression` need to be committed.

## License
This project is licensed under Apache 2.0
