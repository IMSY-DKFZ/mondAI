# Third-Party Notices

## CWSSIM

Based on:
M. P. Sampat, Z. Wang, S. Gupta, A. C. Bovik, and M. K. Markey,
"Complex Wavelet Structural Similarity: A New Image Similarity Index",
IEEE Transactions on Image Processing, vol. 18, no. 11, pp. 2385-2401, 2009,
doi: 10.1109/TIP.2009.2025923.

Implementation adapted from the original MATLAB code by Zhou Wang, Mehul Sampat and Alan Bovik: https://de.mathworks.com/matlabcentral/fileexchange/43017-complex-wavelet-structural-similarity-index-cw-ssim/files/cwssim_index.m (license: BSD 2-Clause)

## DSS

Based on:
Balanov, A., Schwartz, A., Moshe, Y., & Peleg, N. (2015, September).
Image quality assessment based on DCT subband similarity.
In 2015 IEEE international conference on image processing (ICIP) (pp. 2105-2109). IEEE.

Implementation adapted from original MATLAB implementation by Yair Moshe:
https://de.mathworks.com/matlabcentral/fileexchange/53708-dct-subband-similarity-index-for-measuring-image-quality
(licence: BSD 3-Clause License)

## FSIM

Based on:
L. Zhang, L. Zhang, X. Mou and D. Zhang, "FSIM: A Feature Similarity Index for Image Quality Assessment,"
IEEE Transactions on Image Processing, vol. 20, no. 8, pp. 2378-2386, Aug. 2011, doi: 10.1109/TIP.2011.2109730.
https://ieeexplore.ieee.org/document/5705575

Implementation adapted from PIQ: https://github.com/photosynthesis-team/piq/blob/master/piq/fsim.py, commit: 213a46687ad99098f274784e61d92c5144a94a68 (License: Apache 2.0),
which itself is based on the original MATLAB implementation by Lin Zhang, Lei Zhang, Xuanqin Mou and David Zhang: https://www4.comp.polyu.edu.hk/~cslzhang/IQA/FSIM/Files/FeatureSIM.m


## HaarPSI

Based on:
R. Reisenhofer, S. Bosse, G. Kutyniok and T. Wiegand.
A Haar Wavelet-Based Perceptual Similarity Index for Image Quality Assessment.
Signal Processing: Image Communication, vol. 61, 33-43, 2018.
doi:10.1016/j.image.2017.11.001

Implementation adapted from Anna Breger and Clemens Karner:
https://github.com/ideal-iqa/haarpsi-pytorch/blob/main/haarpsi.py commit:
a64b753b1b95a826996fcc035ce3f4dc4c630a5f (License: MIT),
with the difference that this implementation expects images with pixel values in the range [0, 255] as the original publication, while their implementation expects images with pixel values in the range [0, 1]. Furthermore, this implementation uses a double precision floating point format for all computations, while their implementation uses single precision.
The original MATLAB and NumPy implementations by Rafael Reisenhofer: https://github.com/rgcda/haarpsi/blob/master/HaarPSI.m commit: 2c2793108477deb81971658a7666d5f85ba2587b (License: MIT)
and David Neumann: https://github.com/rgcda/haarpsi/blob/master/haarPsi.py commit: 2c2793108477deb81971658a7666d5f85ba2587b (License: MIT) also expect images with pixel values in the range [0, 255] and use double precision floating point format.

## HaarPSI_MED

Based on:
Karner, C., Gröhl, J., Selby, I., Babar, J., Beckford, J., Else, T. R., Sadler, T. J., Shahipasand, S., Thavakumar, A., Roberts, M., Rudd, J. H. F., Schönlieb, C.-B., Weir-McCall, J. R., & Breger, A. (2025).
Parameter choices in HaarPSI for IQA with medical images. 2025 IEEE International Symposium on Biomedical Imaging (ISBI).

Implementations see HaarPSI

## IW-SSIM

Based on:
Z. Wang, and L. Qiang,
"Information content weighting for perceptual image quality assessment",
IEEE Transactions on Image Processing, vol. 20, no. 5, pp. 1185-1198, 2011,
doi: 10.1109/TIP.2010.2092435

Implementation adapted from the original MATLAB implementation by Zhou Wang:
https://ece.uwaterloo.ca/~z70wang/research/iwssim/ and
PIQ: https://github.com/photosynthesis-team/piq/blob/master/piq/iw_ssim.py commit: 213a46687ad99098f274784e61d92c5144a94a68 (license: Apache 2.0)

## MS-SSIM

Based on:
Z. Wang, E. P. Simoncelli, and A. C. Bovik,
"Multi-scale structural similarity for image quality assessment,"
Proceedings of the 37th Asilomar Conference on Signals, Systems and Computers,
Nov. 2003, pp. 1398-1402, doi: 10.1109/ACSSC.2003.1292216.

Implementation adapted from the original MATLAB implementation by Zhou Wang:
https://ece.uwaterloo.ca/~z70wang/research/iwssim/msssim.zip

## PSNR

Reference implementation used for comparison and API alignment:
https://github.com/scikit-image/scikit-image/blob/main/src/skimage/metrics/simple_metrics.py
commit: d0b36ad1651ee2d7f2a46a84b06ba593a0885c6e (License: BSD-3-Clause)

## SSIM

Based on:
Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli,
"Image quality assessment: From error visibility to structural similarity,"
IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, Apr. 2004,
doi: 10.1109/TIP.2003.819861.

Implementation adapted from the original MATLAB implementation by Zhou Wang_ https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m

## VIFP

Based on:
H.R. Sheikh.and A.C. Bovik, "Image information and visual quality,"
IEEE Transactions on Image Processing , vol.15, no.2,pp. 430- 444, Feb. 2006.

Implementation adapted from PIQ: https://github.com/photosynthesis-team/piq/blob/master/piq/vif.py commit: 213a46687ad99098f274784e61d92c5144a94a68 (License: Apache License 2.0)
and MATLAB code provided by the original authors of VIFP: https://live.ece.utexas.edu/research/Quality/VIF.htm, (License: BSD 3-Clause License)

## PaQ2PiQ

Based on:
Original publication:
Ying, Zhenqiang, et al. "From patches to pictures (PaQ-2-PiQ): Mapping the perceptual space of picture quality."
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.

Implementation adapted from the original author's implementation:
https://github.com/baidut/paq2piq/ commit: 48c91e844e9f7a768f6ddcfc744dddfdd1160fea (License: MIT)
