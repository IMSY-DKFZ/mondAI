from collections.abc import Callable

import torch


def get_original_numpy_haarpsi(preprocess_with_subsampling: bool) -> Callable[..., torch.Tensor]:
    from mondAI.metrics.third_party.others_code.haarpsi_original import haar_psi as org_haarpsi

    def original_haarpsi(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # The original implementation by Rafael Reisenhofer and David Neumann expects images with pixel values
        # in [0, 255] and uses double precision floating point format.

        score, _, _ = org_haarpsi(  # type: ignore [no-untyped-call]
            reference.numpy(force=True),
            image.numpy(force=True),
            preprocess_with_subsampling=preprocess_with_subsampling,
        )
        return torch.tensor(score)

    return original_haarpsi


def get_ideal_iqa_haarpsi(C: float, α: float, preprocess_with_subsampling: bool) -> Callable[..., torch.Tensor]:
    from mondAI.metrics.third_party.others_code.haarpsi_ideal_iqa import haarpsi as haarpsi_ideal

    def ideal_iqa_haarpsi(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # ideal_iqa's implementation expects inputs with shape (N, C, H, W) and
        # pixel values in [0, 1], while the original implementation expects images with pixel values in [0, 255]

        score, _, _ = haarpsi_ideal(  # type: ignore [no-untyped-call]
            reference / 255.0,
            image / 255.0,
            C=C,
            α=α,
            preprocess_with_subsampling=preprocess_with_subsampling,
        )
        return score

    return ideal_iqa_haarpsi


def get_pytorch_iwssim(
    iw_flag: bool, Nsc: int, blSzX: int, blSzY: int, parent: bool, sigma_nsq: float
) -> Callable[..., torch.Tensor]:
    from mondAI.metrics.third_party.others_code.iwssim_pytorch import IW_SSIM as IWSSIMPyTorch

    def iwssim_pytorch(image: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # This is a PyTorch implementation of IW-SSIM that closely follows the
        # original MATLAB code by Zhou Wang and Qiang Li.
        # It is not optimized for speed, uses numpy internally but serves as a useful reference for correctness.
        metric = IWSSIMPyTorch(
            iw_flag=iw_flag,
            Nsc=Nsc,
            blSzX=blSzX,
            blSzY=blSzY,
            parent=parent,
            sigma_nsq=sigma_nsq,
            use_cuda=image.device.type == "cuda",
            use_double=True,
        )  # type: ignore[no-untyped-call]

        return metric.test(reference.numpy(force=True), image.numpy(force=True))  # type: ignore[no-untyped-call]

    return iwssim_pytorch
