from enum import IntEnum

import jax
import jax.numpy as jnp
import numpy as np

from recovar.core import fourier_transform_utils
from recovar.core import relion_interp


class CTFParamIndex(IntEnum):
    DFU = 0
    DFV = 1
    DFANG = 2
    VOLT = 3
    CS = 4
    W = 5
    PHASE_SHIFT = 6
    BFACTOR = 7
    CONTRAST = 8


@jax.jit
def evaluate_ctf(freqs, ctf_params):
    dfu = ctf_params[:, CTFParamIndex.DFU, None]
    dfv = ctf_params[:, CTFParamIndex.DFV, None]
    dfang = ctf_params[:, CTFParamIndex.DFANG, None] * (jnp.pi / 180)
    volt = ctf_params[:, CTFParamIndex.VOLT, None] * 1000
    cs = ctf_params[:, CTFParamIndex.CS, None] * 1e7
    w = ctf_params[:, CTFParamIndex.W, None]
    phase_shift = ctf_params[:, CTFParamIndex.PHASE_SHIFT, None] * (jnp.pi / 180)
    bfactor = ctf_params[:, CTFParamIndex.BFACTOR, None]
    contrast = ctf_params[:, CTFParamIndex.CONTRAST, None]

    lam = 12.2642598 / jnp.sqrt(volt * (1.0 + volt * 9.78475598e-7))
    x = freqs[:, 0]
    y = freqs[:, 1]
    ang = jnp.arctan2(y, x)
    s2 = x * x + y * y
    df = 0.5 * (dfu + dfv + (dfu - dfv) * jnp.cos(2 * (ang - dfang)))
    gamma = 2 * jnp.pi * (-0.5 * df * lam * s2 + 0.25 * cs * lam**3 * s2**2) - phase_shift
    ctf = (1 - w**2) ** 0.5 * jnp.sin(gamma) - w * jnp.cos(gamma)
    ctf = ctf * jnp.exp(-bfactor / 4 * s2)
    return ctf * contrast


class CTFEvaluator:
    def __call__(self, ctf_params, image_shape, voxel_size, *, half_image=False):
        if half_image:
            freqs = fourier_transform_utils.get_k_coordinate_of_each_pixel_half(image_shape, voxel_size, scaled=True)
        else:
            freqs = fourier_transform_utils.get_k_coordinate_of_each_pixel(image_shape, voxel_size, scaled=True)
        return evaluate_ctf(freqs, ctf_params)


def _order(disc_type):
    if disc_type == "linear_interp":
        return 1
    if disc_type == "nearest":
        return 0
    raise ValueError("Only linear_interp and nearest are included in this tutorial subset.")


def slice_volume(
    volume,
    rotation_matrices,
    image_shape,
    volume_shape,
    disc_type,
    half_volume=False,
    half_image=False,
    max_r=None,
):
    return relion_interp.project(
        volume,
        rotation_matrices,
        image_shape,
        volume_shape,
        order=_order(disc_type),
        half_volume=half_volume,
        half_image=half_image,
        max_r=max_r,
    )


def adjoint_slice_volume(
    slices,
    rotation_matrices,
    image_shape,
    volume_shape,
    disc_type,
    volume=None,
    half_image=False,
    half_volume=False,
    max_r=None,
):
    result = relion_interp.backproject(
        slices,
        rotation_matrices,
        image_shape,
        volume_shape,
        order=_order(disc_type),
        half_volume=half_volume,
        half_image=half_image,
        max_r=max_r,
    )
    return result if volume is None else result + volume


@jax.jit
def _translate_single_image(image, translation, lattice):
    phase_shift = jnp.exp(-2j * jnp.pi * (lattice @ translation))
    return image * phase_shift


_batch_translate = jax.vmap(_translate_single_image, in_axes=(0, 0, None))


def translate_images(image, translation, image_shape, *, half_image=False):
    if half_image:
        lattice = fourier_transform_utils.get_k_coordinate_of_each_pixel_half(image_shape, voxel_size=1, scaled=True)
    else:
        lattice = fourier_transform_utils.get_k_coordinate_of_each_pixel(image_shape, voxel_size=1, scaled=True)
    return _batch_translate(image, translation, lattice[:, :2])


__all__ = [
    "CTFEvaluator",
    "CTFParamIndex",
    "adjoint_slice_volume",
    "evaluate_ctf",
    "fourier_transform_utils",
    "slice_volume",
    "translate_images",
]

