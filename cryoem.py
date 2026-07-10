from pathlib import Path
import urllib.request
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import mrcfile
import numpy as np

from recovar import core, utils
from recovar.core import CTFEvaluator
from recovar.core import fourier_transform_utils as ft
from recovar.data_io import metadata_readers
from recovar.simulation import simulate_scattering_potential as scattering
from recovar.simulation import simulator
from recovar.simulation.pdb_utils import parse_pdb

warnings.filterwarnings("ignore", message="Argument `.*` does not satisfy")


def get_pdb(pdb_id="6VXX", path="6vxx.cif"):
    path = Path(path)
    if not path.exists():
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)
    print(f"Using {path.resolve()}")
    return path


def make_scattering_volume(pdb_path, grid_size, box_size_angstrom, out_path):
    voxel_size = box_size_angstrom / grid_size
    atoms = parse_pdb(str(pdb_path))
    atoms.setCoords(atoms.getCoords() - atoms.getCoords().mean(axis=0))

    print(f"Parsed {atoms.numAtoms():,} atoms")
    print("Centered coordinate extent in Angstrom:", np.ptp(atoms.getCoords(), axis=0).round(1))
    print(f"Using grid {grid_size}^3, voxel size {voxel_size:.2f} A")

    volume_ft = scattering.generate_volume_from_atoms(atoms, voxel_size=voxel_size, grid_size=grid_size)
    volume = np.asarray(ft.get_idft3(volume_ft).real, dtype=np.float32)
    save_mrc(out_path, volume, voxel_size)
    print(f"Saved {out_path} with shape {volume.shape}")
    return volume, voxel_size


def save_mrc(path, volume, voxel_size):
    utils.write_mrc(str(path), np.asarray(volume, dtype=np.float32), voxel_size=voxel_size)


def show_slices(volume, title="", percentile=99.5):
    volume = np.asarray(volume)
    n = volume.shape[0]
    lo, hi = np.percentile(volume, [100 - percentile, percentile])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(volume)), float(np.nanmax(volume))
    vmin = lo if lo < 0 else 0
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
    for ax, image, subtitle in zip(
        axes,
        [volume[n // 2], volume[:, n // 2, :], volume[:, :, n // 2]],
        ["z central slice", "y central slice", "x central slice"],
    ):
        ax.imshow(image, cmap="magma", origin="lower", vmin=vmin, vmax=hi)
        ax.set_title(subtitle)
        ax.axis("off")
    fig.suptitle(title)
    plt.show()
    plt.close(fig)


def show_images(images, titles=None, columns=None, cmap="gray", title=""):
    images = list(images)
    columns = columns or len(images)
    rows = int(np.ceil(len(images) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(2.2 * columns, 2.2 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for i, ax in enumerate(axes):
        if i >= len(images):
            ax.axis("off")
            continue
        image = np.asarray(images[i])
        lo, hi = np.percentile(image, [1, 99])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(image)), float(np.nanmax(image))
        ax.imshow(image, cmap=cmap, origin="lower", vmin=lo, vmax=hi)
        if titles is not None:
            ax.set_title(titles[i])
        ax.axis("off")
    fig.suptitle(title)
    plt.show()
    plt.close(fig)


def simple_ctf(n, pixel_size, defocus_um=1.5, voltage_kv=300.0, cs_mm=2.7, amp_contrast=0.1, b_factor=30.0):
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=pixel_size))
    kx, ky = np.meshgrid(freqs, freqs, indexing="xy")
    s2 = kx**2 + ky**2

    voltage = voltage_kv * 1000.0
    wavelength = 12.2639 / np.sqrt(voltage + 0.97845e-6 * voltage**2)
    defocus = defocus_um * 1e4
    cs = cs_mm * 1e7
    gamma = np.pi * (-defocus * wavelength * s2 + 0.5 * cs * wavelength**3 * s2**2)
    ctf = -(np.sqrt(1 - amp_contrast**2) * np.sin(gamma) + amp_contrast * np.cos(gamma))
    return (ctf * np.exp(-b_factor * s2 / 4.0)).astype(np.float32)


def dft2(image):
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(image)))


def rotation_y(degrees):
    theta = np.deg2rad(degrees)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def slice(volume_ft, rotations):
    rotations = np.asarray(rotations, dtype=np.float32)
    if rotations.ndim == 2:
        rotations = rotations[None]
    n = volume_ft.shape[0]
    slices = core.slice_volume(
        jnp.asarray(volume_ft.reshape(-1).astype(np.complex64)),
        jnp.asarray(rotations),
        (n, n),
        (n, n, n),
        "linear_interp",
        max_r=None,
    )
    slices = np.asarray(slices).reshape(rotations.shape[0], n, n)
    return slices[0] if len(slices) == 1 else slices


def backproject(slices_ft, rotations, n=None):
    slices_ft = np.asarray(slices_ft)
    rotations = np.asarray(rotations, dtype=np.float32)
    if n is None:
        n = slices_ft.shape[-1]
    if slices_ft.ndim == 2:
        slices_ft = slices_ft[None]
    if rotations.ndim == 2:
        rotations = rotations[None]
    volume_ft = core.adjoint_slice_volume(
        jnp.asarray(slices_ft.reshape(slices_ft.shape[0], -1).astype(np.complex64)),
        jnp.asarray(rotations),
        (n, n),
        (n, n, n),
        "linear_interp",
        half_image=False,
        max_r=None,
    )
    return np.asarray(volume_ft).reshape(n, n, n)


def ctfs(ctf_params, n, voxel_size):
    ctf_params = np.asarray(ctf_params, dtype=np.float32)
    ctfs = CTFEvaluator()(jnp.asarray(ctf_params), (n, n), voxel_size)
    return np.asarray(ctfs).reshape(ctf_params.shape[0], n, n).astype(np.float32)


def translate(images_ft, translations_pixels, n):
    images_ft = np.asarray(images_ft)
    translations_pixels = np.asarray(translations_pixels, dtype=np.float32)
    if images_ft.ndim == 2:
        images_ft = images_ft[None]
    translated = core.translate_images(
        jnp.asarray(images_ft.reshape(images_ft.shape[0], -1).astype(np.complex64)),
        jnp.asarray(translations_pixels),
        (n, n),
    )
    return np.asarray(translated).reshape(images_ft.shape[0], n, n)


def make_synthetic_dataset(volume_ft, n_images, voxel_size, seed=2):
    rng = np.random.default_rng(seed)
    n = volume_ft.shape[0]
    rotations = simulator.uniform_rotation_sampling(n_images, n, seed=seed).astype(np.float32)
    defocuses = rng.uniform(0.8, 2.5, size=n_images)
    ctfs = np.stack([simple_ctf(n, voxel_size, defocus_um=d) for d in defocuses]).astype(np.float32)
    clean_ft = slice(volume_ft, rotations)

    images = []
    for clean, ctf in zip(clean_ft, ctfs):
        filtered = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(clean * ctf))).real
        images.append(filtered + rng.normal(scale=0.70 * np.std(filtered), size=filtered.shape))
    return np.asarray(images, dtype=np.float32), rotations, ctfs, clean_ft.astype(np.complex64)


def load_empiar10028_particles(
    particles_path="particles_1000_128.mrcs",
    star_path="particles_1000_128.star",
):
    particles_path = Path(particles_path)
    star_path = Path(star_path)
    if not particles_path.exists():
        raise FileNotFoundError(f"Could not find {particles_path}")
    if not star_path.exists():
        raise FileNotFoundError(f"Could not find {star_path}")

    with mrcfile.open(particles_path, permissive=True) as mrc:
        images = np.asarray(mrc.data, dtype=np.float32).copy()

    n = images.shape[-1]
    rotations, translations_fractional = metadata_readers.parse_poses_from_star(str(star_path), n)
    ctf_with_apix = metadata_readers.parse_ctf_from_star(str(star_path), n)
    ctf_with_apix = np.concatenate(
        [
            ctf_with_apix,
            np.zeros_like(ctf_with_apix[:, :1]),
            np.ones_like(ctf_with_apix[:, :1]),
        ],
        axis=1,
    ).astype(np.float32)

    voxel_size = float(ctf_with_apix[0, 0])
    print(f"Loaded {images.shape[0]} EMPIAR-10028 particles of size {n} x {n}")
    print(f"Pixel size after downsampling: {voxel_size:.3f} A/pix")
    return (
        images,
        ctf_with_apix[:, 1:].astype(np.float32),
        voxel_size,
        "from_cryosparc_refinement",
        rotations.astype(np.float32),
        (translations_fractional * n).astype(np.float32),
    )


def load_empiar10028_reference_map(
    data_dir=".",
    out_size=128,
    out_path=None,
):
    from skimage.transform import resize

    data_dir = Path(data_dir)
    map_path = data_dir / "empiar10028_cryosparc_reference_128.mrc"
    if not map_path.exists():
        map_path = Path("empiar10028_subsampled") / "empiar10028_cryosparc_reference_128.mrc"
    if not map_path.exists():
        raise FileNotFoundError(f"Could not find {map_path}")

    with mrcfile.open(map_path, permissive=True) as mrc:
        volume = np.asarray(mrc.data, dtype=np.float32)
        voxel_size = float(mrc.voxel_size.x)
    if volume.ndim == 3 and np.isclose(volume.shape, volume.shape[0]).all():
        volume = np.transpose(volume, (2, 1, 0))

    if volume.shape != (out_size, out_size, out_size):
        original_size = volume.shape[0]
        volume = resize(
            volume,
            (out_size, out_size, out_size),
            order=1,
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        ).astype(np.float32)
        voxel_size = voxel_size * original_size / out_size

    if out_path is not None:
        save_mrc(out_path, volume, voxel_size)
        print(f"Saved {out_path} with voxel size {voxel_size:.3f} A")
    else:
        print(f"Loaded {map_path} with voxel size {voxel_size:.3f} A")
    return volume, voxel_size
