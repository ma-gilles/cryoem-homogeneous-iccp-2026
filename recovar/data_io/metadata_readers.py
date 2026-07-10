"""Minimal RELION STAR metadata reader for the ICCP cryo-EM notebook."""

from __future__ import annotations

import os
from typing import Optional

import mrcfile
import numpy as np

from recovar.utils import R_from_relion


def _read_star_blocks(star_path: str):
    blocks = {}
    current = None
    reading_columns = False

    with open(star_path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("data_"):
                current = line
                blocks[current] = {"columns": [], "rows": []}
                reading_columns = False
                continue
            if current is None or line == "loop_":
                reading_columns = line == "loop_"
                continue
            if line.startswith("_"):
                blocks[current]["columns"].append(line.split()[0])
                reading_columns = True
                continue
            if reading_columns:
                blocks[current]["rows"].append(line.split())

    return blocks


def _load_star(star_path: str):
    blocks = _read_star_blocks(star_path)
    optics = blocks.get("data_optics")
    particle_blocks = [b for name, b in blocks.items() if name != "data_optics" and b["rows"]]
    if not particle_blocks:
        raise ValueError(f"No particle table found in {star_path}")
    particles = max(particle_blocks, key=lambda b: len(b["rows"]))
    return particles, optics


def _column(table, field: str, dtype=None) -> Optional[np.ndarray]:
    if table is None or field not in table["columns"]:
        return None
    idx = table["columns"].index(field)
    values = [row[idx] for row in table["rows"]]
    return np.asarray(values, dtype=dtype) if dtype is not None else np.asarray(values)


def _optics_map(optics):
    if optics is None:
        return None
    group_idx = optics["columns"].index("_rlnOpticsGroup")
    return {row[group_idx]: row for row in optics["rows"]}


def _get_values(particles, optics, field: str, dtype=None) -> Optional[np.ndarray]:
    optics_values = None
    if optics is not None and field in optics["columns"]:
        particle_groups = _column(particles, "_rlnOpticsGroup", dtype=str)
        optics_by_group = _optics_map(optics)
        field_idx = optics["columns"].index(field)
        optics_values = [optics_by_group[group][field_idx] for group in particle_groups]
    elif field in particles["columns"]:
        return _column(particles, field, dtype=dtype)

    if optics_values is None:
        return None
    return np.asarray(optics_values, dtype=dtype) if dtype is not None else np.asarray(optics_values)


def _apix(particles, optics):
    values = _get_values(particles, optics, "_rlnImagePixelSize", dtype=np.float64)
    if values is not None:
        return values
    det = _get_values(particles, optics, "_rlnDetectorPixelSize", dtype=np.float64)
    mag = _get_values(particles, optics, "_rlnMagnification", dtype=np.float64)
    if det is not None and mag is not None:
        return det * 1e4 / mag
    return None


def _resolution(star_path: str, particles, optics):
    values = _get_values(particles, optics, "_rlnImageSize", dtype=np.float64)
    if values is not None:
        return values

    image_name = _get_values(particles, optics, "_rlnImageName", dtype=str)
    if image_name is None:
        return None
    mrcs_path = image_name[0].split("@")[-1]
    if not os.path.isabs(mrcs_path):
        mrcs_path = os.path.join(os.path.dirname(star_path), mrcs_path)
    if not os.path.exists(mrcs_path):
        return None
    with mrcfile.open(mrcs_path, mode="r", header_only=True) as mrc:
        return np.full(len(image_name), int(mrc.header.ny), dtype=np.float64)


def parse_poses_from_star(star_path: str, D: int):
    particles, optics = _load_star(star_path)

    euler = np.stack(
        [
            _get_values(particles, optics, "_rlnAngleRot", dtype=np.float64),
            _get_values(particles, optics, "_rlnAngleTilt", dtype=np.float64),
            _get_values(particles, optics, "_rlnAnglePsi", dtype=np.float64),
        ],
        axis=1,
    )
    rotations = R_from_relion(euler, degrees=True)

    tx = _get_values(particles, optics, "_rlnOriginXAngst", dtype=np.float64)
    ty = _get_values(particles, optics, "_rlnOriginYAngst", dtype=np.float64)
    if tx is not None and ty is not None:
        apix = _apix(particles, optics)
        if apix is None:
            raise ValueError("STAR file has Angstrom origins but no pixel size.")
        trans_pixels = np.stack([tx / apix, ty / apix], axis=1)
    else:
        tx = _get_values(particles, optics, "_rlnOriginX", dtype=np.float64)
        ty = _get_values(particles, optics, "_rlnOriginY", dtype=np.float64)
        trans_pixels = np.stack([tx, ty], axis=1) if tx is not None and ty is not None else np.zeros((len(euler), 2))

    resolution = _resolution(star_path, particles, optics)
    if resolution is not None:
        translations = trans_pixels / resolution.astype(np.float64).reshape(-1, 1)
    else:
        translations = trans_pixels / float(D)
    return rotations, translations


def parse_ctf_from_star(star_path: str, D: int):
    particles, optics = _load_star(star_path)
    n = len(particles["rows"])

    dfu = _get_values(particles, optics, "_rlnDefocusU", dtype=np.float64)
    dfv = _get_values(particles, optics, "_rlnDefocusV", dtype=np.float64)
    dfang = _get_values(particles, optics, "_rlnDefocusAngle", dtype=np.float64)
    volt = _get_values(particles, optics, "_rlnVoltage", dtype=np.float64)
    cs = _get_values(particles, optics, "_rlnSphericalAberration", dtype=np.float64)
    w = _get_values(particles, optics, "_rlnAmplitudeContrast", dtype=np.float64)
    phase = _get_values(particles, optics, "_rlnPhaseShift", dtype=np.float64)
    if phase is None:
        phase = np.zeros(n, dtype=np.float64)

    for name, values in [
        ("_rlnDefocusU", dfu),
        ("_rlnDefocusV", dfv),
        ("_rlnDefocusAngle", dfang),
        ("_rlnVoltage", volt),
        ("_rlnSphericalAberration", cs),
        ("_rlnAmplitudeContrast", w),
    ]:
        if values is None:
            raise ValueError(f"STAR file missing required CTF field {name}")

    orig_apix = _apix(particles, optics)
    orig_D = _resolution(star_path, particles, optics)
    if orig_apix is not None and orig_D is not None:
        new_apix = orig_D * orig_apix / float(D)
    elif orig_apix is not None:
        new_apix = orig_apix
    else:
        new_apix = np.ones(n, dtype=np.float64)

    return np.column_stack([new_apix, dfu, dfv, dfang, volt, cs, w, phase]).astype(np.float64)

