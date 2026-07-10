import mrcfile
import numpy as np
from scipy.spatial.transform import Rotation as SciPyRot


def write_mrc(file, ar, voxel_size=None):
    if ar.ndim == 3 and np.isclose(ar.shape, ar.shape[0]).all():
        ar = np.transpose(ar, (2, 1, 0))
    with mrcfile.new(file, overwrite=True) as mrc:
        mrc.set_data(ar.real.astype(np.float32))
        if voxel_size is not None:
            mrc.voxel_size = voxel_size


def R_from_relion(euler_: np.ndarray, degrees: bool = True) -> np.ndarray:
    angles = euler_.copy()
    angles = angles.reshape(1, 3) if angles.shape == (3,) else angles
    angles[:, 0] = angles[:, 0] + 90.0
    angles[:, 2] = angles[:, 2] - 90.0
    matrices = SciPyRot.from_euler("zxz", angles, degrees=degrees).as_matrix()
    frame_adjust = np.array([[1, -1, 1], [-1, 1, -1], [1, -1, 1]], dtype=np.float64)
    return matrices * frame_adjust

