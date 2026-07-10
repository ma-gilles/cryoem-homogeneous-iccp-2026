from scipy.spatial.transform import Rotation


def uniform_rotation_sampling(n_images, grid_size, seed=0):
    return Rotation.random(n_images, random_state=seed).as_matrix()

