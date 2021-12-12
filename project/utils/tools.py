import numpy as np
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
#sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # append back in order to import
import random
from copy import deepcopy

def yaw_from_quaternion(q):
    """Convert from quaternion to yaw.

    Args:
        q (object): Quaternion object.

    Returns:
        float: Yaw angle in radians.
    """
    yaw = np.arctan2(2.0 * (q.y * q.z + q.w * q.x), q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z)
    pitch = np.arcsin(-2.0 * (q.x * q.z - q.w * q.y))
    roll = np.arctan2(2.0 * (q.x * q.y + q.w * q.z), q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z)
    return roll


def matrix_from_angle(current_angle):
    """Return a matrix representing the current angle in radians .

    Args:
        current_angle (float): angle in radians.

    Returns:
        np.Array(): Rotation matrix
    """
    c, s = np.cos(current_angle), np.sin(current_angle)
    return np.array(((c, -s), (s, c)))


def extract_apg(observed, n_bins, max_range, normalize=True, invert=True):
    """Extract the angular pedestrian grid from a list of observed pedestrians .

    Args:
        observed (np.Array()): Positions of observed pedestrians
        n_bins (int): Angular grid resolution
        max_range (float): Max range for pedestrians to consider
        normalize (bool, optional): Normalize angular grid values. Defaults to True.
        invert (bool, optional): Invert angular grid values. Defaults to True.

    Returns:
        [type]: [description]
    """

    apg = max_range * np.ones(n_bins)

    observed = observed[[None not in o for o in observed]]
    if len(observed.shape) >= 2:
        # drop pedestrian further away than 'apg_max_range'
        # also drop if too close (the pedestrian himself)
        dists = np.sum(np.square(observed), axis=1)
        observed = [obs for i, obs in enumerate(observed) if 0.01 < dists[i] < max_range ** 2]

        # store pedestrians in apg grid
        for pos in observed:
            phi = np.arctan2(pos[1], pos[0])
            rad_idx = int(np.floor(phi * (n_bins / (2 * np.pi))))
            apg[rad_idx] = min(apg[rad_idx], np.linalg.norm(pos))

    if invert:
        apg = max_range-apg

    if normalize:
        apg /= max_range

    return apg
    # return apg[np.newaxis, :]


def extract_occ_grid(pos, angle, map_state, grid_size_meters, grid_size_bins, lookahead=0.5):
    """Extracts a grid from a position in a map . The extracted patch is aligned with the angle.
    NOTE: images are read as (y,x)

    Args:
        pos ([x, y]): Center position of the grid.
        angle (float): Angle to extract a patch along, in radians.
        map_state (object): Contains entire map, with it's resolution and origin.
        grid_size_meters (float): How big should the extracted grid be, in meters?
        grid_size_bins (int): What should the resolution of the grid be, in total bins? 
        lookahead (float): What fraction of the gridpatch should look forward? 

    Returns:
        np.Array([grid_size_bins, grid_size_bins]): scaled and aligned, extracted grid patch
    """
    if not map_state:
        return np.zeros([grid_size_bins, grid_size_bins])

    grid_size = grid_size_meters / map_state['resolution']  # grid size meters -> bins
    map = map_state['image']  # map is (y,x)
    x, y = pos
    origin_x, origin_y, _ = map_state['origin']
    center = (
        (x - origin_x) / map_state['resolution'],
        (y - origin_y) / map_state['resolution']
    )
    h, w = map.shape
    rotation_angle = (angle * 180 / np.pi)

    # Extract local patch to rotate
    local_map = map[
        round(center[1]) - round(grid_size): round(center[1]) + round(grid_size),
        round(center[0]) - round(grid_size): round(center[0]) + round(grid_size),
    ]
    h, w = local_map.shape

    # if h != grid_size or w != grid_size:
    #     np.pad(local_map, ((3, 0), (3, 0)))

    M = cv2.getRotationMatrix2D( (grid_size, grid_size), rotation_angle, 1.0)
    rotated_local_map = cv2.warpAffine(local_map.astype("float32"), M, (w, h))
    local_grid = rotated_local_map[
        round(grid_size) - round(0.5 * grid_size): round(grid_size) + round(0.5 * grid_size),
        round(grid_size) - round((1-lookahead) * grid_size): round(grid_size) + round(lookahead * grid_size)
    ]
    local_grid = np.abs(local_grid/(np.max(local_grid)+1e-9))
    if local_grid.shape[1] == 0:
        local_grid = np.zeros([round(grid_size), round(grid_size)])

    local_grid = cv2.resize(local_grid, (grid_size_bins, grid_size_bins), interpolation=cv2.INTER_AREA)

    return local_grid


def constant_velocity_predictor(inputs, pred_horizon, target_mode='velocity'):

    if target_mode == 'velocity':
        velocities = inputs['ego_input'].repeat(1, 1, pred_horizon)
    else:
        raise NotImplementedError

    return velocities

def split_task(task, val_split=0.2):
    deck = list(range(len(task)))
    random.shuffle(deck)
    val_idxs = deck[:int(val_split*len(task))]

    train_mask = np.ones(len(task), np.bool)
    train_mask[val_idxs] = 0
    task_train = deepcopy(task)
    for key, value in task_train._data.items():
        task_train._data[key] = value[train_mask]

    val_mask = np.zeros(len(task), np.bool)
    val_mask[val_idxs] = 1
    task_val = deepcopy(task)
    for key, value in task_val._data.items():
        task_val._data[key] = value[val_mask]

    return task_train, task_val

