import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
from tqdm import tqdm
from copy import deepcopy
from project.utils.tools import matrix_from_angle


def generate_gif_single_pedestrian(dataset, exp, model, processor, p_id, map_state=None, save_path='../../saves/misc/generated.gif', gif_length=100, gif_start=0):
    # ------------
    # Qualitative analysis
    # TODO: add state propagation # TODO: make sure pedestrian tracks don't contain frame jumps
    # ------------
    # experiment_track_file_paths = dataset._get_raw_file_paths()
    # exp = experiment_track_file_paths[dataset.experiments.index(experiment)]
    df = dataset._load_experiment_df(exp)
    map_state = dataset._load_map_state_from_experiment_path(exp)
    vis_map_state = dataset._load_map_state_from_experiment_path(exp, rgb=True)

    color_1 = (89, 110, 145)
    color_2 = (100, 189, 116)
    # color_1 = (255, 0, 0)
    # color_2 = (0, 255, 0)
    scale_percent = 100  # percent of original size
    if vis_map_state is not None:
        img = ~vis_map_state['image']
        origin = vis_map_state['origin']
        resolution = vis_map_state['resolution']

        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        canvas = Image.fromarray(resized)
    else:
        origin = [-10, -10, 0]
        resolution = 0.05
        canvas = Image.fromarray(np.ones([600, 600, 3], dtype=np.uint8)*255)

    pen = Pen(scale_percent, origin, resolution)

    pred_horizon = processor.pred_horizon
    prev_horizon = processor.prev_horizon -1
    chunk_length = pred_horizon + prev_horizon + 1

    # full pedestrian track ordered by frame
    pedestrian_track = df[df.PedestrianId == p_id].sort_values('Time')

    # part of pedestrian track because full track gif overloads my ram.
    if len(pedestrian_track) > gif_length + chunk_length:
        pedestrian_track = pedestrian_track[gif_start: gif_start + gif_length + chunk_length]

    # store images to generate gif
    images = []

    # Loop through track
    for t in tqdm(range(prev_horizon + 1, len(pedestrian_track) - pred_horizon)):
        ego_history_chunk = pedestrian_track[t:t+prev_horizon+1]
        observed_history_chunk = df[df.Time.isin(ego_history_chunk.Time.values)]

        # 'extract_example_input_from_df' is an interface function that should be implemented for each
        # model specifically in it's respective processor. This interface function may be subject to change
        # when we want to use models that use different input features than are provided in the arguments
        input = processor.extract_example_input_from_df(
            ego_history_chunk,
            observed_history_chunk,
            map_state
        )

        # predict
        for key in input.keys():
            input[key] = torch.tensor(input[key]).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            prediction = model(input)
            prediction = np.array(prediction).reshape(-1, 2)
        # post
        if processor.target_mode == 'velocity':
            p = np.array([0, 0])
            for t, vel in enumerate(prediction):
                p = p + (vel * (1/dataset.frequency))
                prediction[t, :] = p
        yaw = ego_history_chunk.Yaw.values[-1]
        if processor.rotate_scene:
            R = matrix_from_angle(yaw)
            prediction[:, :] = R.dot(prediction[:, :].T).T
        prediction += ego_history_chunk[['X', 'Y']].values[-1]

        # used to generate targets. MUST NOT LEAK INTO INPUT
        ego_future_chunk = pedestrian_track[t+prev_horizon+1:t+prev_horizon+1+pred_horizon]

        # visualize
        im = deepcopy(canvas)
        draw = ImageDraw.Draw(im)
        lastly_observed = df[df.Time == max(ego_history_chunk.Time.values)][['X', 'Y']].values
        for point in lastly_observed:
            pen.draw_ellipse(draw, point, color=color_2)

        for point in prediction:
            pen.draw_ellipse(draw, point, color=color_1)
        images.append(im)

    images[0].save(save_path, save_all=True, append_images=images[1:], optimize=True, duration=(1000/dataset.frequency), loop=0)

    return images


class Pen:
    def __init__(self, scale_percent, origin, resolution):
        self.scale_percent = scale_percent
        self.origin = origin
        self.resolution = resolution

    def draw_ellipse(self, draw, point, color, size=10):
        point *= (self.scale_percent / 100)
        origin = np.array(self.origin[:2:])*(self.scale_percent / 100)
        point = (point - origin) // (self.resolution)
        x, y = point
        draw.ellipse(
            (x - size, y - size, x + size, y + size),
            fill=color,
            outline=(0,0,0)
        )
