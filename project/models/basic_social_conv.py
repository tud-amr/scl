import torch
from torch import nn
import numpy as np
from argparse import ArgumentParser
from project.utils.tools import matrix_from_angle
from copy import deepcopy


class Model(nn.Module):
    """
        A conv2d architecture, that predicts the differences from CV from a local snapshot of surrounding agents.

        Implemented by:
            - Chadi Salmi
    """

    def __init__(
            self,
            pred_horizon: int = 15,
            input_horizon: int = 1,
            **kwargs
    ):
        super(Model, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(512, 128)
        self.out = nn.Linear(128, pred_horizon*2)

    def forward(self, x):
        cv = x['const_velocity_input']
        N, _, P = cv.size()
        x = self.encoder(x['social_snapshot_input'].float())
        x = self.fc(x.view(-1, 512))
        x = self.out(x)
        x = x + cv.float().view(N, P)
        return x

    @staticmethod
    def init_from_args():
        parser = ArgumentParser(add_help=False)

        parser.add_argument("--input_horizon", type=int, default=1)
        parser.add_argument("--pred_horizon", type=int, default=15)
        args, _ = parser.parse_known_args()
        return Model(**vars(args)), args


class Processor:
    def __init__(
        self,
        pred_horizon: int = 15,
        input_horizon: int = 1,
        target_mode: str = 'velocity',
        **kwargs
    ):
        self.pred_horizon = pred_horizon
        self.input_horizon = input_horizon
        self.target_mode = target_mode

    def _extract_target(self, ego_track):
        example = {}
        rot = matrix_from_angle(-ego_track[0, 2])  # map -> ego
        example['target'] = rot.dot(ego_track[1:, 3:].T).T.flatten()

        return example

    def _extract_inputs(self, ego_input_horizon, other_states, map_state):
        ego_state = ego_input_horizon[-1]
        if len(other_states) > 1:
            other_states = np.array([state for state in other_states if (state != ego_state).any()])

        social_grid = np.zeros([60, 60])
        grid_size = 5 # meters

        rot = matrix_from_angle(-ego_state[2])  # map -> ego

        # relative position
        other_states[:, :2] -= ego_state[:2]
        for state in other_states:
            state[:2] = rot.dot(state[:2])

        for pos in other_states[:, :2]:
            if np.linalg.norm(pos) < grid_size/2:
                center = np.array((pos * (60/grid_size)) + 30).astype(np.int32)

                p_size = 2
                for i in range(-p_size, p_size+1):
                    for j in range(-p_size, p_size+1):
                        if center[0]+i >= 60 or center[0]+i < 0 or center[1]+j >= 60 or center[1]+j < 0:
                            continue
                        social_grid[center[0]+i, center[1]+j] = 1

        example = {}
        example['social_snapshot_input'] = social_grid
        example['const_velocity_input'] = np.array(list(rot.dot(ego_state[3:])) * self.pred_horizon)
        return example

    def extract_example_input_from_df(self, ego_history, others_history, map_state):
        ego_input_horizon = ego_history.tail(self.input_horizon)[['X', 'Y', 'Yaw', 'Vx', 'Vy']].to_numpy()
        latest_others_state = others_history[others_history.Time == ego_history.tail(1).Time.values[0]][['X', 'Y', 'Yaw', 'Vx', 'Vy']].to_numpy()

        example = self._extract_inputs(ego_input_horizon, latest_others_state, map_state)
        return example

    def extract_example_target_from_df(self, ego_history, ego_future):
        latest_ego_state = ego_history.tail(1)[['X', 'Y', 'Yaw', 'Vx', 'Vy']].to_numpy()[0]
        future_ego_states = ego_future[['X', 'Y', 'Yaw', 'Vx', 'Vy']].to_numpy()

        target = self._extract_target(np.vstack((latest_ego_state, future_ego_states)))
        return target

    def extract_batch_inputs_from_ros(self, tracked_agents, map_state):
        # interface assumes a sequence of history states to be used.
        history = tracked_agents
        tracked_agents = tracked_agents[-1]

        examples = {
            'ego_input': [],
            'others_input': [],
            'ids': []
        }
        for a in tracked_agents:
            ego_state = a['state']
            other_states = [i['state'] for i in tracked_agents]
            ego_history = [aa['state'] for hist_step in history for aa in hist_step if aa['id'] == a['id']]
            if len(ego_history) < self.input_horizon:
                continue
            example = self._extract_inputs(np.array(ego_history), np.array(other_states), None)

            for key, item in example.items():
                examples[key].append(item)
            examples['ids'].append(a['id'])

        examples = {key: torch.unsqueeze(torch.tensor(item), 1) for key, item in examples.items()}

        return examples

    # def map_predictions_to_world(self, tracked_agents, predictions):
    #     predictions = np.array(predictions).reshape(len(predictions), -1, 2)
    #
    #     # this assumes tracked_agents and predictions are ordered the same
    #     for agent, prediction in zip(tracked_agents, predictions):
    #         if self.target_mode == 'velocity':
    #             p = np.array([0, 0])
    #             for t, vel in enumerate(prediction):
    #                 p = p + (vel * (1/5))
    #                 prediction[t, :] = p
    #
    #         if self.rotate_scene:
    #             R = matrix_from_angle(agent['state'][2])
    #             prediction[:, :] = R.dot(prediction[:, :].T).T
    #         prediction[:, :] += agent['state'][:2]
    #
    #     return predictions

    @staticmethod
    def init_from_args():
        parser = ArgumentParser(add_help=False)

        parser.add_argument("--target_mode", type=str, default="velocity", choices=["velocity", "position"])
        # parser.add_argument("--rotate_scene", action='store_true')
        parser.add_argument("--pred_horizon", type=int, default=15)
        parser.add_argument("--input_horizon", type=int, default=1)
        args, _ = parser.parse_known_args()
        return Processor(**vars(args)), args
