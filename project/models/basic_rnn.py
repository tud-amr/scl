from torch import nn
import numpy as np
from argparse import ArgumentParser
from project.utils.tools import matrix_from_angle
import torch
from copy import deepcopy


class Model(nn.Module):
    """
        A basic rnn architecture. Without Social pooling.

        Implemented by:
            - Chadi Salmi
    """

    def __init__(
            self,
            pred_horizon: int = 15,
            ego_input: int = 2,
            ego_rnn: int = 128,
            ego_fc: int = 64,
            **kwargs
    ):
        super(Model, self).__init__()

        self.ego = nn.RNN(input_size=ego_input, hidden_size=ego_rnn, batch_first=True)
        self.predictor = nn.Sequential(
            nn.Linear(in_features=ego_rnn, out_features=ego_fc),
            nn.ReLU(),
            nn.Linear(in_features=ego_fc, out_features=2*pred_horizon)
        )
        self.rnn_ego_size = ego_rnn
        self._rnn_ego_state = None
        self._state_ids = []

    def zero_states(self, batch_size):
        self._rnn_ego_state = torch.zeros([1, batch_size, self.rnn_ego_size])

    def init_states(self, ids):
        ids = list(ids)
        prev_rnn_ego_state = deepcopy(self._rnn_ego_state)
        self.zero_states(len(ids))

        for i, current_id in enumerate(ids):
            if current_id in self._state_ids:
                prev_index = self._state_ids.index(current_id)
                self._rnn_ego_state[0, i, :] = prev_rnn_ego_state[0, prev_index, :]

        self._state_ids = ids

    def forward(self, x):
        N, T, _ = x['ego_input'].size()
        if 'ids' not in x.keys():
            self.zero_states(N)
        else:
            self.init_states(x['ids'])
        _, self._rnn_ego_state = self.ego(x['ego_input'].float(), self._rnn_ego_state)
        x = self.predictor(self._rnn_ego_state.view(N, -1))
        return x

    @staticmethod
    def init_from_args():
        parser = ArgumentParser(add_help=False)

        parser.add_argument("--ego_input", type=int, default=2)
        parser.add_argument("--ego_rnn", type=int, default=128)
        parser.add_argument("--ego_fc", type=int, default=64)
        parser.add_argument("--pred_horizon", type=int, default=15)
        args, _ = parser.parse_known_args()
        return Model(**vars(args)), args


class Processor:
    def __init__(
        self,
        pred_horizon: int = 15,
        target_mode: str = 'velocity',
        rotate_scene: bool = False,
        **kwargs
    ):
        self.pred_horizon = pred_horizon
        self.target_mode = target_mode
        self.rotate_scene = rotate_scene

    def _extract_target(self, ego_track):
        example = {}
        pos = np.array(ego_track[0, :2])
        if self.rotate_scene:
            rot = matrix_from_angle(-ego_track[0, 2])  # map -> ego
        else:
            rot = np.eye(2)

        if self.target_mode == 'position':
            example['target'] = rot.dot((ego_track[1:, :2] - pos).T).T.flatten()
        else:
            example['target'] = rot.dot(ego_track[1:, 3:].T).T.flatten()

        return example

    def _extract_inputs(self, ego_state):
        example = {}
        vel = np.array(ego_state[:, 3:])
        if self.rotate_scene:
            rot = matrix_from_angle(-ego_state[-1, 2])  # map -> ego
        else:
            rot = np.eye(2)
        example['ego_input'] = rot.dot(vel.T).T.flatten()
        return example

    def extract_example_input_from_df(self, ego_history, others_history, map_state):
        # TODO: make this an abstract function in a base class
        latest_ego_state = ego_history[['X', 'Y', 'Yaw', 'Vx', 'Vy']].to_numpy()

        example = self._extract_inputs(latest_ego_state)
        return example

    def extract_example_target_from_df(self, ego_history, ego_future):
        # TODO: make this an abstract function in a base class
        latest_ego_state = ego_history.tail(1)[['X', 'Y', 'Yaw', 'Vx', 'Vy']].to_numpy()[0]
        future_ego_states = ego_future[['X', 'Y', 'Yaw', 'Vx', 'Vy']].to_numpy()

        target = self._extract_target(np.vstack((latest_ego_state, future_ego_states)))
        return target

    def extract_batch_inputs_from_ros(self, tracked_agents, map_state):
        # interface assumes a sequence of history states to be used.
        tracked_agents = tracked_agents[-1]

        examples = {
            'ego_input': [],
            'diff_input': [],
            'ids': []
        }
        for a in tracked_agents:
            ego_state = a['state']
            other_states = [i['state'] for i in tracked_agents]
            example = self._extract_inputs(np.expand_dims(np.array(ego_state), axis=0))

            for key, item in example.items():
                examples[key].append(item)
            examples['ids'].append(a['id'])

        examples = {key: torch.unsqueeze(torch.tensor(item), 1) for key, item in examples.items()}

        return examples

    def map_predictions_to_world(self, tracked_agents, predictions):
        predictions = np.array(predictions).reshape(len(predictions), -1, 2)

        # this assumes tracked_agents and predictions are ordered the same
        for agent, prediction in zip(tracked_agents, predictions):
            if self.target_mode == 'velocity':
                p = np.array([0, 0])
                for t, vel in enumerate(prediction):
                    p = p + (vel * (1/5))
                    prediction[t, :] = p

            if self.rotate_scene:
                R = matrix_from_angle(agent['state'][2])
                prediction[:, :] = R.dot(prediction[:, :].T).T
            prediction[:, :] += agent['state'][:2]

        return predictions

    @staticmethod
    def init_from_args():
        parser = ArgumentParser(add_help=False)

        parser.add_argument("--target_mode", type=str, default="velocity", choices=["velocity", "position"])
        parser.add_argument("--rotate_scene", action='store_true')
        parser.add_argument("--pred_horizon", type=int, default=15)
        args, _ = parser.parse_known_args()
        return Processor(**vars(args)), args
