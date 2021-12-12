import torch
from torch import nn
import numpy as np
from project.utils.tools import matrix_from_angle, extract_occ_grid
from project.models.lit_autoencoder import LitAutoEncoder
from copy import deepcopy
import os


class StateDiffs(nn.Module):
    """
        An rnn architecture, with addition stream of info about others.
        The info consists of the difference between states [dx, dy ,dvx, dvy, d dist, d angle].

        With Euclidean pooling. (e.g. euclidean attention)

        Implemented by:
            - Chadi Salmi
    """

    def __init__(
            self,
            pred_horizon: int = 15,
            input_horizon: int = 1,
            ae_state_dict: str = 'saves/misc/eth_autoencoder.h5',
            target_mode: str = 'velocity',
            rotate_scene: bool = True,
            map_size: int = 8,
            **kwargs
    ):
        super(StateDiffs, self).__init__()

        self.pred_horizon = pred_horizon
        self.input_horizon = input_horizon
        self.ae_state_dict = ae_state_dict
        self.map_size = map_size
        self.target_mode = target_mode
        self.rotate_scene = rotate_scene

        self._build_model()

    def _build_model(self):
        self.ego = nn.LSTM(2*self.input_horizon, 32, batch_first=True)
        self.others = nn.LSTM(30, 128, batch_first=True)
        self.ae = LitAutoEncoder()
        if self.ae_state_dict is not None and os.path.exists(self.ae_state_dict):
            self.ae.load_state_dict(torch.load(self.ae_state_dict))
        self.ae.freeze()
        self.map = nn.LSTM(64, 256, batch_first=True)
        self.concat = nn.LSTM(416, 512, batch_first=True)
        self.linear = nn.Sequential(
            # nn.Dropout(p=0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            # nn.Dropout(p=0.1),
            nn.Linear(256, 2*self.pred_horizon)
        )

        # Internal state cache of the prediction model
        # _state_ids defines the tracking ids of the cached states
        self.lstm_ego_size = 32
        self.lstm_others_size = 128
        self.lstm_map_size = 256
        self.lstm_concat_size = 512
        self._lstm_map_state = None
        self._lstm_concat_state = None
        self._lstm_ego_state = None
        self._lstm_others_state = None
        self._state_ids = []

    def _ae_loop_tbptt(self, x):
        for i in range(x.shape[1]):
            if i == 0:
                output = self.ae(x[:, i, :, :].unsqueeze(1)).unsqueeze(1)
            else:
                output = torch.cat([output, self.ae(x[:, i, :, :].unsqueeze(1)).unsqueeze(1)], dim=1)
        return output

    def forward(self, i):
        x = i['ego_input'].float()
        y = i['others_input'].float()
        z = i['map_input'].float()

        N, T, _ = x.size()
        if 'ids' not in i.keys():
            self.zero_states(N)
        else:
            self.init_states(i['ids'])

        x, self._lstm_ego_state = self.ego(x, self._lstm_ego_state)
        y, self._lstm_others_state = self.others(y, self._lstm_others_state)
        z = self._ae_loop_tbptt(z)
        z, self._lstm_map_state = self.map(z, self._lstm_map_state)
        x = torch.cat((x, y, z), dim=2)
        x, self._lstm_concat_state = self.concat(x, self._lstm_concat_state)
        pred = self.linear(x)[:, -1, :]
        return pred

    def zero_states(self, batch_size):
        self._lstm_ego_state = (
            torch.zeros([1, batch_size, self.lstm_ego_size]),
            torch.zeros([1, batch_size, self.lstm_ego_size])
        )
        self._lstm_concat_state = (
            torch.zeros([1, batch_size, self.lstm_concat_size]),
            torch.zeros([1, batch_size, self.lstm_concat_size])
        )
        self._lstm_others_state = (
            torch.zeros([1, batch_size, self.lstm_others_size]),
            torch.zeros([1, batch_size, self.lstm_others_size])
        )
        self._lstm_map_state = (
            torch.zeros([1, batch_size, self.lstm_map_size]),
            torch.zeros([1, batch_size, self.lstm_map_size])
        )

    def init_states(self, ids):
        ids = list(ids)
        prev_lstm_ego_state = deepcopy(self._lstm_ego_state)
        prev_lstm_others_state = deepcopy(self._lstm_others_state)
        prev_lstm_map_state = deepcopy(self._lstm_map_state)
        prev_lstm_concat_state = deepcopy(self._lstm_concat_state)
        self.zero_states(len(ids))

        for i, current_id in enumerate(ids):
            if current_id in self._state_ids:
                prev_index = self._state_ids.index(current_id)
                self._lstm_ego_state[0][0, i, :] = prev_lstm_ego_state[0][0, prev_index, :]
                self._lstm_ego_state[1][0, i, :] = prev_lstm_ego_state[1][0, prev_index, :]
                self._lstm_others_state[0][0, i, :] = prev_lstm_others_state[0][0, prev_index, :]
                self._lstm_others_state[1][0, i, :] = prev_lstm_others_state[1][0, prev_index, :]
                self._lstm_map_state[0][0, i, :] = prev_lstm_map_state[0][0, prev_index, :]
                self._lstm_map_state[1][0, i, :] = prev_lstm_map_state[1][0, prev_index, :]
                self._lstm_concat_state[0][0, i, :] = prev_lstm_concat_state[0][0, prev_index, :]
                self._lstm_concat_state[1][0, i, :] = prev_lstm_concat_state[1][0, prev_index, :]
        self._state_ids = ids

    def _extract_target(self, ego_track):
        example = {}
        # vel = np.array(ego_track[0, 3:])
        pos = np.array(ego_track[0, :2])
        angle = -ego_track[0, 2]  #np.arctan2(vel[1], vel[0])
        if self.rotate_scene:
            rot = matrix_from_angle(angle)  # map -> ego
        else:
            rot = np.eye(2)

        if self.target_mode == 'position':
            example['target'] = rot.dot((ego_track[1:, :2] - pos).T).T.flatten()
        else:
            example['target'] = rot.dot(ego_track[1:, 3:].T).T.flatten()

        return example

    def _extract_inputs(self, ego_input_horizon, other_states, map_state):
        ego_state = ego_input_horizon[-1]
        if len(other_states) > 1:
            other_states = np.array([state for state in other_states if (state != ego_state).any()])

        vel = np.array(ego_state[3:])
        pos = np.array(ego_state[:2])
        angle = -np.arctan2(vel[1], vel[0])

        if self.rotate_scene:
            rot = matrix_from_angle(-ego_state[2])  # map -> ego
        else:
            rot = np.eye(2)

        # ego_state[:2] = rot.dot(ego_state[:2])
        # relative other states
        other_states[:, :2] -= ego_state[:2]  # relative position
        other_states[:, 3:] -= ego_state[3:]  # relative velocity

        # rotation for ego input
        ego_input_horizon[:, 3:] = rot.dot(ego_input_horizon[:, 3:].T).T

        # rotation for relative others input
        for state in other_states:
            state[:2] = rot.dot(state[:2])
            state[3:] = rot.dot(state[3:])

        diff_input = []  # right format [dx, dy ,dvx, dvy, dist, angle]
        for state in other_states:
            diff_input.append([
                state[0],
                state[1],
                state[3],
                state[4],
                np.sqrt(state[0]**2+state[1]**2),
                np.arctan2(state[1], state[0])
            ])
        while len(diff_input) < 5:
            diff_input.append(diff_input[0])
        diff_input = np.array(diff_input)
        diff_input = diff_input[diff_input[:, 4].argsort()]
        if len(diff_input) > 5:
            diff_input = diff_input[:5, :]

        example = {}
        example['map_input'] = extract_occ_grid(pos, -angle, map_state, self.map_size, 60)
        example['others_input'] = diff_input.flatten()
        example['ego_input'] = np.array([e[3:] for e in ego_input_horizon]).flatten()
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
            'map_input': [],
            'ids': []
        }
        for a in tracked_agents:
            ego_state = a['state']
            other_states = [i['state'] for i in tracked_agents]
            ego_history = [aa['state'] for hist_step in history for aa in hist_step if aa['id'] == a['id']]
            if len(ego_history) < self.input_horizon:
                continue
            example = self._extract_inputs(np.array(ego_history), np.array(other_states), map_state)

            for key, item in example.items():
                examples[key].append(item)
            examples['ids'].append(a['id'])

        examples = {key: torch.unsqueeze(torch.tensor(item), 1) for key, item in examples.items()}

        return examples

    def map_predictions_to_world(self, tracked_agents, predictions):
        predictions = np.array(predictions).reshape(len(predictions), -1, 2)
        world_predictions = np.empty([len(predictions), self.pred_horizon, 4])  # [x, y, vx, vy]

        # this assumes tracked_agents and predictions are ordered the same
        for i, (agent, prediction) in enumerate(zip(tracked_agents, predictions)):
            if self.rotate_scene:
                R = matrix_from_angle(tracked_agents[i]['state'][2])
            else:
                R = np.eye(2)

            prediction = R.dot(prediction.T).T

            if self.target_mode == 'velocity':
                world_predictions[i, :, 2:] = prediction
                p = np.array([0, 0])
                for t, vel in enumerate(prediction):
                    p = p + (vel * (1/5))
                    world_predictions[i, t, :2] = p
            elif self.target_mode == 'position':
                world_predictions[i, :, :2] = prediction
                world_predictions[i, 1:, 2:] = np.diff(prediction, axis=0) * 5
                world_predictions[i, 0, 2:] = world_predictions[i, 1, 2:]
                # world_predictions[i, 0, 2:] = agent['state'][3:]

            world_predictions[i, :, :2] += agent['state'][:2]

        return world_predictions
