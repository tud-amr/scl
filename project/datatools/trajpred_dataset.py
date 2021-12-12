import numpy as np
import pandas as pd
import os
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
#sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # append back in order to import
import yaml
import glob
import torch.nn as nn
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.data import Dataset
from collections import defaultdict


class TrajpredDataset(Dataset):
    """Base class for trajectory prediction datasets with map information

    Inherit this class and generate processed examples from track DataFrames, with the 'extract_from_tracks_df' method.

    The track DataFrame needs to have at least these columns: ['Frame', 'PedestrianId', 'X', 'Y', 'Yaw', 'Vx', 'Vy']
    """
    def __init__(
            self,
            predictor: nn.Module,
            data_dir: str = 'data',
            train: bool = True,
            datasets: list = None,
            experiments: list = None,
            tbptt: int = 3,
            frequency: int = 5,
            stride: int = 10,
            min_track_length: float = 2.5,
            max_effective_velocity: float = 3.0,
            max_effective_curv: float = 1.8,
            save_tracks: bool = False,
            **kwargs
    ):
        # Dataset properties
        self.save_tracks = save_tracks
        self.train = train
        self.data_dir = data_dir
        self.datasets = datasets
        self.experiments = experiments
        self.tbptt = tbptt
        self.frequency = frequency
        self.dt = int(1000//self.frequency)
        self.stride = stride
        self.min_track_length = min_track_length
        self.max_effective_velocity = 2.0
        self.max_effective_curv = 2.0 #rad
        self.min_ade_from_cv = 0.1
        # TODO: Add max/min_track_velocity, max/min_track_acceleration, max/min_angular_velocity

        # Dataset storage
        self._data = None

        if save_tracks:
            self.tracks = None

        experiment_track_file_paths = self._get_raw_file_paths()
        if len(experiment_track_file_paths) == 0: return

        print('Processing datasets:')
        for exp in experiment_track_file_paths:
            print(f'-- {exp}')
            df = self._load_experiment_df(exp)
            if self.save_tracks:
                self.tracks = df
            map_state = self._load_map_state_from_experiment_path(exp)
            self._extract_from_experiment_df(
                predictor,
                tracks=df,
                map_state=map_state,
                experiment_name=os.path.abspath(exp)
            )
        print(f'Total dataset size: {self.__len__()}')

    def _extract_from_experiment_df(self, processor, tracks, map_state, experiment_name):
        """
        Total track : input_horizon - pred_horizon
        """
        data = defaultdict(list)
        pred_horizon = processor.pred_horizon
        input_horizon = processor.input_horizon
        chunk_length = pred_horizon + input_horizon + self.tbptt

        for scene in tqdm(tracks.Scene.unique()):
            scene_tracks = tracks[tracks.Scene == scene]

            # loop through every unique pedestrian's entire track
            for p_id in tqdm(scene_tracks.PedestrianId.unique()):

                # full pedestrian track ordered by frame
                pedestrian_track = scene_tracks[scene_tracks.PedestrianId == p_id].sort_values('Time')

                # Split entire track in chunks. The chunks include a number of tbptt steps.
                for i in range(0, len(pedestrian_track) - chunk_length, self.stride):
                    pedestrian_chunk = pedestrian_track[i: i + chunk_length]

                    if not self._check_chunk(pedestrian_chunk):
                        continue

                    # use 'tbptt' previous input steps so the states of any recurrent units are approximately good.
                    example = defaultdict(list)
                    for t in range(self.tbptt + 1):
                        ego_history_chunk = pedestrian_chunk[t:t+input_horizon]
                        observed_history_chunk = scene_tracks[scene_tracks.Time.isin(ego_history_chunk.Time.values)]

                        # 'extract_example_input_from_df' is an interface function that should be implemented for each
                        # model specifically in it's respective processor. This interface function may be subject to change
                        # when we want to use models that use different input features than are provided in the arguments
                        input = processor.extract_example_input_from_df(
                            ego_history_chunk,
                            observed_history_chunk,
                            map_state
                        )
                        for key, item in input.items():
                            example[key].append(item)

                        # MUST NOT LEAK INTO INPUT. Used to generate targets.
                        ego_future_chunk = pedestrian_chunk[t+input_horizon:t+input_horizon+pred_horizon]

                        # 'extract_example_target_from_df' is an interface function that should be implemented for each
                        # model specifically in it's respective processor. This interface function may be subject to change
                        # when we want to use models that use different targets than are provided in the arguments
                        target = processor.extract_example_target_from_df(ego_history_chunk, ego_future_chunk)
                        example['target'].append(target['target'])

                        # add cv ade / fde
                        if processor.target_mode == 'position':
                            cv_pred = np.array(ego_history_chunk[['X', 'Y']].values[-1] + np.array([np.arange(1, pred_horizon+1), np.arange(1, pred_horizon+1)]).T*ego_history_chunk[['Vx', 'Vy']].values[-1]*(1/self.frequency))
                            example['cv_ade'] = np.mean(np.linalg.norm(ego_future_chunk[['X', 'Y']].values - cv_pred, axis=1))
                            example['cv_fde'] = np.linalg.norm(ego_future_chunk[['X', 'Y']].values[-1] - cv_pred[-1])
                        else:
                            example['cv_ade'] = np.mean(np.linalg.norm(ego_future_chunk[['Vx', 'Vy']].values - ego_history_chunk[['Vx', 'Vy']].values[-1], axis=1))
                            example['cv_fde'] = np.linalg.norm(ego_future_chunk[['Vx', 'Vy']].values[-1] - ego_history_chunk[['Vx', 'Vy']].values[-1])

                    example['Time'] = ego_history_chunk.Time.values[0]
                    example['ExperimentPath'] = experiment_name
                    example['PedestrianId'] = p_id

                    for key, item in example.items():
                        data[key].append(item)

        # convert to np array
        data = {k: np.array(i) for k, i in data.items()}
        if self._data is None:
            self._data = data
        else:
            for k, i in data.items():
                if len(i.shape) > 1:
                    self._data[k] = np.vstack((self._data[k], i))
                else:
                    self._data[k] = np.concatenate((self._data[k], i))


    def _get_raw_file_paths(self):
        """ There are 3 options to specify data files.

        1 - Highest priority is given to 'experiments' if this property is specified then only load these experiments,
        ignore everything else. (Note: experiments from different datasets can't have the same name)

        2 - Second highest priority is given to 'datasets' if this property is specified, all experiments from the
        datasets in the list are loaded and nothing else.

        3 - (Default) Load every experiment of every dataset found in the 'data_dir'.

        Returns:

        """
        mode = 'train' if self.train else 'test'
        paths = []

        if self.experiments:
            all_csv = glob.glob(self.data_dir + '/**/*.csv', recursive=True)
            all_experiments_csv = [f for f in all_csv if any(exp in f for exp in self.experiments)]
            return all_experiments_csv

        if self.datasets:
            for d in self.datasets:
                dataset_dir_path = f'{self.data_dir}/{d}/{mode}'
                paths += glob.glob(f'{dataset_dir_path}/*.csv')
            return paths

        return []
        # all_csv = glob.glob(self.data_dir + '/**/*.csv', recursive=True)
        # all_mode_csv = [f for f in all_csv if mode in f]
        # return all_mode_csv

    @staticmethod
    def _load_map_state_from_experiment_path(f, rgb=False):
        """ An experiment file consists of "{map_name}_{experiment_flag}.csv".
        We load the map_state based on {map_name}

        Args:
            f: experiment file path

        Returns:

        """
        map_name = os.path.basename(f).split('_')[0]
        map_dir = os.path.dirname(os.path.dirname(f))+'/maps'
        if not os.path.exists(f'{map_dir}/{map_name}.yaml'):
            return None

        with open(f'{map_dir}/{map_name}.yaml', 'r') as yaml_file:
            map_state = yaml.safe_load(yaml_file)
        if not rgb:
            map_state['image'] = ~cv2.flip(cv2.imread(f'{map_dir}/{map_name}.png', 0), 0)
        else:
            map_state['image'] = ~cv2.flip(cv2.imread(f'{map_dir}/{map_name}.png'), 0)

        return map_state

    def _load_experiment_df(self, f):
        """ Loads an experiment DataFrame of ['PedestrianId', 'Time', 'X', 'Y', 'Yaw', 'Vx', 'Vy'].
        With some checks and filters to transform the tracks to the specified frequency.

        Args:
            f: experiment file path

        Returns:

        """
        tracks = pd.read_csv(f)

        # TODO: this could be done cleaner.
        tracks_frequencies = np.abs(np.diff(tracks.Time.values))
        tracks_frequency = int(1000 / np.min(tracks_frequencies[tracks_frequencies != 0])) #int(np.mean(np.diff(tracks.Frame.values)))

        if self.frequency > tracks_frequency:
            raise RuntimeError(f'Specified frequency of {self.frequency} too high for {os.path.abspath(f)} with '
                               f'recording frequency of {tracks_frequency}, interpolation not implemented.')

        if tracks_frequency % self. frequency != 0:
            raise RuntimeError(f'Base frequency {os.path.abspath(f)} of {tracks_frequency} is not exactly divisible by '
                               f'specified frequency of {self.frequency}, interpolation not implemented.')

        # map to appropriate frequency
        # dt_stride = (tracks_frequency // self.frequency) * (1000/tracks_frequency)
        dt = (1000/self.frequency)  # Required timestep in ms
        tracks_df = tracks[tracks.Time % dt == 0]
        return tracks_df

    def _check_chunk(self, pedestrian_chunk):
        # filter if track in chunk is too short
        if np.sum(
                np.linalg.norm(np.diff(pedestrian_chunk[['X', 'Y']].values, axis=0), axis=1)) <= self.min_track_length:
            return False

        # filter if unusually high distance jump is detected
        if any((dist * self.frequency) > self.max_effective_velocity for dist in
               np.linalg.norm(np.diff(pedestrian_chunk[['X', 'Y']].values, axis=0), axis=1)):
            return False

        # filter if track in chunk contains large frame gaps
        if np.sum(np.diff(pedestrian_chunk.Time.values)) > (len(pedestrian_chunk) + 5) * (1000 / self.frequency):
            return False

        # filter if unusually high change in velocity direction is detected
        # test = [np.arctan2(dv[1], dv[0]) for dv in np.diff(pedestrian_chunk[['Vx', 'Vy']].values, axis=0)]
        # if any(np.arctan2(dv[1], dv[0]) > self.max_effective_velocity and np.arctan2(dv[1], dv[0]) < (2*np.pi - self.max_effective_curv) for dv in np.diff(pedestrian_chunk[['Vx', 'Vy']].values, axis=0)):
        #         continue

        if any(dangle > self.max_effective_curv for dangle in np.diff(pedestrian_chunk[['Yaw']].values, axis=0)):
            return False

        # filter if track in chunk is too similar to Constant Velocity
        # print(np.mean(np.linalg.norm(pedestrian_chunk[['Vx', 'Vy']].values[1:] - pedestrian_chunk[['Vx', 'Vy']].values[0], axis=1)))
        # if np.mean(np.linalg.norm(pedestrian_chunk[['Vx', 'Vy']].values[1:] - pedestrian_chunk[['Vx', 'Vy']].values[0], axis=1)) <= self.min_ade_from_cv:
        #     continue

        if any([sum(row[-3:]) == 0 for row in pedestrian_chunk.to_numpy()]):
            return False

        return True

    def __len__(self):
        return len(self._data['target'])

    def __getitem__(self, idx):
        inputs = {}
        target = self._data['target'][idx]
        for key, item in self._data.items():
            if 'input' in key or 'cv' in key:
                inputs[key] = item[idx]
        return inputs, target

    @staticmethod
    def add_dataset_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--data_dir', type=str, default='data'),
        parser.add_argument('--train', type=bool, default=True),
        parser.add_argument('--datasets', nargs='+', default=None),
        parser.add_argument('--experiments', nargs='+', default=None),
        parser.add_argument('--tbptt', type=int, default=3),
        parser.add_argument('--frequency', type=int, default=5),
        parser.add_argument('--stride', type=int, default=20),
        parser.add_argument('--min_track_length', type=float, default=2.5),
        parser.add_argument('--save_tracks', type=bool, default=False)
        return parser



# if __name__ == '__main__':
#     model = EwcEthPredictor()
#     d = TrajpredDataset(model.processor, experiments=['obstacles_4-agents'])
#     print('yes')
