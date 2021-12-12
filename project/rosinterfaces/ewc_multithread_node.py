import rospy
import pandas as pd
import torch
import threading
from argparse import ArgumentParser
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from time import time, sleep
from copy import deepcopy
import os
from collections import defaultdict
from project.rosinterfaces.rosbackbone import RosBackbone
from project.training_routines.ewc import EwcPredictor
from project.datatools.trajpred_dataset import TrajpredDataset


class EwcNode(RosBackbone):
    """ Self-supervised Continual Learning node for Pedestrian Prediction Models.

    Every 'task_time' a dataframe is aggregated of the form:

    [Time', 'Scene', 'PedestrianId', 'X', 'Y', 'Yaw', 'Vx', 'Vy']

    After 'task_time' the model is trained using the aggregated dataframe.

    To prevent catastrophic forgetting, our framework uses Elastic Weight Consolidation and the Rehearsal of a small
    set of examples 'coreset'.
    """
    def __init__(
        self,
        prediction_model,
        save_dir,
        inference_frequency: float = 5.0,
        task_time: int = 100,
        batch_size: int = 10,
        tbptt: int = 5,
        max_epochs: int = 10,
        stride: int = 2,
        **kwargs
    ):
        RosBackbone.__init__(self)

        # Model stuff
        self.model = prediction_model
        self.model_copy = deepcopy(prediction_model)
        self.prediction_dt = 1/5  #1/self.model.predictor.frequency
        self.save_dir = save_dir
        self.inference_frequency = inference_frequency
        self.aggregation_frequency = 5
        self.model_state = {}

        # Time keeping
        self.last_inference = None
        self.last_aggregation = None

        # Aggregation stuff
        # self.last_time = time()
        self.step = 0
        self.task_time = task_time
        self.batch_size = batch_size
        self.tbptt = tbptt
        self.stride = stride
        self.input_horizon = self.model.predictor.input_horizon
        self.pred_horizon = self.model.predictor.pred_horizon
        self.max_epochs = max_epochs

        # Buffer / Task data
        self.agg_dataset = pd.DataFrame(columns={'Time', 'Scene', 'PedestrianId', 'X', 'Y', 'Yaw', 'Vx', 'Vy'})
        self.history = []
        self.current_task = defaultdict(list)
        rospy.sleep(1)

        # for now the two frequencies have to be the same
        # TODO: add higher inference frequency in the same node
        # assert(self.inference_frequency == self.prediction_frequency)

    def inference(self):
        while not rospy.is_shutdown():
            # ------------
            # time keeping
            # ------------
            self.check_time('inference')

            # ------------
            # fill buffer for inference (input_horizon)
            # ------------
            self.update_inference_buffer()

            # ------------
            # inference
            # ------------
            with torch.no_grad():
                # pre process current inputs and handle state propagation
                batch = self.model.predictor.extract_batch_inputs_from_ros(self.history, self.map_state)
                self.update_model_state(batch['ids'])

                # run inference on current state
                self.model.predictor.set_states(self.model_state['state'])
                predictions = self.model.predictor(batch, reset_state=False)
                self.model_state['state'] = self.model.predictor.get_states()

                # post process predictions
                trajectories = self.model.predictor.map_predictions_to_world(self.history[-1], predictions)

            # publish visualization msg of trajectories to ros
            self.visualize_trajectories(self.history[-1], trajectories[:, :, :2])

            # publish msg of trajectories to ros for motion planning
            self.publish_trajectories(self.history[-1], trajectories)

            if self.last_inference is None: self.last_inference = time()
            if (time() - self.last_inference) < (1 / self.inference_frequency):
                sleep((1 / self.inference_frequency) - (time() - self.last_inference))
            else:
                print("WARN : Not keeping up to rate")
            self.last_inference = time()

    def online_learning(self):
        print('aggregation')
        self.task_start_time = time()
        while not rospy.is_shutdown():
            # ------------
            # time keeping
            # ------------
            self.check_time('aggregation')

            # ------------
            # save to agg DF
            # ------------
            self.save_state_to_df()

            # train on current_task and refresh task every 'task_time' secs for 'max_epochs' epochs
            if time() - self.task_start_time > self.task_time:
                # ------------
                # Train model online
                # ------------
                self.train_online()

    def train_online(self):
        # ------------
        # Extract examples
        # ------------
        self.agg_dataset.to_csv(f'{self.save_dir}/agg_data_dir/train/task_{len(self.model_copy.prev_tasks)}.csv')
        dataset = TrajpredDataset(
            self.model_copy.predictor,
            tbptt=self.tbptt,
            stride=self.stride,  # delta t between examples, if lower than 'tbptt + pred_horizon', examples can overlap.
            min_track_length=1.0)
        dataset._extract_from_experiment_df(
            self.model_copy.predictor,
            tracks=self.agg_dataset,
            map_state=self.map_state,
            experiment_name='ewc'
        )
        task_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        print(f"Train task {len(self.model_copy.prev_tasks)} containing {len(task_loader.dataset)} examples")

        # ------------
        # Combine with coreset
        # ------------
        task_loader = self.model_copy.add_coreset_to_loader(task_loader)

        # ------------
        # trainer
        # ------------
        trainer = pl.Trainer(
            default_root_dir=self.save_dir,
            max_epochs=self.max_epochs
        )

        # ------------
        # online training
        # ------------
        trainer.fit(self.model_copy, task_loader)
        self.model_copy.register_ewc_params(trainer.train_dataloader)

        # ------------
        # saving
        # ------------
        ckpt_name = f'{self.save_dir}/task_{len(self.model_copy.prev_tasks) - 1}.ckpt'
        trainer.save_checkpoint(ckpt_name)
        self.previous_ckpt = ckpt_name

        # ------------
        # update coreset
        # ------------
        self.model_copy.update_coreset(task_loader)

        # ------------
        # clear from memory and reset states
        # ------------
        del trainer, dataset, task_loader
        self.agg_dataset = pd.DataFrame()

        self.model_copy.predictor.zero_states(1)
        self.model_copy.predictor._state_ids = []
        self.model_state = {}

        self.model = self.model_copy
        self.model_copy = deepcopy(self.model)

        self.task_start_time = time()

    def check_time(self, mode='inference'):
        prev = self.last_inference if mode == 'inference' else self.last_aggregation
        # print(prev)
        prev = time() if prev is None else prev
        frequency = self.inference_frequency if mode == 'inference' else self.aggregation_frequency
        now = time()
        if (now - prev) < (1 / frequency):
            # print("WARN : keeping up to rate")
            # print(now, prev, now-prev)
            sleep((1 / self.inference_frequency) - (time() - self.last_inference))
            rospy.sleep((1 / frequency) - (now - prev))
            # print(time() - prev)
        else:
            print("WARN : Not keeping up to rate")

        if mode == 'inference':
            # if self.last_inference: print(time() - self.last_inference)
            self.last_inference = time()
        else:
            self.last_aggregation = time()

    def manage_buffer_size(self, tracked_agents):
        buffer_size = self.input_horizon + 1
        self.history.append(tracked_agents)
        while len(self.history) > buffer_size:
            self.history.pop(0)

    def save_state_to_df(self):
        new_tracks = pd.DataFrame({
            'Time': [self.step * (1000/self.inference_frequency)] * len(self.tracked_agents),
            'Scene': len(self.model_copy.prev_tasks),
            'PedestrianId': [a['id'] for a in self.tracked_agents],
            'X': [a['state'][0] for a in self.tracked_agents],
            'Y': [a['state'][1] for a in self.tracked_agents],
            'Yaw': [a['state'][2] for a in self.tracked_agents],
            'Vx': [a['state'][3] for a in self.tracked_agents],
            'Vy': [a['state'][4] for a in self.tracked_agents]
        })
        self.agg_dataset = self.agg_dataset.append(new_tracks)
        self.step += 1

    def update_inference_buffer(self):
        self.history.append(deepcopy(self.tracked_agents))
        self.history.pop(0)

        if len(self.history) > self.model.predictor.input_horizon:
            raise BufferError

        while len(self.history) != self.model.predictor.input_horizon:
            self.history.append(deepcopy(self.tracked_agents))

    def update_model_state(self, new_ids):
        new_ids = [int(id) for id in new_ids]
        if 'state' not in self.model_state.keys():
            self.model.predictor.zero_states(len(new_ids))
            self.model_state['state'] = self.model.predictor.get_states_placeholder(len(new_ids))
            self.model_state['ids'] = new_ids
        else:
            new_model_state = self.model.predictor.get_states_placeholder(len(new_ids))

            # fill new state with existing states from recurring track ids
            for i, current_id in enumerate(new_ids):
                if current_id in self.model_state['ids']:
                    prev_index = self.model_state['ids'].index(current_id)
                    for j in range(len(new_model_state)):
                        new_model_state[j][0][0, i, :] = self.model_state['state'][j][0][0, prev_index, :]
                        new_model_state[j][1][0, i, :] = self.model_state['state'][j][1][0, prev_index, :]
            # self._state_ids = ids
            self.model_state['state'] = new_model_state
            self.model_state['ids'] = new_ids


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='StateDiffs')
    parser.add_argument('--save_name', type=str, default='eth-ucy_coreset')

    parser.add_argument('--task_time', type=int, default=200)
    parser.add_argument('--tbptt', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--inference_frequency', type=int, default=5)
    args = parser.parse_args()

    save_dir = f'saves/{args.model}/{args.save_name}'
    os.makedirs(f'{save_dir}/agg_data_dir/train', exist_ok=True)

    # ------------
    # model
    # ------------
    model = EwcPredictor.load_from_checkpoint(f'{save_dir}/final.ckpt')
    model.eval()

    # ------------
    # Online training
    # ------------
    rospy.init_node('predictor_node')
    ewc_node = EwcNode(model, save_dir, **vars(args))
    ewc_node.lock = threading.Lock()
    try:
        t1 = threading.Thread(target=ewc_node.inference, args=())
        t2 = threading.Thread(target=ewc_node.online_learning, args=())
        t1.start()
        t2.start()
        t1.join()
        t2.join()
    except:
        print('something went wrong when creating threads')


if __name__ == '__main__':
    cli_main()
