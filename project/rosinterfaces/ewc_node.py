import rospy
import pandas as pd
import torch
from argparse import ArgumentParser
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from time import time
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
        self.save_dir = save_dir
        self.inference_frequency = inference_frequency
        self.prediction_frequency = 5#self.model.predictor.frequency

        # Aggregation stuff
        self.last_time = time()
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
        assert(self.inference_frequency == self.prediction_frequency)

    def run(self):
        self.task_start_time = time()
        self.last_time = time()
        while not rospy.is_shutdown():
            # ------------
            # time keeping
            # ------------
            self.check_time()

            # ------------
            # save to agg DF
            # ------------
            self.save_state_to_df()

            # ------------
            # fill buffer for inference (input_horizon)
            # ------------
            self.update_inference_buffer()

            # ------------
            # inference
            # ------------
            with torch.no_grad():
                # pre process current state
                inputs = self.model.predictor.extract_batch_inputs_from_ros(self.history, self.map_state)
                if (inputs['ego_input'].size()[-1] != 2): continue

                # run inference on current state
                predictions = self.model(inputs)

                # post process predictions
                trajectories = self.model.predictor.map_predictions_to_world(self.history[-1], predictions)

            # publish visualization msg of trajectories to ros
            self.visualize_trajectories(self.history[-1], trajectories[:, :, :2])

            # publish msg of trajectories to ros for motion planning
            self.visualize_trajectories(self.history[-1], trajectories[:, :, :2])

            # train on current_task and refresh task every 120 secs for 10 epochs
            if time() - self.task_start_time > self.task_time:
                # ------------
                # Train model online
                # ------------
                self.train_online()

    def train_online(self):
        # ------------
        # Extract examples
        # ------------
        self.agg_dataset.to_csv(f'{self.save_dir}/agg_data_dir/train/task_{len(self.model.prev_tasks)}.csv')
        dataset = TrajpredDataset(
            self.model.predictor,
            tbptt=self.tbptt,
            stride=self.stride,  # delta t between examples, if lower than 'tbptt + pred_horizon', examples can overlap.
            min_track_length=0.5)
        dataset._extract_from_experiment_df(
            self.model.predictor,
            tracks=self.agg_dataset,
            map_state=self.map_state,
            experiment_name='ewc'
        )
        task_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        print(f"Train task {len(self.model.prev_tasks)} containing {len(task_loader.dataset)} examples")

        # ------------
        # Combine with coreset
        # ------------
        task_loader = self.model.add_coreset_to_loader(task_loader)

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
        trainer.fit(self.model, task_loader)
        self.model.register_ewc_params(trainer.train_dataloader)

        # ------------
        # saving
        # ------------
        ckpt_name = f'{self.save_dir}/task_{len(self.model.prev_tasks) - 1}.ckpt'
        trainer.save_checkpoint(ckpt_name)
        self.previous_ckpt = ckpt_name

        # ------------
        # update coreset
        # ------------
        self.model.update_coreset(task_loader)

        # ------------
        # clear from memory and reset states
        # ------------
        del trainer, dataset, task_loader
        self.agg_dataset = pd.DataFrame()

        self.model.predictor.zero_states(1)
        self.model.predictor.prev_ids = []
        self.task_start_time = time()

    def check_time(self):
        prev = self.last_time
        now = time()
        if (now - prev) < (1 / self.inference_frequency):
            rospy.sleep((1 / self.inference_frequency) - (now - prev))
        else:
            print("WARN : Not keeping up to rate")

        self.last_time = time()

    def manage_buffer_size(self, tracked_agents):
        buffer_size = self.input_horizon + 1
        self.history.append(tracked_agents)
        while len(self.history) > buffer_size:
            self.history.pop(0)

    def save_state_to_df(self):
        new_tracks = pd.DataFrame({
            'Time': [self.step * (1000/self.inference_frequency)] * len(self.tracked_agents),
            'Scene': len(self.model.prev_tasks),
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


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='StateDiffs')
    parser.add_argument('--save_name', type=str, default='eth-ucy')

    parser.add_argument('--task_time', type=int, default=100)
    parser.add_argument('--tbptt', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--stride', type=int, default=2)
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
    ewc_node.run()


if __name__ == '__main__':
    cli_main()
