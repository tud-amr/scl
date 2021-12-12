import rospy
import torch
from argparse import ArgumentParser
import pytorch_lightning as pl
from project.rosinterfaces.rosbackbone import RosBackbone
from time import time
from copy import deepcopy
from project.training_routines.ewc import EwcPredictor

from project.datatools.trajpred_dataset import TrajpredDataset
from torch.utils.data import DataLoader


class PredictorNode(RosBackbone):
    def __init__(self, frequency: float = 20.0, **kwargs):
        RosBackbone.__init__(self)
        self.frequency = frequency
        rospy.sleep(1)
        self.history = []

    def run(self, model):
        prev = time()
        while not rospy.is_shutdown():
            now = time()
            if (now - prev) < (1 / self.frequency):
                rospy.sleep((1 / self.frequency) - (now - prev))
            else:
                print("WARN : Not keeping up to rate")
            prev = time()

            # Add tracks to DataFrame of past tracks
            self.history.append(deepcopy(self.tracked_agents))
            if len(self.history) < model.predictor.input_horizon:
                continue

            while len(self.history) > model.predictor.input_horizon:
                self.history.pop(0)

            if len(self.tracked_agents) == 0:
                continue

            with torch.no_grad():
                # pre process current state
                inputs = model.predictor.extract_batch_inputs_from_ros(self.history, self.map_state)
                # if (inputs['ego_input'].size()[-1] != 2): continue

                # run inference on current state
                predictions = model(inputs)

                # post process predictions
                trajectories = model.predictor.map_predictions_to_world(self.history[-1], predictions)

            # publish visualization msg of trajectories to ros
            self.visualize_trajectories(self.history[-1], trajectories[:, :, :2])

            # publish msg of trajectories to ros for motion planning
            self.publish_trajectories(self.history[-1], trajectories)


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='EthPredictor')
    parser.add_argument('--save_name', type=str, default='default')
    parser.add_argument('--frequency', type=int, default=15)
    args = parser.parse_args()

    # ------------
    # model
    # ------------
    model = EwcPredictor.load_from_checkpoint(f'saves/{args.model}/{args.save_name}/final.ckpt')
    # model = EwcPredictor.load_from_checkpoint(f'saves/{args.model}/{args.save}/epoch=15-step=495-val_ade=0.22.ckpt')
    model.eval()

    # ------------
    # prediction
    # ------------
    rospy.init_node('predictor_node')
    ewc_node = PredictorNode(**vars(args))
    ewc_node.run(model)


if __name__ == '__main__':
    cli_main()
