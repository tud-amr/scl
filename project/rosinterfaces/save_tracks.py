import pandas as pd
import rospy
from time import time
from spencer_tracking_msgs.msg import TrackedPersons
import numpy as np

'''
Script that Extracts a dataset in the correct format from tracked persons using ROS.
'''

class TrackSaver:
    def __init__(self, frequency, run_time, save_name):
        self.current_time_ms = 0
        self.save_name = save_name
        self.run_time = run_time
        self.frequency = frequency
        self.tracked_agents = None
        self.tracks = pd.DataFrame()
        rospy.Subscriber('/tracking3D_vel', TrackedPersons, self.peds_state_cb, queue_size=1)
        rospy.sleep(1)
        self.start_time = time()

    def peds_state_cb(self, msg):
        self.tracked_agents = msg.tracks

    def run(self):
        while not rospy.is_shutdown():
            loop_time = time()
            new_tracks = pd.DataFrame({
                'Time': [self.current_time_ms]*len(self.tracked_agents),
                'Scene': [0]*len(self.tracked_agents),
                'PedestrianId': [a.track_id for a in self.tracked_agents],
                'X': [a.pose.pose.position.x for a in self.tracked_agents],
                'Y': [a.pose.pose.position.y for a in self.tracked_agents],
                # 'Yaw': [yaw_from_quaternion(a.pose.pose.orientation) for a in self.tracked_agents],
                'Yaw': [np.arctan2(a.twist.twist.linear.y, a.twist.twist.linear.x) for a in self.tracked_agents],
                'Vx': [a.twist.twist.linear.x for a in self.tracked_agents],
                'Vy': [a.twist.twist.linear.y for a in self.tracked_agents],
            })

            self.tracks = self.tracks.append(new_tracks)
            self.current_time_ms += (1000/self.frequency)

            if (time() - self.start_time) > self.run_time:
                print('saving')
                self.save()
                return

            print(len(self.tracks))
            now = time()
            if (now - loop_time) < (1 / self.frequency):
                rospy.sleep((1 / self.frequency) - (now - loop_time))
            else:
                print("WARN : Not keeping up to rate")

    def save(self):
        self.tracks.to_csv(self.save_name+'.csv')


if __name__ == '__main__':
    rospy.init_node('track_saver_node')
    ewc_node = TrackSaver(frequency=5, run_time=90, save_name='../../data/tudelft/train/cyberzoo-square')
    ewc_node.run()
