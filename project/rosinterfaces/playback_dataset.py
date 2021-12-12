import numpy as np
import rospy
import csv
import pandas as pd
from pyquaternion import Quaternion
from time import time
from spencer_tracking_msgs.msg import TrackedPersons, TrackedPerson

'''
Script to playback saved datasets over ROS. Usefull for quickly visualizing.
'''

def new_track(state):
    p_id, x, y, vx, vy = state
    res = TrackedPerson()
    res.track_id = int(p_id)
    res.pose.pose.position.x = x
    res.pose.pose.position.y = y
    res.twist.twist.linear.x = vx
    res.twist.twist.linear.y = vy
    q = Quaternion(axis=[0, 0, 1], angle=np.arctan2(vy, (vx + 1e-9)))
    if vx != 0:
        res.pose.pose.orientation.x = q.x
        res.pose.pose.orientation.y = q.y
        res.pose.pose.orientation.w = q.w
        res.pose.pose.orientation.z = q.z
    return res


def _load_tsv(f, frequency=10):
    res = []
    read_tsv = csv.reader(open(f), delimiter="\t")

    for row in read_tsv:
        try:
            frame = int(row[0])
        except:
            continue

        if frame % (100 / frequency) != 0:  # 100hz -> 10hz
            continue

        # Extract [frame, pedestrian, x, y] for all 10 helmets of the THOR dataset
        for i in range(1, 11):
            if float(row[2 + i * 17]) == 0 and float(row[3 + i * 17]) == 0:
                continue
            res.append([int(frame), int(i), float(row[2 + i * 17])/1000, float(row[3 + i * 17])/1000])

    res = np.array(res)

    return res


def thor():
    rospy.init_node('play_back')
    pub = rospy.Publisher('/pedsim_visualizer/tracked_persons', TrackedPersons, queue_size=10)
    frequency = 20
    data = _load_tsv('../../data/thor/train/Exp_3_run_1_6D.tsv', frequency)
    frames = np.unique(data[:, 0])
    prev_state = None

    for frame in frames:
        start_time = time()
        msg = TrackedPersons()
        msg.header.frame_id = 'map'
        state = data[data[:, 0] == frame]
        if prev_state is not None:
            for p in state:
                p_prev = prev_state[prev_state[:, 1] == p[1]]
                if len(p_prev) > 0:
                    p_prev = p_prev[0]
                    v = -(p_prev[2:] - p[2:]) * frequency
                    p_track = new_track([p[1], p[2], p[3], v[0], v[1]])
                    msg.tracks.append(p_track)
        prev_state = state
        rospy.sleep((1 / frequency) - (time() - start_time))
        pub.publish(msg)

def play_df(path):
    rospy.init_node('play_back')
    pub = rospy.Publisher('/tracking3D', TrackedPersons, queue_size=10)

    speedup = 1
    data = pd.read_csv(path)
    tracks_frequencies = np.abs(np.diff(data.Time.values))
    frequency = int(1000 / np.min(tracks_frequencies[tracks_frequencies != 0]))

    # TODO: fix Time.value_counts of 12 (should be 6)

    # Remove track_id 1 because it's the mpc and does weird stuff (I think).

    for scene in data.Scene.unique():
        scene_data = data[data.Scene == scene]
        start_time = min(scene_data.Time.values)
        end_time = max(scene_data.Time.values)
        unique_times = scene_data.Time.unique()
        for current_time in unique_times:
            msg = TrackedPersons()
            msg.header.frame_id = 'map'
            pedestrian_states = scene_data[scene_data.Time == current_time][['PedestrianId', 'X', 'Y', 'Vx', 'Vy']].to_numpy()
            for p_state in pedestrian_states:
                p_track = new_track(p_state)
                msg.tracks.append(p_track)
            rospy.sleep((1 / frequency) / speedup)
            pub.publish(msg)


# def play_df(path):
#     rospy.init_node('play_back')
#     pub = rospy.Publisher('/tracked_persons', TrackedPersons, queue_size=10)
#
#     speedup = 1
#     data = pd.read_csv(path)
#     tracks_frequencies = np.abs(np.diff(data.Time.values))
#     frequency = int(1000 / np.min(tracks_frequencies[tracks_frequencies != 0]))
#
#     # TODO: fix Time.value_counts of 12 (should be 6)
#
#     # Remove track_id 1 because it's the mpc and does weird stuff (I think).
#
#     start_time = min(data.Time.values)
#     end_time = max(data.Time.values)
#
#     for current_time in range(int(start_time), int(end_time), int(1000/frequency)):
#         msg = TrackedPersons()
#         msg.header.frame_id = 'map'
#         pedestrian_states = data[data.Time == current_time][['PedestrianId', 'X', 'Y', 'Vx', 'Vy']].to_numpy()
#         for p_state in pedestrian_states:
#             p_track = new_track(p_state)
#             msg.tracks.append(p_track)
#         rospy.sleep((1 / frequency) / speedup)
#         pub.publish(msg)


if __name__ == "__main__":
    # thor()
    # play_df('data/gym_corridor/train/nomap_RVO1.csv')
    # play_df('data/gym_random_swap/train/nomap_RVO0.csv')
    # play_df('data/eth/test/nomap_univ.csv')
    # play_df('data/ucy/test/nomap_zara03.csv')

    # play_df('data/tudelft/train/mainentrance_test.csv')
    # play_df('data/pedsim/train/square_6-agents.csv')
    # play_df('data/pedsim/test/hallway_2-agents.csv')
    # play_df('data/pedsim/test/hallway-obstacle_2-agents.csv')
    # play_df('data/rvo2/train/obstacles_4-agents.csv')
    # play_df('data/rvo2/test/coop_4-agents.csv')

    # play_df('data/gym_corridor/train/nomap_RVO0.csv')
    # play_df('data/gym_collision_avoidance/train/nomap_RVO1.csv')
    play_df('data/gym_corridor/train/corridor_RVO0.csv')
    # play_df('data/gym_random_swap/train/nomap_RVO0.csv')
    # play_df('data/gym_collision_avoidance/train/nomap_RVO2.csv')
    # play_df('data/gym_collision_avoidance/train/nomap_RVO3.csv')
    # play_df('data/gym_collision_avoidance/train/nomap_RVO4.csv')
    # play_df('data/gym_collision_avoidance/test/nomap_RVO5.csv')
    # play_df('data/gym_collision_avoidance/test/nomap_RVO6.csv')
