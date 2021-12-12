import pandas as pd
import numpy as np
import pickle as pkl

'''
Script to extract datasets of the correct format from different sources
'''


def interp_df(df, times=2):
    new_tracks = []
    for pid in df.PedestrianId.unique():
        track = df[df.PedestrianId == pid].values
        prev_entry = None
        for entry in track:
            if prev_entry is None or (entry[0]-prev_entry[0]) > 400:
                prev_entry = entry
                continue

            interp_entry = [
                (prev_entry[0] + entry[0]) / 2,
                pid,
                (prev_entry[2] + entry[2]) / 2,
                (prev_entry[3] + entry[3]) / 2,
                (prev_entry[4] + entry[4]) / 2,
                (prev_entry[5] + entry[5]) / 2,
                (prev_entry[6] + entry[6]) / 2,
                ]
            new_tracks.append(interp_entry)
            new_tracks.append(entry)
            prev_entry = entry

    interpDf = pd.DataFrame(new_tracks, columns=['Time', 'PedestrianId', 'X', 'Y', 'Yaw', 'Vx', 'Vy'])
    interpDf = interpDf.sort_values('Time')
    return interpDf


def biwi(raw_data_path):
    biwi_recording_fps = 2.5
    dataset = []
    with open(raw_data_path, 'r') as f:
        for line in f:
            line = line.replace('\n', '')
            entry = [float(e) for e in line.split(' ') if e != '']
            entry.append(np.arctan2(entry[-1], entry[-3]))
            dataset.append(entry)

    columns = ['Frame', 'PedestrianId', 'X', 'Z', 'Y', 'Vx', 'Vz', 'Vy', 'Yaw']
    df = pd.DataFrame(dataset, columns=columns)

    if 'eth' in raw_data_path:
        df['Time'] = df.Frame * (400/6)
    elif 'hotel' in raw_data_path:
        df['Time'] = df.Frame * (400/10) - 40
    else:
        raise RuntimeError("Datapath of raw file doesn't belong to biwi")

    df = df[['Time', 'PedestrianId', 'X', 'Y', 'Yaw', 'Vx', 'Vy']]
    interpDf = interp_df(df, 2)

    return interpDf


def crowds(raw_data_path):
    # TODO
    raise NotImplementedError


def gym_collision_avoidance(raw_data_path):

    raw_data = pkl.load(open(raw_data_path, 'rb'))
    new_data = []

    current_t = 0
    for s, i in enumerate(range(0, len(raw_data), 5)):
        part_track = raw_data[i]

        # if part_track[0]['time'] <= current_t:
        #     continue
        current_t = part_track[0]['time'] * 1000,

        track_pids = np.arange(s*6, (s*6)+6)

        for track_state in part_track:

            new_data.append([
                track_state['time']*1000,
                track_pids[0],
                track_state['pedestrian_state']['position'][0],
                track_state['pedestrian_state']['position'][1],
                np.arctan2(
                    track_state['pedestrian_state']['velocity'][1],
                    track_state['pedestrian_state']['velocity'][0],
                ),
                track_state['pedestrian_state']['velocity'][0],
                track_state['pedestrian_state']['velocity'][1],
            ])

            for p in range(len(track_state['other_agents_pos'])):
                # if p+1 == 1: continue
                new_data.append([
                    track_state['time']*1000,
                    track_pids[p+1],
                    track_state['other_agents_pos'][p][0],
                    track_state['other_agents_pos'][p][1],
                    np.arctan2(
                        track_state['other_agents_vel'][p][1],
                        track_state['other_agents_vel'][p][0],
                    ),
                    track_state['other_agents_vel'][p][0],
                    track_state['other_agents_vel'][p][1],
                ])

    return pd.DataFrame(new_data, columns=['Time', 'PedestrianId', 'X', 'Y', 'Yaw', 'Vx', 'Vy'])


def from_star(raw_data_path):
    """ Method to extract data in the right format, from data found in: https://github.com/Majiker/STAR/tree/master/data
    """

    with open(raw_data_path) as csv_file:
        test = csv_file.read()
    raw_data = pd.read_csv(raw_data_path, index_col=None).T
    raw_data.columns = ['PedestrianId', 'X', 'Y']
    raw_data['Time'] = raw_data.index.astype('float64')
    if 'eth' in raw_data_path and 'univ' in raw_data_path:
        raw_data.Time = raw_data.Time.apply(np.floor)*(400/6)  # to get the right time in ms (2.5hz / 400ms)
    else:
        raw_data.Time = raw_data.Time.apply(np.floor)*40  # to get the right time in ms (2.5hz / 400ms)

    raw_data = raw_data.reset_index(drop=True)

    all_tracks = np.array([])
    for p_id in raw_data.PedestrianId.unique():
        pedestrian_track = raw_data[raw_data.PedestrianId == p_id].sort_values('Time') #[['X','Y']].to_numpy()
        vx = np.diff(pedestrian_track.X.values)*2.5
        vy = np.diff(pedestrian_track.Y.values)*2.5
        yaw = np.arctan2(vy, vx)
        new_tracks = pedestrian_track[['Time', 'PedestrianId', 'X', 'Y']].to_numpy()[1:]
        if new_tracks.shape[0] > 0:
            new_tracks = np.concatenate((new_tracks, np.expand_dims(yaw, axis=1), np.expand_dims(vx, axis=1), np.expand_dims(vy, axis=1)), axis=1)
        else:
            continue

        if all_tracks.shape[0] > 0:
            all_tracks = np.vstack((all_tracks, new_tracks))
        else:
            all_tracks = new_tracks

    df = pd.DataFrame(all_tracks, columns=['Time', 'PedestrianId', 'X', 'Y', 'Yaw', 'Vx', 'Vy'])
    interpDf = interp_df(df, 2)
    interpDf['Scene'] = [0] * len(interpDf)

    return interpDf


def temp_conv_time(df):
    df = pd.read_csv(df)
    # df.Time = df.Time.apply(lambda x: int(x*1000))
    df.Time = df.Time.apply(lambda x: int(np.ceil(x / 100.0)) * 100)
    return df[['Scene','Time','PedestrianId','X','Y','Yaw','Vx','Vy']]

def add_scene(df):
    df = pd.read_csv(df)
    df['Scene'] = [0]*len(df.Time)
    return df[['Scene','Time','PedestrianId','X','Y','Yaw','Vx','Vy']]


if __name__ == '__main__':
    # Eth original
    # df = biwi('raw_data/biwi/seq_hotel/obsmat.txt')
    # df.to_csv('data/biwi/train/nomap_hotel.csv')

    # df = add_scene(f'data/pedsim/train/square_6-agents.csv')
    # df.to_csv(f'data/pedsim/train/square_6-agents.csv')
    # df = add_scene(f'data/pedsim/test/hallway_2-agents.csv')
    # df.to_csv(f'data/pedsim/test/hallway_2-agents.csv')
    # df = add_scene(f'data/pedsim/test/hallway-obstacle_2-agents.csv')
    # df.to_csv(f'data/pedsim/test/hallway-obstacle_2-agents.csv')

    # Gym collision avoidance
    # df = gym_collision_avoidance('raw_data/gym_collision_avoidance/RVO0.pkl')
    # df.to_csv('data/gym_collision_avoidance/train/nomap_RVO0.csv')
    for i in range(4):
        df = add_scene(f'data/gym_collision_avoidance/test/nomap_RVO{i}-corridor.csv')
        df.to_csv(f'data/gym_collision_avoidance/test/nomap_RVO{i}-corridor.csv')
        # df = temp_conv_time(f'data/gym_coop_6agents/train/nomap_RVO{i}.csv')
        # df.to_csv(f'data/gym_coop_6agents/train/nomap_RVO{i}.csv')
        # df = temp_conv_time(f'data/gym_corridor_10agents/train/nomap_RVO{i}.csv')
        # df.to_csv(f'data/gym_corridor_10agents/train/nomap_RVO{i}.csv')
        # df = temp_conv_time(f'data/gym_corridor/train/nomap_RVO{i}.csv')
        # df.to_csv(f'data/gym_corridor/train/nomap_RVO{i}.csv')
        # df = temp_conv_time(f'data/gym_random_swap/train/nomap_RVO{i}.csv')
        # df.to_csv(f'data/gym_random_swap/train/nomap_RVO{i}.csv')
        # df = temp_conv_time(f'data/gym_corridor_10agents/test/nomap_RVO{i}.csv')
        # df.to_csv(f'data/gym_corridor_10agents/test/nomap_RVO{i}.csv')
    # df = temp_conv_time('data/gym_collision_avoidance/train/RVO0.csv')
    # df.to_csv('data/gym_collision_avoidance/train/RVO0.csv')

    # UCY from STAR repo
    # df = from_star('raw_data/ucy_from_star/zara/zara01/true_pos_.csv')
    # df.to_csv('data/ucy/train/nomap_zara01.csv')
    # df = from_star('raw_data/ucy_from_star/zara/zara02/true_pos_.csv')
    # df.to_csv('data/ucy/train/nomap_zara02.csv')
    # df = from_star('raw_data/ucy_from_star/zara/zara03/true_pos_.csv')
    # df.to_csv('data/ucy/test/nomap_zara03.csv')
    # #
    # df = from_star('raw_data/ucy_from_star/univ/students001/true_pos_.csv')
    # df.to_csv('data/ucy/train/nomap_students01.csv')
    # df = from_star('raw_data/ucy_from_star/univ/students003/true_pos_.csv')
    # df.to_csv('data/ucy/test/nomap_students03.csv')
    #
    # # ETH from STAR repo
    # df = from_star('raw_data/eth_from_star/hotel/true_pos_.csv')
    # df.to_csv('data/eth/train/nomap_hotel.csv')
    # df = from_star('raw_data/eth_from_star/univ/true_pos_.csv')
    # df.to_csv('data/eth/test/nomap_univ.csv')
