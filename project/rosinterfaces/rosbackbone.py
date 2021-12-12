import rospy
import numpy as np
from geometry_msgs.msg import Pose
from spencer_tracking_msgs.msg import TrackedPersons, Trajectories, Trajectory
from nav_msgs.msg import OccupancyGrid, Odometry
from visualization_msgs.msg import MarkerArray, Marker
from pyquaternion import Quaternion
from project.utils.tools import yaw_from_quaternion


class RosBackbone:
    """
    Class containing interfaces with ros (publisher and subscribers).
    """
    def __init__(self):
        self.tracked_agents = []
        self.map_state = None

        # ROS subscribers
        # self.peds_state_topic = rospy.get_param('/other_agents_topic', "/pedsim_visualizer/tracked_persons")
        self.peds_state_topic = '/pedsim_visualizer/tracked_persons'
        # self.peds_state_topic = '/tracked_persons'
        # self.peds_state_topic = '/tracking3D_vel'
        self.grid_topic = rospy.get_param('/grid_topic', "/map")
        rospy.Subscriber(self.peds_state_topic, TrackedPersons, self.peds_state_cb, queue_size=1)
        rospy.Subscriber(self.grid_topic, OccupancyGrid, self.map_state_cb, queue_size=1)

        # ROS Publisher(s)
        self.prediction_viz_topic = rospy.get_param('/prediction_topic', "predictions/visualizations")
        self.prediction_topic = rospy.get_param('/prediction_topic', "~predictions")
        self.pub_viz = rospy.Publisher(self.prediction_viz_topic, MarkerArray, queue_size=10)
        self.pub_trajs = rospy.Publisher(self.prediction_topic, Trajectories, queue_size=10)

        # Colors
        self.colors = []
        self.colors.append([0.8500, 0.3250, 0.0980])  # orange
        self.colors.append([0.0, 0.4470, 0.7410])  # blue
        self.colors.append([0.4660, 0.6740, 0.1880])  # green
        self.colors.append([0.4940, 0.1840, 0.5560])  # purple
        self.colors.append([0.9290, 0.6940, 0.1250])  # yellow
        self.colors.append([0.3010, 0.7450, 0.9330])  # cyan
        self.colors.append([0.6350, 0.0780, 0.1840])  # chocolate
        self.colors.append([1, 0.6, 1])  # pink
        self.colors.append([0.505, 0.505, 0.505])  # grey

    def peds_state_cb(self, msg):
        """
        Update state of the currently tracked agents (pedestrians and the robot)

        Args:
            msg: TrackedPersons (spencer_msgs)
        """
        self.tracked_agents = []
        for p in msg.tracks:
            # if p.track_id == 0: continue
            self.tracked_agents.append({
                'state': [
                    p.pose.pose.position.x,
                    p.pose.pose.position.y,
                    # yaw_from_quaternion(p.pose.pose.orientation),
                    np.arctan2(
                        p.twist.twist.linear.y,
                        p.twist.twist.linear.x
                    ),
                    p.twist.twist.linear.x,
                    p.twist.twist.linear.y
                ],
                'id': p.track_id
            })

    def map_state_cb(self, msg):
        """
        Updates state of the map

        Args:
            msg: Occupancy grid of map state
        """
        res = {}
        msg.data = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        res['image'] = msg.data
        res['origin'] = [
            np.array(msg.info.origin.position.x),
            np.array(msg.info.origin.position.y),
            0 ]
        res['resolution'] = msg.info.resolution

        self.map_state = res

    def visualize_trajectories(self, tracked_agents, predictions):
        """
        Publish markers to visualize trajectories

        Args:
            tracked_agents:
            predictions:

        Returns:

        """
        global_trajectory = MarkerArray()
        m_id = 0
        for agent, prediction in zip(tracked_agents, predictions):
            i = agent['id'] % 9
            for t, pos in enumerate(prediction):
                marker = Marker()
                marker.header.frame_id = "map"
                # marker.header.frame_id = "odom"
                marker.header.stamp = rospy.Time(0)
                marker.ns = "goal_marker"
                marker.type = 3
                marker.scale.x = 0.3 * 2.0
                marker.scale.y = 0.3 * 2.0
                marker.scale.z = 0.1
                marker.color.a = 0.0
                marker.id = m_id #int(str(agent['id'])+str(t))
                marker.lifetime = rospy.Duration(0, int(20*10e6))
                pose = Pose()
                pose.position.x = pos[0]
                pose.position.y = pos[1]
                marker.color.a = 0.8 * ((len(prediction) - t) / len(prediction))
                marker.color.r = self.colors[i][0]
                marker.color.g = self.colors[i][1]
                marker.color.b = self.colors[i][2]
                pose.orientation.w = 1.0
                marker.pose = pose
                global_trajectory.markers.append(marker)
                m_id+=1

        self.pub_viz.publish(global_trajectory)

    def publish_trajectories(self, tracked_agents, predictions):
        """
        Publish trajectories

        Args:
            tracked_agents:
            predictions:

        Returns:

        """
        result = Trajectories()
        for agent, prediction in zip(tracked_agents, predictions):
            agent_traj = Trajectory()
            agent_traj.track_id = agent['id']
            agent_traj.frequency = 5

            for t, state in enumerate(prediction):
                odom = Odometry()
                odom.pose.pose.position.x = state[0]
                odom.pose.pose.position.y = state[1]
                vx, vy = state[2:4]
                q = Quaternion(axis=[0, 0, 1], angle=np.arctan2(vy, (vx + 1e-9)))
                if vx != 0:
                    odom.pose.pose.orientation.x = q.x
                    odom.pose.pose.orientation.y = q.y
                    odom.pose.pose.orientation.w = q.w
                    odom.pose.pose.orientation.z = q.z
                odom.twist.twist.linear.x = vx
                odom.twist.twist.linear.y = vy
                odom.pose.covariance = (np.eye(6) * 0.6).flatten() # using radius of 0.6m of pedestrian as unit covariance
                agent_traj.data.append(odom)
            result.trajectories.append(agent_traj)

        self.pub_trajs.publish(result)


