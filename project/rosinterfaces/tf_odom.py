import rospy
from tf import TransformListener
import numpy as np
from pyquaternion import Quaternion
from nav_msgs.msg import Odometry
from copy import deepcopy

def yaw_from_quaternion(q):
    """Convert from quaternion to yaw.

    Args:
        q (object): Quaternion object.

    Returns:
        float: Yaw angle in radians.
    """
    yaw = np.arctan2(2.0 * (q.y * q.z + q.w * q.x), q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z)
    pitch = np.arcsin(-2.0 * (q.x * q.z - q.w * q.y))
    roll = np.arctan2(2.0 * (q.x * q.y + q.w * q.z), q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z)
    return roll

def quaternion_from_yaw(y):
    return Quaternion(axis=[0, 0, 1], angle=y)


def yaw_from_quaternion_list(q):
    """Convert from quaternion to yaw.
    Args:
        q (object): Quaternion object.
    Returns:
        float: Yaw angle in radians.
    """
    roll = np.arctan2(2.0 * (q[0] * q[1] + q[3] * q[2]), q[3] * q[3] + q[0] * q[0] - q[1] * q[1] - q[2] * q[2])
    return roll


def matrix_from_angle(current_angle):
    """Return a matrix representing the current angle in radians .
    Args:
        current_angle (float): angle in radians.
    Returns:
        np.Array(): Rotation matrix
    """
    c, s = np.cos(current_angle), np.sin(current_angle)
    return np.array(((c, -s), (s, c)))


def quaternion_vector(q):
    return Quaternion(w=q.w, x=q.x, y=q.y, z=q.z)


class ConvertDetections:
    def __init__(self):
        self.previous = None
        self.last_t = None

        self.tf = TransformListener()
        rospy.sleep(3)
        rospy.Subscriber('/odometry/filtered', Odometry, self.callback, queue_size=1)
        self.pub = rospy.Publisher('/amcl/odometry/filtered', Odometry, queue_size=10)

    def callback(self, msg):
        print('yes')
        old_frame = msg.header.frame_id
        print(old_frame)
        new_frame = 'map'
        t = self.tf.getLatestCommonTime(new_frame, old_frame)
        position, quaternion_diff_list = self.tf.lookupTransform(new_frame, old_frame, t)

        yaw_diff = yaw_from_quaternion_list(quaternion_diff_list)
        quaternion_diff = Quaternion(w=quaternion_diff_list[3], x=quaternion_diff_list[0], y=quaternion_diff_list[1], z=quaternion_diff_list[2])
        R = matrix_from_angle(yaw_diff)

        quaternion_original = msg.pose.pose.orientation
        yaw_original = yaw_from_quaternion(quaternion_original)

        new_msg = deepcopy(msg)
        new_msg.header.frame_id = 'map'

        # translate position
        [new_msg.pose.pose.position.x, new_msg.pose.pose.position.y] = R.dot(
            np.array([msg.pose.pose.position.x, msg.pose.pose.position.y]))
        new_msg.pose.pose.position.x += position[0]
        new_msg.pose.pose.position.y += position[1]
        new_msg.pose.pose.position.z += position[2]

        # rotate heading
        # q_original = quaternion_vector(msg.pose.pose.orientation)
        new_yaw = yaw_original + yaw_diff
        # q_new = q_original + quaternion
        q_new = quaternion_from_yaw(new_yaw)
        new_msg.pose.pose.orientation.x = q_new.x
        new_msg.pose.pose.orientation.y = q_new.y
        new_msg.pose.pose.orientation.z = q_new.z
        new_msg.pose.pose.orientation.w = q_new.w

        # rotate velocity
        new_msg.twist.twist.linear.x, new_msg.twist.twist.linear.y = R.dot([msg.twist.twist.linear.x, msg.twist.twist.linear.y])

        self.pub.publish(new_msg)


if __name__ == '__main__':
    rospy.init_node('convert_detections')
    ConvertDetections()
    rospy.spin()