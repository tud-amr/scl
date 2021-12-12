import rospy
from spencer_tracking_msgs.msg import TrackedPersons
import numpy as np
from pyquaternion import Quaternion


def quaternion_from_yaw(y):
    return Quaternion(axis=[0, 0, 1], angle=y)


def yaw_from_quaternion(q):
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


class SimpleVel:
    def __init__(self):
        self.previous = None
        self.last_t = None

        rospy.sleep(3)
        rospy.Subscriber('/tracking3D', TrackedPersons, self.callback, queue_size=1)
        self.pub = rospy.Publisher('/tracking3D_vel', TrackedPersons, queue_size=10)

    def callback(self, msg):
        for p in msg.tracks:
            vx = p.twist.twist.linear.x
            vy = p.twist.twist.linear.y
            q = Quaternion(axis=[0, 0, 1], angle=np.arctan2(vy, (vx + 1e-9)))
            if vx != 0:
                p.pose.pose.orientation.x = q.x
                p.pose.pose.orientation.y = q.y
                p.pose.pose.orientation.w = q.w
                p.pose.pose.orientation.z = q.z
            p.twist.twist.linear.x *= 8
            p.twist.twist.linear.y *= 8

        self.pub.publish(msg)


if __name__ == '__main__':
    rospy.init_node('simple_velocity')
    SimpleVel()
    rospy.spin()