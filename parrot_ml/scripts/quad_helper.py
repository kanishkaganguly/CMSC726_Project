#! /usr/bin/env python
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty
import tf
import rospy


class QuadHelper(object):
    def __init__(self):
        self.curr_state = [0.0, 0.0, 0.0, 0.0]
        self.target_state = [0.5, 0.0, 0.5, 0.0]

        self.vel_msg = Twist()
        rospy.init_node('bebop_move', anonymous=True)

        self.mover_pub = rospy.Publisher('/bebop/cmd_vel', Twist, queue_size=10)
        self.takeoff_pub = rospy.Publisher('/bebop/takeoff', Empty, queue_size=10)
        self.land_pub = rospy.Publisher('/bebop/land', Empty, queue_size=10)

        self.vicon_sub()
        print("Ready to fly...")

    def move_quad(self, direction):

        if direction == "FWD":
            self.vel_msg.linear.x = 0.3
            self.vel_msg.linear.y = 0.0
            self.vel_msg.linear.z = 0.0
            self.vel_msg.angular.x = 0.0
            self.vel_msg.angular.y = 0.0
            self.vel_msg.angular.z = 0.0
        elif direction == "BCK":
            self.vel_msg.linear.x = -0.3
            self.vel_msg.linear.y = 0.0
            self.vel_msg.linear.z = 0.0
            self.vel_msg.angular.x = 0.0
            self.vel_msg.angular.y = 0.0
            self.vel_msg.angular.z = 0.0
        elif direction == "LFT":
            self.vel_msg.linear.x = 0.0
            self.vel_msg.linear.y = 0.3
            self.vel_msg.linear.z = 0.0
            self.vel_msg.angular.x = 0.0
            self.vel_msg.angular.y = 0.0
            self.vel_msg.angular.z = 0.0
        elif direction == "RGT":
            self.vel_msg.linear.x = 0.0
            self.vel_msg.linear.y = -0.3
            self.vel_msg.linear.z = 0.0
            self.vel_msg.angular.x = 0.0
            self.vel_msg.angular.y = 0.0
            self.vel_msg.angular.z = 0.0
        elif direction == "UP":
            self.vel_msg.linear.x = 0.0
            self.vel_msg.linear.y = 0.0
            self.vel_msg.linear.z = 0.3
            self.vel_msg.angular.x = 0.0
            self.vel_msg.angular.y = 0.0
            self.vel_msg.angular.z = 0.0
        elif direction == "DWN":
            self.vel_msg.linear.x = 0.0
            self.vel_msg.linear.y = 0.0
            self.vel_msg.linear.z = -0.3
            self.vel_msg.angular.x = 0.0
            self.vel_msg.angular.y = 0.0
            self.vel_msg.angular.z = 0.0
        elif direction == "ROT_CW":
            self.vel_msg.linear.x = 0.0
            self.vel_msg.linear.y = 0.0
            self.vel_msg.linear.z = 0.0
            self.vel_msg.angular.x = 0.0
            self.vel_msg.angular.y = 0.0
            self.vel_msg.angular.z = -0.3
        elif direction == "ROT_CCW":
            self.vel_msg.linear.x = 0.0
            self.vel_msg.linear.y = 0.0
            self.vel_msg.linear.z = 0.0
            self.vel_msg.angular.x = 0.0
            self.vel_msg.angular.y = 0.0
            self.vel_msg.angular.z = 0.3
        else:
            self.vel_msg.linear.x = 0.0
            self.vel_msg.linear.y = 0.0
            self.vel_msg.linear.z = 0.0
            self.vel_msg.angular.x = 0.0
            self.vel_msg.angular.y = 0.0
            self.vel_msg.angular.z = 0.0

        print("Moving %s" % direction)
        while self.mover_pub.get_num_connections() == 0:
            pass
        self.mover_pub.publish(self.vel_msg)

    def stop_quad(self):
        self.vel_msg.linear.x = 0.0
        self.vel_msg.linear.y = 0.0
        self.vel_msg.linear.z = 0.0
        self.vel_msg.angular.x = 0.0
        self.vel_msg.angular.y = 0.0
        self.vel_msg.angular.z = 0.0

        print("Stopping quad")
        while self.mover_pub.get_num_connections() == 0:
            pass

        self.mover_pub.publish(self.vel_msg)
        return

    def takeoff(self):
        print("Taking off...")
        while self.takeoff_pub.get_num_connections() == 0:
            pass
        self.takeoff_pub.publish()
        return

    def land(self):
        print("Landing...")
        while self.land_pub.get_num_connections() == 0:
            pass
        self.land_pub.publish()
        return

    def vicon_sub(self):
        rospy.Subscriber('/parrot/odom', Odometry, self.vicon_callback)

    def vicon_callback(self, data):
        pos = data.pose.pose.position
        eul = tf.transformations.euler_from_quaternion(
            [data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z,
             data.pose.pose.orientation.w])
        self.curr_state = [pos.x, pos.y, pos.z, eul[2]]

    def get_curr_state(self):
        return self.curr_state
