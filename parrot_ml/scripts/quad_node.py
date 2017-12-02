#! /usr/bin/env python
from quad_helper import QuadHelper
import rospy


def main():
    move_quad = QuadHelper()

    takeoff = False
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        if not takeoff:
            move_quad.takeoff()
            rospy.sleep(5.0)
            takeoff = True
            for i in range(5):
                move_quad.move_quad('FWD')
                rospy.sleep(0.2)
                print(move_quad.get_curr_state())
            move_quad.stop_quad()
        if takeoff:
            move_quad.land()
            print(move_quad.get_curr_state())
        rate.sleep()


if __name__ == '__main__':
    main()
