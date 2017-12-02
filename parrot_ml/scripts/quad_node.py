#! /usr/bin/env python
from quad_helper import QuadHelper
from pytorch_helper import QuadDQN
import numpy as np
import rospy


def main():
    control_quad = QuadHelper()
    dqn_quad = QuadDQN()

    takeoff = False
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        if not takeoff:
            control_quad.takeoff()
            rospy.sleep(5.0)
            takeoff = True
            print('\n')
            for i in range(1000):
                print("Iteration %d" % i)
                # Get current state
                print("Getting current state")
                curr_state = np.array(control_quad.get_curr_state(), dtype=np.float32)
                print(curr_state)
                # Get action q_values
                print("Getting predicted q_values")
                pred_q = dqn_quad.predict_action(curr_state)
                # Get action with max q_value
                print("Getting best action")
                max_q_idx = np.argmax(pred_q)
                max_q = np.amax(pred_q)
                # Do action
                print("Moving quadcopter")
                control_quad.move_quad(dqn_quad.do_action(max_q_idx))
                # Get new state
                print("Getting new state")
                new_state = control_quad.get_curr_state()
                print(new_state)
                # Get reward
                print("Getting reward")
                reward = dqn_quad.get_reward(curr_state, new_state)
                # Set target q_values for backprop
                print("Setting target values")
                target_q = pred_q
                target_q[max_q_idx] = reward + dqn_quad.gamma * max_q
                print(target_q)
                print("Computing loss")
                dqn_quad.get_loss(target_q, pred_q)
                # Do backprop
                print("Backpropagation")
                dqn_quad.backprop()
                print('\n')

        if takeoff:
            control_quad.land()
            print(control_quad.get_curr_state())
        rate.sleep()


if __name__ == '__main__':
    main()
