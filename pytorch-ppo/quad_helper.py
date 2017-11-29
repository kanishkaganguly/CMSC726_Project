#!/usr/bin/python3

import math

import numpy as np
from scipy.stats import norm

import vrep_rotors
import vrep_state


class QuadHelper(object):
    def __init__(self, clientID, quadHandle, targetHandle):
        self.clientID = clientID
        self.quadHandle = quadHandle
        self.targetHandle = targetHandle
        self.reset_pos = [0, 0, 0]
        self.reset_euler = [0, 0, 0]
        self.reset_rotor = [0.0, 0.0, 0.0, 0.0]

    '''
    Initialize and reset quadcopter position in world
    '''

    def init_quad(self):
        vrep_rotors.init_rotors(self.clientID)
        vrep_state.init_state(self.clientID, self.quadHandle)
        return

    '''
    Apply thrust to rotors
    '''

    def apply_rotor_thrust(self, thrusts):
        if any(i >= 9.0 for i in thrusts):
            thrusts = [9.0, 9.0, 9.0, 9.0]
        if any(i <= -9.0 for i in thrusts):
            thrusts = [-9.0, -9.0, -9.0, -9.0]
        vrep_rotors.set_rotors(self.clientID, thrusts)
        print("Actual Rotor Thrusts: " + str(thrusts))
        return

    '''
    Fetch current state
    '''

    def fetch_quad_state(self):
        pos = vrep_state.get_quad_pos(self.clientID, self.quadHandle)
        euler = vrep_state.get_quad_orientation(self.clientID, self.quadHandle)
        return pos, euler

    def fetch_target_state(self):
        pos = vrep_state.get_quad_pos(self.clientID, self.targetHandle)
        euler = vrep_state.get_quad_orientation(self.clientID, self.targetHandle)
        return pos, euler

    def set_target(self, pos, euler, rotor_data):
        vrep_state.set_pos_and_euler(self.clientID, self.targetHandle, pos=pos, euler=euler, rotor_data=rotor_data)

    def set_quad_pos(self, pos, euler, rotor_data):
        vrep_state.set_pos_and_euler(self.clientID, self.quadHandle, pos=pos, euler=euler, rotor_data=rotor_data)

    '''
    This function returns reward based on current state and target state (x,y,z,yaw,pitch,roll)
    '''

    def get_reward(self, curr_thrusts, curr_pos, curr_euler, target_pos, target_euler):

        gaussian = norm(0, 2)

        deviation_x = np.linalg.norm(curr_pos[0] - target_pos[0])
        deviation_y = np.linalg.norm(curr_pos[1] - target_pos[1])
        deviation_z = np.linalg.norm(curr_pos[2] - target_pos[2])
        if deviation_x > 1.5 or deviation_y > 1.5 or deviation_z > 1.5:
            print("Quadrotor Flew Too Far. Bad Quadrotor.")
            pos_reward = -100.0
        else:
            print("Quadrotor Position Within Bounds. Good Quadrotor!")
            sigma_x = 0.1
            sigma_y = 0.1
            sigma_z = 0.01
            reward_x = math.exp(-deviation_x ** 2 / (2 * sigma_x))
            reward_y = math.exp(-deviation_y ** 2 / (2 * sigma_y))
            reward_z = math.exp(-deviation_z ** 2 / (2 * sigma_z))
            pos_reward = self.sigmoid(reward_x + reward_y + reward_z)
        print("Position Reward: %f" % pos_reward)

        deviation_yaw = np.linalg.norm(curr_euler[0] - target_euler[0])
        deviation_pitch = np.linalg.norm(curr_euler[1] - target_euler[1])
        deviation_roll = np.linalg.norm(curr_euler[2] - target_euler[2])
        if abs(curr_euler[0]) > 1.0 or abs(curr_euler[1]) > 1.0:
            print("Quadrotor Flipped. Bad Quadrotor.")
            orientation_reward = -100.0
        else:
            print("Quadrotor Orientation Within Bounds. Good Quadrotor!")
            reward_yaw = gaussian.pdf(deviation_yaw)
            reward_pitch = gaussian.pdf(deviation_pitch)
            reward_roll = gaussian.pdf(deviation_roll)
            orientation_reward = self.sigmoid(reward_yaw + reward_pitch + reward_roll)
        print("Orientation Reward: %f" % orientation_reward)

        if any(abs(i) >= 9.0 for i in curr_thrusts):
            print("Quadrotor Thrust Too High. Bad Quadrotor!")
            rotor_reward = -100.0
        else:
            print("Quadrotor Thrust Within Bounds. Good Quadrotor!")
            rotor_reward = 10
        print("Thrust Reward: %f" % rotor_reward)

        reward = ((0.8 * pos_reward + 0.8 * orientation_reward + 1.0 * rotor_reward) / 3)
        print("Total Reward: %f" % (reward))
        print('\n')
        return reward

    def sigmoid(self, val):
        return math.exp(val) / (math.exp(val) + 1)
