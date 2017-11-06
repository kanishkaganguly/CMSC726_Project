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
        self.reset_rotor = [1, 1, 1, 1]

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
        vrep_rotors.set_rotors(self.clientID, thrusts)
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
            pos_reward = -100000
        else:
            print("Quadrotor Position Within Bounds. Good Quadrotor!")
            reward_x = gaussian.pdf(deviation_x)
            reward_y = gaussian.pdf(deviation_y)
            reward_z = gaussian.pdf(deviation_z)
            pos_reward = self.sigmoid(reward_x + reward_y + reward_z)

        deviation_yaw = np.linalg.norm(curr_euler[0] - target_euler[0])
        deviation_pitch = np.linalg.norm(curr_euler[1] - target_euler[1])
        deviation_roll = np.linalg.norm(curr_euler[2] - target_euler[2])
        if curr_euler[0] > 2.0 or curr_euler[1] > 2.0 or curr_euler[0] < -2.0 or curr_euler[1] < -2.0:
            print("Quadrotor Flipped. Bad Quadrotor.")
            orientation_reward = -100000
        else:
            print("Quadrotor Orientation Within Bounds. Good Quadrotor!")
            reward_yaw = gaussian.pdf(deviation_yaw)
            reward_pitch = gaussian.pdf(deviation_pitch)
            reward_roll = gaussian.pdf(deviation_roll)
            orientation_reward = self.sigmoid(reward_yaw + reward_pitch + reward_roll)

        if any(i >= 11 for i in curr_thrusts):
            print("Quadrotor Thrust Too High. Bad Quadrotor!")
            rotor_reward = -100000
        else:
            print("Quadrotor Thrust Within Bounds. Good Quadrotor!")
            rotor_reward = 100

        reward = ((pos_reward + orientation_reward + rotor_reward) / 2)

        return reward

    def sigmoid(self, val):
        return math.exp(val) / (math.exp(val) + 1)
