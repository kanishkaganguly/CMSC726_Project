#!/usr/bin/python3

import math

import numpy as np
from scipy.stats import norm

import vrep_rotors
import vrep_state


class QuadHelper(object):
    def __init__(self, clientID, quadHandle, targetHandle, rootHandle):
        self.clientID = clientID
        self.quadHandle = quadHandle
        self.targetHandle = targetHandle
        self.rootHandle = rootHandle
        self.reset_pos = [0, 0, 0]
        self.reset_euler = [0, 0, 0]
        self.reset_rotor = [1e-6, 1e-6, 1e-6, 1e-6]

    '''
    Initialize and reset quadcopter position in world
    '''

    def init_quad(self):
        vrep_rotors.init_rotors(self.clientID)
        vrep_state.init_state(self.clientID, self.quadHandle)
        return

    '''
    Reset Quadcopter Position
    '''

    def reset_quad(self):
        vrep_state.set_quad_position(self.clientID, self.quadHandle, self.rootHandle, self.reset_pos)
        vrep_state.set_quad_euler(self.clientID, self.quadHandle, self.rootHandle, self.reset_euler)
        vrep_rotors.set_rotors(self.clientID, self.reset_rotor)
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

    '''
    This function returns reward based on current state and target state (x,y,z,yaw,pitch,roll)
    '''

    def get_reward(self, curr_pos, curr_euler, target_pos, target_euler):
        gaussian = norm(0, 2)

        deviation_x = np.linalg.norm(curr_pos[0] - target_pos[0])
        deviation_y = np.linalg.norm(curr_pos[1] - target_pos[1])
        deviation_z = np.linalg.norm(curr_pos[2] - target_pos[2])
        reward_x = gaussian.pdf(deviation_x)
        reward_y = gaussian.pdf(deviation_y)
        reward_z = gaussian.pdf(deviation_z)
        pos_reward = self.sigmoid(reward_x + reward_y + reward_z)

        deviation_yaw = np.linalg.norm(curr_euler[0] - target_euler[0])
        deviation_pitch = np.linalg.norm(curr_euler[1] - target_euler[1])
        deviation_roll = np.linalg.norm(curr_euler[2] - target_euler[2])
        reward_yaw = gaussian.pdf(deviation_yaw)
        reward_pitch = gaussian.pdf(deviation_pitch)
        reward_roll = gaussian.pdf(deviation_roll)
        orientation_reward = self.sigmoid(reward_yaw + reward_pitch + reward_roll)

        return ((pos_reward + orientation_reward) / 2)

    def sigmoid(self, val):
        return math.exp(val) / (math.exp(val) + 1)
