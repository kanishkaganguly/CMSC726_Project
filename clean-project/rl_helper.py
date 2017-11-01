#!/usr/bin/python3

import math
from random import randint

import numpy as np
from scipy.stats import norm

import vrep
import vrep_imu
import vrep_rotors


class RL(object):
    def __init__(self, clientID):
        self.clientID = clientID
        self.quadHandle = None
        self.pos = [0, 0, 0]
        self.euler = [0, 0, 0]
        self.rotor_data = [0.0, 0.0, 0.0, 0.0]
        self.orig_location = [0, 0, 0]
        self.curr_location = [0, 0, 0]
        self.curr_euler = [0, 0, 0]
        self.target_location = [0.0, 0.0, 0.0]
        self.target_orientation = [0.0, 0.0, 0.0]

    '''
    Initialize all sensors and reset quadcopter position in world
    '''

    def init_sensors(self):
        # Initialize IMU
        err, self.quadHandle = vrep.simxGetObjectHandle(self.clientID, 'Quadricopter', vrep.simx_opmode_blocking)
        vrep_imu.init_imu(self.clientID, self.quadHandle)

        # Initialize Rotors
        vrep_rotors.init_rotors(self.clientID)

    '''
    Reset Quadcopter Position
    '''

    def reset_quad(self):
        # Reset quadcopter position
        self.pos = [0, 0, 0]
        self.euler = [0, 0, 0]
        self.rotor_data = [0.0, 0.0, 0.0, 0.0]
        vrep.simxSetObjectPosition(self.clientID, self.quadHandle, -1, self.pos, vrep.simx_opmode_oneshot)
        vrep.simxSetObjectOrientation(self.clientID, self.quadHandle, -1, self.euler, vrep.simx_opmode_oneshot)
        return

    '''
    This function returns reward based on current and previous location data (x,y,z)
    '''

    def get_reward(self):
        self.curr_location = self.get_state()
        deviation_x = np.linalg.norm(self.curr_location[0] - self.target_location[0])
        deviation_y = np.linalg.norm(self.curr_location[1] - self.target_location[1])
        deviation_z = np.linalg.norm(self.curr_location[2] - self.target_location[2])
        gaussian = norm(0, 2)

        reward_x = gaussian.pdf(deviation_x)
        reward_y = gaussian.pdf(deviation_y)
        reward_z = 1 - math.exp(deviation_z)

        total_reward = 2 * (0.5 * reward_x + 0.5 * reward_y + reward_z)
        return total_reward

    '''
    This function moves quadcopter rotors
    '''

    def do_action(self):
        vrep_rotors.move_rotors(self.clientID, self.rotor_data)
        return

    '''
    This function picks one action
    '''

    def pick_action(self):
        actions = [0.5, -0.5, 0.0]
        return actions[randint(0, 2)]

    '''
    This function gets quadcopter state
    '''

    def get_state(self):
        self.pos = vrep_imu.get_pos(self.clientID, self.quadHandle)
        self.euler = vrep_imu.get_orientation(self.clientID, self.quadHandle)
        return self.pos, self.euler
