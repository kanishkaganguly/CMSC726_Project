#! /usr/bin/env python3

import numpy as np

from vrep_helper import SimHelper
from vrep_state import StateHelper


class QuadHelper(object):
    def __init__(self):
        self.quad_state = np.zeros(4)
        self.target_state = np.array([2.0, 0.0, 3.0, 0.0])
        self.x_target_limits = [-5, 5]
        self.y_target_limits = [-5, 5]
        self.z_target_limits = [1, 5]

        self.display_disabled = False

        print("Initializing simulator")
        self.sim_quad = SimHelper()
        self.sim_quad.load_scene('quad_scene')
        if self.display_disabled:
            self.sim_quad.display_disabled()

        print("Fetching quad, target handles")
        self.quad_handle = self.sim_quad.get_handle('Quadricopter_target')
        self.target_handle = self.sim_quad.get_handle('Target')

        print("Initializing state functions")
        self.states_quad = StateHelper(self.sim_quad.clientID, [self.quad_handle, self.target_handle])
        self.quad_state = self.states_quad.get_state(self.sim_quad.clientID, self.quad_handle)
        self.states_quad.set_state(self.sim_quad.clientID, self.target_handle, self.target_state)

        self.sim_quad.start_sim()
        print("Ready to fly...")

    def move_quad(self, direction, step_size=0.05):
        if direction == "FWD":
            move_to = self.quad_state
            move_to[0] += step_size
        if direction == "BCK":
            move_to = self.quad_state
            move_to[0] -= step_size
        if direction == "RGT":
            move_to = self.quad_state
            move_to[1] += step_size
        if direction == "LFT":
            move_to = self.quad_state
            move_to[1] -= step_size
        if direction == "UP":
            move_to = self.quad_state
            move_to[2] += step_size
        if direction == "DWN":
            move_to = self.quad_state
            move_to[2] -= step_size
        if direction == "ROT_CW":
            move_to = self.quad_state
            move_to[3] += step_size
        if direction == "ROT_CCW":
            move_to = self.quad_state
            move_to[3] -= step_size

        print("Moving %s" % direction)
        self.states_quad.set_state(self.sim_quad.clientID, self.quad_handle, move_to)
        # Allow quadcopter to settle down to new position
        for i in range(5):
            self.step()

    def step(self):
        self.sim_quad.step_sim()

    def reset(self, rand_target=False, display_disabled=False):
        self.sim_quad.reset(display_disabled)
        self.quad_state = np.zeros(4)
        if rand_target:
            x_rand = np.random.uniform(self.x_target_limits[0], self.x_target_limits[1])
            y_rand = np.random.uniform(self.y_target_limits[0], self.y_target_limits[1])
            z_rand = np.random.uniform(self.z_target_limits[0], self.z_target_limits[1])
            self.target_state = np.array([x_rand, y_rand, z_rand, 0.0])

        self.set_target_state(self.target_state)

    def set_target_state(self, target):
        self.target_state = np.array(target)

        print("New target state: (%f,%f,%f,%f)" % (
            self.target_state[0], self.target_state[2], self.target_state[2], self.target_state[3]))

        self.states_quad.set_state(self.sim_quad.clientID, self.target_handle, self.target_state)

    def get_target_state(self):
        return self.states_quad.get_state(self.sim_quad.clientID, self.target_handle)

    def get_quad_state(self):
        return self.states_quad.get_state(self.sim_quad.clientID, self.quad_handle)

    def check_target_reached(self):
        curr_state = self.get_quad_state()
        deviation_x = np.linalg.norm(curr_state[0] - self.target_state[0])
        deviation_y = np.linalg.norm(curr_state[1] - self.target_state[1])
        deviation_z = np.linalg.norm(curr_state[2] - self.target_state[2])
        deviation_yaw = np.linalg.norm(curr_state[3] - self.target_state[3])

        total = deviation_x + deviation_y + deviation_z + deviation_yaw

        return total < 0.1
