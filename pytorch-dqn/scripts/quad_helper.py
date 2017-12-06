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

        print("Initializing simulator")
        self.sim_quad = SimHelper()
        self.sim_quad.load_scene('quad_scene')

        print("Fetching quad, target handles")
        self.quad_handle = self.sim_quad.get_handle('Quadricopter_target')
        self.target_handle = self.sim_quad.get_handle('Target')

        print("Initializing state functions")
        self.states_quad = StateHelper(self.sim_quad.clientID, [self.quad_handle, self.target_handle])
        self.quad_state = self.states_quad.get_state(self.sim_quad.clientID, self.quad_handle)
        self.states_quad.set_state(self.sim_quad.clientID, self.target_handle, self.target_state)

        self.sim_quad.start_sim()
        print("Ready to fly...")

    def move_quad(self, direction):
        if direction == "FWD":
            move_to = self.quad_state
            move_to[0] += 0.05
        if direction == "BCK":
            move_to = self.quad_state
            move_to[0] -= 0.05
        if direction == "RGT":
            move_to = self.quad_state
            move_to[1] += 0.05
        if direction == "LFT":
            move_to = self.quad_state
            move_to[1] -= 0.05
        if direction == "UP":
            move_to = self.quad_state
            move_to[2] += 0.05
        if direction == "DWN":
            move_to = self.quad_state
            move_to[2] -= 0.05
        if direction == "ROT_CW":
            move_to = self.quad_state
            move_to[3] += 0.05
        if direction == "ROT_CCW":
            move_to = self.quad_state
            move_to[3] -= 0.05

        print("Moving %s" % direction)
        self.states_quad.set_state(self.sim_quad.clientID, self.quad_handle, move_to)
        # Allow quadcopter to settle down to new position
        for i in range(5):
            self.step()

    def step(self):
        self.sim_quad.step_sim(self.sim_quad.clientID)

    def reset(self, rand_target=False):
        self.sim_quad.reset()
        self.quad_state = np.zeros(4)
        if rand_target:
            x_rand = np.random.uniform(self.x_target_limits[0], self.x_target_limits[1])
            y_rand = np.random.uniform(self.y_target_limits[0], self.y_target_limits[1])
            z_rand = np.random.uniform(self.z_target_limits[0], self.z_target_limits[1])
            self.target_state = np.array([x_rand, y_rand, z_rand, 0.0])
        print("New target state: (%f,%f,%f,%f)" % (
            self.target_state[0], self.target_state[2], self.target_state[2], self.target_state[3]))
        self.states_quad.set_state(self.sim_quad.clientID, self.target_handle, self.target_state)

    def get_state(self):
        return self.states_quad.get_state(self.sim_quad.clientID, self.quad_handle)
