#! /usr/bin/env python3

import numpy as np

from vrep_helper import SimHelper
from vrep_state import StateHelper


class QuadHelper(object):
    def __init__(self):
        self.quad_state = np.zeros(4)
        self.target_state = np.zeros(4)

        print("Initializing simulator")
        self.sim_quad = SimHelper()
        self.sim_quad.load_scene('quad_scene')
        self.sim_quad.start_sim()

        print("Fetching quad, target handles")
        self.quad_handle = self.sim_quad.get_handle('Quadricopter_target')
        self.target_handle = self.sim_quad.get_handle('Target')

        print("Initializing state functions")
        self.states_quad = StateHelper(self.sim_quad.clientID, [self.quad_handle, self.target_handle])
        self.quad_state = self.states_quad.get_state(self.sim_quad.clientID, self.quad_handle)
        self.target_state = self.states_quad.get_state(self.sim_quad.clientID, self.target_handle)
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

        self.states_quad.set_state(self.sim_quad.clientID, self.quad_handle, move_to)
        self.step()

    def step(self):
        self.sim_quad.step_sim(self.sim_quad.clientID)

    def get_state(self):
        return self.states_quad.get_state(self.sim_quad.clientID, self.quad_handle)
