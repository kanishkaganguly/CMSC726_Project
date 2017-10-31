#       __QUADROTORSIMSPARAMS__
#       This file implements the sim parameters
#       simple quadrotor simulation tool
#
#       Largely based on the work of https://github.com/nikhilkalige

from quadrotor_dynamics import QuadrotorDynamics
import numpy as np
import random
import time
import matplotlib.pyplot as plt

TURNS = 3


class SimulationParams(object):
    def __init__(self):
        self.mass = 1
        self.Ixx = 0.0053
        self.length = 0.2
        self.Bup = 21.58
        self.Bdown = 3.92
        self.Cpmax = np.pi * 1800 / 180
        self.Cn = TURNS
        self.gravity = 9.81

    def get_acceleration(self, p0, p3):
        ap = {
            'acc'    : (-self.mass * self.length * (self.Bup - p0) / (4 * self.Ixx)),
            'start'  : (self.mass * self.length * (self.Bup - self.Bdown) / (4 * self.Ixx)),
            'coast'  : 0,
            'stop'   : (-self.mass * self.length * (self.Bup - self.Bdown) / (4 * self.Ixx)),
            'recover': (self.mass * self.length * (self.Bup - p3) / (4 * self.Ixx)),
        }
        return ap

    def get_initial_parameters(self):
        p0 = p3 = 0.9 * self.Bup
        p1 = p4 = 0.1
        acc_start = self.get_acceleration(p0, p3)['start']
        p2 = (2 * np.pi * self.Cn / self.Cpmax) - (self.Cpmax / acc_start)
        return [p0, p1, p2, p3, p4]

    def get_sections(self, parameters):
        sections = np.zeros(5, dtype='object')
        [p0, p1, p2, p3, p4] = parameters

        ap = self.get_acceleration(p0, p3)

        T2 = (self.Cpmax - p1 * ap['acc']) / ap['start']
        T4 = -(self.Cpmax + p4 * ap['recover']) / ap['stop']

        aq = 0
        ar = 0

        sections[0] = (self.mass * p0, [ap['acc'], aq, ar], p1)

        temp = self.mass * self.Bup - 2 * abs(ap['start']) * self.Ixx / self.length
        sections[1] = (temp, [ap['start'], aq, ar], T2)

        sections[2] = (self.mass * self.Bdown, [ap['coast'], aq, ar], p2)

        temp = self.mass * self.Bup - 2 * abs(ap['stop']) * self.Ixx / self.length
        sections[3] = (temp, [ap['stop'], aq, ar], T4)

        sections[4] = (self.mass * p3, [ap['recover'], aq, ar], p4)
        return sections


def fly_quadrotor(params=None):
    gen = SimulationParams()
    quadrotor = QuadrotorDynamics()
    if not params:
        params = gen.get_initial_parameters()
    sections = gen.get_sections(params)
    state = quadrotor.update_state(sections)
    plt.plot(state)
    plt.show()
    raw_input()