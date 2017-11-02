#!/usr/bin/python3

import vrep_rotors
import vrep_state


class RL(object):
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
    Reset Quadcopter Position
    '''

    def reset_quad(self):
        vrep_state.set_quad_position(self.clientID, self.quadHandle, self.reset_quad())
        vrep_state.set_quad_euler(self.clientID, self.quadHandle, self.euler)
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

# '''
# This function returns reward based on current and previous location data (x,y,z)
# '''
#
# def get_reward(self):
#     self.curr_location = self.get_state()
#     deviation_x = np.linalg.norm(self.curr_location[0] - self.target_location[0])
#     deviation_y = np.linalg.norm(self.curr_location[1] - self.target_location[1])
#     deviation_z = np.linalg.norm(self.curr_location[2] - self.target_location[2])
#     gaussian = norm(0, 2)
#
#     reward_x = gaussian.pdf(deviation_x)
#     reward_y = gaussian.pdf(deviation_y)
#     reward_z = 1 - math.exp(deviation_z)
#
#     total_reward = 2 * (0.5 * reward_x + 0.5 * reward_y + reward_z)
#     return total_reward
