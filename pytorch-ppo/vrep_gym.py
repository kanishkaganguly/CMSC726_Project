import numpy as np

import quad_helper
import vrep
import vrep_helper


class VrepGym(object):
    def __init__(self):
        self.observation_space = np.zeros((6, 1))
        self.action_space = np.zeros((4, 1))
        self.run_time = 30
        self.curr_rotor_thrusts = [0.0, 0.0, 0.0, 0.0]

    def make_gym(self, name):
        self.env_name = name
        try:
            vrep.simxFinish(-1)
            self.clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
            self.sim_functions = vrep_helper.Helper(self.clientID)
            self.sim_functions.load_scene("vrep-quad-scene")
            self.quadHandle = self.sim_functions.get_handle("Quadricopter")
            self.targetHandle = self.sim_functions.get_handle("Quadricopter_target")
            self.quad_functions = quad_helper.QuadHelper(self.clientID, self.quadHandle, self.targetHandle)
            self.target_pos, self.target_euler = self.quad_functions.fetch_target_state()

        except KeyboardInterrupt:
            self.sim_functions.exit_sim()

        if self.clientID == -1:
            print("Failed to connect to remote API Server")
            self.sim_functions.exit_sim()
        self.quad_functions.init_quad()
        print('Quadrotor Initialized')

        self.reset()

    def get_curr_state(self):
        self.curr_pos, self.curr_euler = self.quad_functions.fetch_quad_state()
        self.curr_state = np.array(self.curr_pos + self.curr_euler, dtype=np.float32)

    def step(self, delta_rotor_thrusts):
        if vrep.simxGetConnectionId(self.clientID) != -1:
            self.curr_rotor_thrusts = (np.array(self.curr_rotor_thrusts) + delta_rotor_thrusts).tolist()
            self.quad_functions.apply_rotor_thrust(self.curr_rotor_thrusts)
            for i in range(self.run_time):
                vrep.simxSynchronousTrigger(self.clientID)
            self.get_curr_state()
            reward = self.quad_functions.get_reward(self.curr_rotor_thrusts, self.curr_pos, self.curr_euler,
                                                    self.target_pos, self.target_euler)
            goaldiffpos = np.linalg.norm(np.array(self.curr_pos) - np.array(self.target_pos))
            goaldiffeuler = np.linalg.norm(np.array(self.curr_euler) - np.array(self.target_euler))
            done = goaldiffpos < 0.5 and goaldiffeuler < 0.1  # TODO set appropriate threshold
            return np.array(self.curr_state), reward, done, None

    def reset(self):
        print("Reset Quadrotor")
        self.curr_rotor_thrusts = [0.001, 0.001, 0.001, 0.001]
        self.sim_functions.stop_sim()
        self.quad_functions.set_target([0, 0, 0.5], [0.0] * 3, [0.0] * 4)
        self.sim_functions.start_sim()
        self.get_curr_state()
        print(self.curr_state)
        return self.curr_state
