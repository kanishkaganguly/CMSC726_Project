#!/usr/bin/python3

import time

import vrep


class SimHelper(object):
    def __init__(self):
        try:
            vrep.simxFinish(-1)
            self.clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
            if self.clientID == -1:
                print("Failed to connect to remote API Server")
                self.sim_functions.exit_sim()
        except KeyboardInterrupt:
            self.exit_sim()

    '''
    Start V-REP simulation
    '''

    def start_sim(self):
        # Set Simulator
        vrep.simxSynchronous(self.clientID, True)
        dt = .05
        vrep.simxSetFloatingParameter(self.clientID,
                                      vrep.sim_floatparam_simulation_time_step,
                                      dt,  # specify a simulation time step
                                      vrep.simx_opmode_oneshot)
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)
        return

    '''
        Stop V-REP simulation
    '''

    def stop_sim(self):
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot_wait)
        return

    '''
    Pause V-REP simulation
    '''

    def pause_sim(self):
        vrep.simxPauseSimulation(self.clientID, vrep.simx_opmode_oneshot_wait)
        return

    '''
    Exit sequence
    '''

    def exit_sim(self):
        self.stop_sim()
        vrep.simxFinish(self.clientID)
        return

    '''
    Step V-REP simulation
    '''

    def step_sim(self, clientID):
        vrep.simxSynchronousTrigger(clientID)
        return

    '''
    Reset V-REP simulation
    '''

    def reset(self):
        self.stop_sim()
        time.sleep(0.1)
        self.start_sim()
        return

    ''' 
    Fetch handle for object
    '''

    def get_handle(self, obj_name):
        err, handle = vrep.simxGetObjectHandle(self.clientID, obj_name, vrep.simx_opmode_blocking)
        return handle

    '''
    Load V-REP scene
    '''

    def load_scene(self, scene_name):
        vrep.simxLoadScene(self.clientID, scene_name + ".ttt", 0xFF, vrep.simx_opmode_blocking)
        return