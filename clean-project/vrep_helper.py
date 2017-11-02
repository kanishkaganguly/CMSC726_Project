#!/usr/bin/python3

import vrep


class Helper(object):
    def __init__(self, clientID):
        self.clientID = clientID

    '''
    Start V-REP simulation
    '''

    def start_sim(self):
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot_wait)
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
    Fetch handle for object
    '''

    def get_handle(self, obj_name):
        err, handle = vrep.simxGetObjectHandle(self.clientID, obj_name, vrep.simx_opmode_blocking)
        return handle
