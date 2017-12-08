#!/usr/bin/python3

import platform
import subprocess as sp
import threading
import time

import vrep


class SimHelper(object):
    def __init__(self):
        if platform.system() != 'Darwin':
            self.start_vrep()
        self.setup_vrep_remote()
        if platform.system() != 'Darwin':
            self.check_vrep()

    '''
    Turn on V-REP application
    '''

    def start_vrep(self):
        try:
            check_vrep_running = sp.check_output(["pidof", "vrep"])
            self.pid = int(check_vrep_running.split()[0])
            print("V-REP already running...")
            launch_vrep = False
        except sp.CalledProcessError:
            launch_vrep = True
            pass

        if launch_vrep:
            print("Starting V-REP...")
            sp.call(['/bin/bash', '-i', '-c', "vrep"])
            time.sleep(5.0)
            check_vrep_running = sp.check_output(["pidof", "vrep"])
            self.pid = int(check_vrep_running.split()[0])
            pass

    '''
    Setup V-REP remote connection
    '''

    def setup_vrep_remote(self):
        try:
            vrep.simxFinish(-1)
            self.clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
            if self.clientID == -1:
                print("Failed to connect to remote API Server")
                self.sim_functions.exit_sim()
        except KeyboardInterrupt:
            self.exit_sim()

    '''
    Check V-REP running
    '''

    def check_vrep(self):
        t = threading.Timer(60.0, self.check_vrep)
        t.daemon = True
        print("Checking V-REP")
        t.start()
        self.start_vrep()

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

    def step_sim(self):
        vrep.simxSynchronousTrigger(self.clientID)
        return

    '''
    Reset V-REP simulation
    '''

    def reset(self, display_disabled):
        self.stop_sim()
        time.sleep(0.1)
        self.start_sim()
        if display_disabled:
            self.display_disabled()
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

    '''
    Turn off V-REP display
    '''

    def display_disabled(self):
        vrep.simxSetBooleanParameter(self.clientID, vrep.sim_boolparam_display_enabled, False,
                                     vrep.simx_opmode_oneshot_wait)
        vrep.simxSetBooleanParameter(self.clientID, vrep.sim_boolparam_browser_visible, False,
                                     vrep.simx_opmode_oneshot_wait)
        vrep.simxSetBooleanParameter(self.clientID, vrep.sim_boolparam_hierarchy_visible, False,
                                     vrep.simx_opmode_oneshot_wait)
        return
