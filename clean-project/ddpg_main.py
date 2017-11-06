#!/usr/bin/python3

import time, argparse, sys, pdb

sys.path.append('./ddpg_helpers')

from ddpg_utils import *
from ddpg_pytorch_helper import *
from ddpg_evaluator import *
import numpy as np

import pytorch_helper
import quad_helper
import vrep
import vrep_helper
from torch.autograd import Variable
from copy import deepcopy
#clientID = -1
#sim_functions = None
#quad_functions = None
#nn_functions = None
dt = 0.05
        
        
#https://github.com/ghliu/pytorch-ddpg
def train(num_iterations, gent, env,  evaluate, validate_steps, output, max_episode_length=None, debug=False):
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    while step < num_iterations:
        print (step, '/', num_iterations)
        # reset if it is the start of episode
        #pdb.set_trace()
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)

        # agent pick action ...
        if step <= args.warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(observation)
        
        # env response with next_observation, reward, terminate_info
        observation2, reward, done, info = env.step(action)
        print ('State:', observation2, 'Reward:', reward)
        observation2 = deepcopy(observation2)
        if max_episode_length and episode_steps >= max_episode_length -1:
            done = True

        # agent observe and update policy
        agent.observe(reward, observation2, done)
        if step > args.warmup :
            print('Finished Warmup. Training')
            agent.update_policy()
        
        # [optional] evaluate
        if evaluate is not None and validate_steps > 0 and step % validate_steps == 0:
            policy = lambda x: agent.select_action(x, decay_epsilon=False)
            validate_reward = evaluate(env, policy, debug=False, visualize=False)
            if debug: prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))

        # [optional] save intermideate model
        if step % int(num_iterations/3) == 0:
            agent.save_wts(output, step)

        # update 
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done: # end of episode
            if debug: prGreen('#{}: episode_reward:{} steps:{}'.format(episode,episode_reward,step))

            agent.memory.append(
                observation,
                agent.select_action(observation),
                0., False
            )

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1

def test(num_episodes, agent, env, evaluate, model_path, visualize=True, debug=False):
    agent.load_wts(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x, decay_epsilon=False)

    for i in range(num_episodes):
        validate_reward = evaluate(env, policy, debug=debug, visualize=visualize, save=False)
        if debug: prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))

        
#TODO: have different target points at each iteration
#TODO: have last few iterations' actions in state space
class Env(object):
    def __init__(self):
        try:
            vrep.simxFinish(-1)
            self.clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
            self.sim_functions = vrep_helper.Helper(self.clientID)
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
        self.run_time = 3  #TODO check
 
        self.observation_space = 10  #3 + 3 + 4
        self.action_space = 4

    def get_curr_state(self):
        self.curr_pos, self.curr_euler = self.quad_functions.fetch_quad_state()
        self.curr_state = np.array(self.curr_pos + self.curr_euler + self.curr_rotor_thrusts, dtype=np.float32)

    def step(self, delta_rotor_thrusts):
        if vrep.simxGetConnectionId(self.clientID) != -1:
            #pdb.set_trace()
            self.curr_rotor_thrusts = (np.array(self.curr_rotor_thrusts) + delta_rotor_thrusts).tolist()
            self.quad_functions.apply_rotor_thrust(self.curr_rotor_thrusts)
            for i in range(self.run_time):
                vrep.simxSynchronousTrigger(self.clientID)
            self.get_curr_state()
            reward = self.quad_functions.get_reward(self.curr_rotor_thrusts, self.curr_pos, self.curr_euler, self.target_pos, self.target_euler)
            goaldiffpos = np.linalg.norm(np.array(self.curr_pos) - np.array(self.target_pos)) 
            goaldiffeuler = np.linalg.norm(np.array(self.curr_euler) - np.array(self.target_euler))
            pdb.set_trace()
            done = goaldiffpos < 0.1 and goaldiffeuler < 0.01  #TODO set appropriate threshold
            return np.array(self.curr_state), reward, done, None
        
    def reset(self):
        self.curr_rotor_thrusts = [0.000, 0.000, 0.000, 0.000]
        #TODO: how to reset?
        #self.curr_pos, self.curr_euler = quad_functions.fetch_quad_state()
        print('In reset', self.quad_functions.fetch_quad_state())
        self.sim_functions.stop_sim()
        self.quad_functions.set_target([0,0,0.5], [0.0]*3, [0.0]*4)
        self.quad_functions.set_quad_pos([1,1,1], [0.0]*3, [0.0]*4)
        self.sim_functions.start_sim()
        print(self.quad_functions.fetch_quad_state())

        self.get_curr_state()
        return self.curr_state
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quadcopter flying')
    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--hidden1', default=400, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=100, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=6000000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma') 
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu') 
    parser.add_argument('--validate_episodes', default=20, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=500, type=int, help='')
    parser.add_argument('--validate_steps', default=2000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='') 
    parser.add_argument('--train_iter', default=200000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    # parser.add_argument('--l2norm', default=0.01, type=float, help='l2 weight decay') # TODO
    # parser.add_argument('--cuda', dest='cuda', action='store_true') # TODO

    
    args = parser.parse_args()
    argsenv = 'Quad_DDPG'
    args.output = get_output_folder(args.output, argsenv)
    if args.resume == 'default':
        args.resume = 'output/{}-run0'.format(argsenv)
    
    env = Env()

    if args.seed > 0:
        assert False  #Not implemented for now
        np.random.seed(args.seed)
        env.seed(args.seed)

    nb_states = env.observation_space#.shape[0]
    nb_actions = env.action_space#.shape[0]


    agent = DDPG(nb_states, nb_actions, args)
    evaluate = Evaluator(args.validate_episodes, 
        args.validate_steps, args.output, max_episode_length=args.max_episode_length)

    if args.mode == 'train':
        #TODO, pass evaluate instad of None
        train(args.train_iter, agent, env, None, 
            args.validate_steps, args.output, max_episode_length=args.max_episode_length, debug=args.debug)

    elif args.mode == 'test':
        test(args.validate_episodes, agent, env, evaluate, args.resume,
            visualize=True, debug=args.debug)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
