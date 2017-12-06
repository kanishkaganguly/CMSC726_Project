import numpy as np


class Quad(object):
    def __init__(self, dqn_quad, control_quad):
        self.mode = "train"
        self.dqn_quad = dqn_quad
        self.control_quad = control_quad
        self.epoch_size = 0

    def write_data(self, epoch, reward, iteration):
        with open('dqn_outputs.txt', 'a') as the_file:
            the_file.write('Epoch: %d Episode: %d\n' % (epoch, iteration))
            the_file.write('Epsilon Greedy: %f\n' % self.dqn_quad.eps)
            the_file.write('Reward: %f\n' % reward)
            the_file.write('Loss: %f\n' % float(self.dqn_quad.loss.data[0]))
            the_file.write('Learning Rate: %f\n' % float(self.dqn_quad.scheduler.get_lr()[0]))
            the_file.write('\n')

    def run_one_episode(self, curr_epoch, curr_episode):
        res = {}
        print("Epoch: %d Episode %d" % (curr_epoch, curr_episode))
        print("Epsilon Greedy: %f" % self.dqn_quad.eps)
        print("DQN Discount Factor: %f" % self.dqn_quad.gamma)

        # Get current state
        print("Getting current state")
        curr_state = np.array(self.control_quad.get_quad_state(), dtype=np.float32)

        # Get action q_values
        print("Getting predicted q_values")
        pred_q = self.dqn_quad.predict_action(curr_state)

        # Get action with max q_value
        print("Getting best action")
        max_q_idx = np.argmax(pred_q)
        max_q = np.amax(pred_q)

        # Do action
        print("Moving quadcopter")
        self.control_quad.move_quad(self.dqn_quad.do_action(max_q_idx))

        # Get new state
        print("Getting new state")
        new_state = self.control_quad.get_quad_state()

        # Test out of bounds
        if abs(new_state[0]) > 10.0 or abs(new_state[1]) > 10.0 or not 0.0 <= new_state[2] <= 5.0:
            print("Quadcopter out of bounds")
            if self.mode == 'test':
                res['status'] = 'fail'
            elif self.mode == 'train':
                # Get reward
                print("Getting reward")
                reward = -50
                res['reset'] = True
        else:
            # Get reward
            print("Getting reward")
            reward = self.dqn_quad.get_reward(new_state, self.control_quad.get_target_state())
            res['reset'] = False

        # Set target q_values for backprop
        print("Setting target q_values")
        target_q = np.copy(pred_q)
        target_q[max_q_idx] = reward + self.dqn_quad.gamma * max_q
        print("Computing loss")
        self.dqn_quad.get_loss(target_q, pred_q)

        # Do backprop
        print("Backpropagation")
        self.dqn_quad.backprop()
        print('\n')

        if curr_episode % 100 == 0:
            self.write_data(curr_epoch, reward, curr_episode)

        return res

    def task_every_n_epochs(self, curr_epoch):
        if curr_epoch % 10 == 0:
            self.dqn_quad.save_wts('dqn_quad.pth', curr_epoch)
        if curr_epoch % 5 == 0:
            self.dqn_quad.eps += (1. / (1. + self.dqn_quad.eps_decay * self.epoch_size))
            self.dqn_quad.gamma += (1. / (1. + self.dqn_quad.gamma_decay * self.epoch_size))
            self.control_quad.reset(rand_target=True)

    def run_one_epoch(self, curr_epoch):
        for i in range(self.dqn_quad.episode_size):
            res = self.run_one_episode(curr_epoch, i)
            if res['reset']:
                self.control_quad.reset()

    def test_quad(self):
        self.control_quad.reset()
        self.control_quad.set_target_state([2, 1, 1, 0])
        self.dqn_quad.gamma = 0.8
        self.dqn_quad.eps = 1.0
        while not self.control_quad.check_target_reached():
            res = self.run_one_episode(100, 100)
            if res['status'] == 'fail':
                print('Our quadrotor failed test.')
                break
            else:
                print('Continuing test.')

        print("Our quadrotor has reached the test target.")
