__author__ = 'patras'

import pytorch_helper
import matplotlib.pyplot as plt

def testRewards():
    module = pytorch_helper.QuadDQN()

    target_state = [5,0,0,0]

    value_list = []
    r_x = []
    r_y = []
    r_z = []

    for val in range(100, 501, 1):
        value_list.append(val/100)
        r_x.append(module.get_reward([float(val/100), 0, 0, 0], target_state))
        r_y.append(module.get_reward([0, float(val/100), 0, 0], target_state))
        r_z.append(module.get_reward([0, 0, float(val/100), 0], target_state))

    PlotRewards(value_list, r_x, r_y, r_z)

def PlotRewards(val, x, y, z):
    plt.plot(val, x, label='x')
    plt.plot(val, y, label='y')
    plt.plot(val, z, label='z')
    plt.ylabel('Reward')
    plt.xlabel('Position')
    plt.legend(bbox_to_anchor=(1.05, 0.9), loc=2, borderaxespad=0.)
    plt.savefig('Reward_with_position.png', bbox_inches='tight')


if __name__=="__main__":
    testRewards()

