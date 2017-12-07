from collections import defaultdict

import numpy as np
from visdom import Visdom


class Visualizer(object):
    def __init__(self):
        self.viz = Visdom()
        self.plot_list = defaultdict(lambda: None)

        self.viz.close(win=None)

        self.init_plots()

    def init_plots(self):
        if self.plot_list['epsilon'] is None:
            eps = self.viz.scatter(
                X=np.array([[0.0, 0.0]]),
                opts=dict(
                    showlegend=False,
                    ylabel='Epsilon Greedy',
                    xlabel='Episode',
                    title='Epsilon Greedy vs. Episode',
                    marginleft=30,
                    marginright=30,
                    marginbottom=80,
                    margintop=30,
                ),
            )
            self.plot_list['epsilon'] = eps

        if self.plot_list['gamma'] is None:
            gamma = self.viz.scatter(
                X=np.array([[0.0, 0.0]]),
                opts=dict(
                    showlegend=False,
                    ylabel='Discount Factor',
                    xlabel='Episode',
                    title='Discount Factor vs. Episode',
                    marginleft=30,
                    marginright=30,
                    marginbottom=80,
                    margintop=30,
                ),
            )
            self.plot_list['gamma'] = gamma

        if self.plot_list['learning_rate'] is None:
            lr = self.viz.scatter(
                X=np.array([[0.0, 0.0]]),
                opts=dict(
                    showlegend=False,
                    ylabel='Learning Rate',
                    xlabel='Episode',
                    title='Learning Rate vs. Episode',
                    marginleft=30,
                    marginright=30,
                    marginbottom=80,
                    margintop=30,
                ),
            )
            self.plot_list['learning_rate'] = lr

    def append_plots(self, plot, x_val, y_val):
        self.viz.scatter(
            X=np.array([[y_val, x_val]]),
            win=self.plot_list[plot],
            update='append'
        )

# if __name__ == '__main__':
#     viz = Visualizer()
#     viz.init_plots()
#     viz.append_plots('epsilon', 4.5, 5.5)
#     viz.append_plots('gamma', 2.5, 2.0)
