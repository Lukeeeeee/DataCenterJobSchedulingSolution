import numpy as np
import matplotlib.pyplot as plt
import math
from src.config import Config as con


# sever_state[bs, server_count, server_state_dim] a np arrray
# state [R, ENERGY, Y, HEAT, IS_VALID, USAGE, X]
# action[bs, 1]
# q[bs, 5]

def plot_mean_of_efficiency_with_epoch(res):
    plt.xlabel('Epoch')
    plt.ylabel('Mean of Efficiency')

    choosed_action_efficiency = [[] for _ in range(con.epoch // con.save_data_every_epoch)]
    non_choosed_action_efficiency = [[] for _ in range(con.epoch // con.save_data_every_epoch)]

    for sample in res:
        for (state, action) in zip(sample['SERVER_STATE'], sample['ACTION']):
            for i in range(len(state)):
                eff = (state[i][5] + 0.000001) / (state[i][1] + 0.000001)
                if i == action:
                    choosed_action_efficiency[sample['EPOCH'] // con.save_data_every_epoch].append(eff)
                else:
                    non_choosed_action_efficiency[sample['EPOCH'] // con.save_data_every_epoch].append(eff)
    choosed_action_efficiency = np.array(choosed_action_efficiency).mean(axis=1)
    non_choosed_action_efficiency = np.array(non_choosed_action_efficiency).mean(axis=1)

    plt.scatter([i for i in range(len(choosed_action_efficiency))], choosed_action_efficiency, c='r')
    plt.scatter([i for i in range(len(choosed_action_efficiency))], non_choosed_action_efficiency, c='k')

    plt.plot([i for i in range(len(choosed_action_efficiency))], choosed_action_efficiency, c='g',
             label='Chosen Server')
    plt.plot([i for i in range(len(choosed_action_efficiency))], non_choosed_action_efficiency, c='b',
             label='Not Chosen Server')
    plt.legend()
    plt.show()


def plot_mean_of_dis_with_epoch(res):
    plt.xlabel('Epoch')
    plt.ylabel('Mean of Distance to center')

    choosed_action_dis = [[] for _ in range(con.epoch // con.save_data_every_epoch)]
    non_choosed_action_dis = [[] for _ in range(con.epoch // con.save_data_every_epoch)]

    for sample in res:
        for (state, action) in zip(sample['SERVER_STATE'], sample['ACTION']):
            for i in range(len(state)):
                eff = math.sqrt(state[i][2] * state[i][2] + state[i][6] * state[i][6])
                if i == action:
                    choosed_action_dis[sample['EPOCH'] // con.save_data_every_epoch].append(eff)
                else:
                    non_choosed_action_dis[sample['EPOCH'] // con.save_data_every_epoch].append(eff)
    choosed_action_dis = np.array(choosed_action_dis).mean(axis=1)
    non_choosed_action_dis = np.array(non_choosed_action_dis).mean(axis=1)

    plt.scatter([i for i in range(len(choosed_action_dis))], choosed_action_dis, c='r')
    plt.scatter([i for i in range(len(choosed_action_dis))], non_choosed_action_dis, c='k')

    plt.plot([i for i in range(len(choosed_action_dis))], choosed_action_dis, c='g', label='Chosen Server')
    plt.plot([i for i in range(len(choosed_action_dis))], non_choosed_action_dis, c='b', label='Not Chosen Server')
    plt.legend()
    plt.show()


def visual(res=None):
    if res is None:
        log_dir = '../log/1-21-0-27-31/training_data.npy'
        res = np.load(log_dir)
    # plot_mean_of_efficiency_with_epoch(res)
    plot_mean_of_dis_with_epoch(res)
    plot_mean_of_efficiency_with_epoch(res)
    pass


if __name__ == '__main__':
    visual()
