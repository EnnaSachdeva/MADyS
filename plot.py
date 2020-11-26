import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.optimize import curve_fit 
import pandas as pd
import random

def convolution(x, n_point):
    '''
    :param x: array of input time series data
    :param n_point: moving average filter length
    :return: array after convolved operation
    '''
    # get the mask for convolution
    mask = np.ones((1, n_point))/n_point
    mask = mask[0, :]

    # convolve the mask with the raw data
    convolved_data = np.convolve(x, mask, 'same')
    return convolved_data


def get_mean_variance(data_x, data_y):
    mean = np.mean(data_y, axis=0)
    standard_dev = 1.960*np.std(data_y, axis=0) /np.sqrt(len(data_y))
    return mean, standard_dev


########################## Homogeneous case, with just fire trucks

base_folder = '/home/aadi-z640/research/H_MRERL_sequential_tasks_actual/1_agent_2_temporal_coupling__1_POI_each_type_1_spatial_coupling/evolution_only/'

max_lim = 8000000 # 10 million


x1, y1 = [], []
with open(base_folder + 'Homogeneous_with_cardinal_only_1_1_2_1/metrics/reward_pop20_env-rover_heterogeneous_fire_truck_uav_action_same_seed1_rewardmultiple_uav1_trucks0_coupling-uav1_coupling-truck0_obs-uav800_obs-truck800.csv', 'r') as csvfile:

    plots1 = csv.reader(csvfile, delimiter=',')

    for row in plots1:
        if (float(row[0]) > max_lim):
            break
        else:
            x1.append(float(row[0]))
            y1.append(float(row[1]))


x2, y2 = [], []
with open(base_folder + 'Homogeneous_with_cardinal_only_1_1_2_2000/metrics/reward_pop20_env-rover_heterogeneous_fire_truck_uav_action_same_seed2000_rewardmultiple_uav1_trucks0_coupling-uav1_coupling-truck0_obs-uav800_obs-truck800.csv', 'r') as csvfile:

    plots1 = csv.reader(csvfile, delimiter=',')

    for row in plots1:
        if (float(row[0]) > max_lim):
            break
        else:
            x2.append(float(row[0]))
            y2.append(float(row[1]))


'''
x3, y3 = [], []
with open(base_folder + 'Homogeneous_with_cardinal_only_2_22010/metrics/reward_pop20_roll0_env-rover_heterogeneous_fire_truck_uav_action_same_seed2010_rewardmultiple_uav2_trucks0_coupling-uav2_coupling-truck0_obs-uav800_obs-truck800.csv', 'r') as csvfile:

    plots1 = csv.reader(csvfile, delimiter=',')

    for row in plots1:
        if (float(row[0]) > max_lim):
            break
        else:
            x3.append(float(row[0]))
            y3.append(float(row[1]))


x4, y4 = [], []
with open(base_folder + 'Homogeneous_with_cardinal_only_2_22080/metrics/reward_pop20_roll0_env-rover_heterogeneous_fire_truck_uav_action_same_seed2080_rewardmultiple_uav2_trucks0_coupling-uav2_coupling-truck0_obs-uav800_obs-truck800.csv',
        'r') as csvfile:

    plots1 = csv.reader(csvfile, delimiter=',')

    for row in plots1:
        if (float(row[0]) > max_lim):
            break
        else:
            x4.append(float(row[0]))
            y4.append(float(row[1]))

'''

# get interpolated values of all evaluated at x1
x_ref = x1
y1_interp = np.interp(x_ref, x1, y1)
y2_interp = np.interp(x_ref, x2, y2)

#y3_interp = np.interp(x_ref, x3, y3)
#y4_interp = np.interp(x_ref, x4, y4)

data_x = x1
data_y = []
data_y.append(y1_interp)
data_y.append(y2_interp)
'''
data_y.append(y3_interp)
data_y.append(y4_interp)
'''
#data_y.append(y3_interp)
#data_y.append(y4_interp)

mean, standard_dev = get_mean_variance(data_x, data_y)

plt.fill_between(data_x, mean - standard_dev, mean + standard_dev, color='g', alpha=0.1)
mean = convolution(mean, 2)
plt.plot(data_x, mean, 'g', linewidth=2)



plt.rcParams.update({'font.size': 20})

plt.tick_params(labelsize=20)

plt.xlabel('Time Steps', fontsize=20)
plt.ylabel('Performance (Average Reward)', fontsize=20)
#plt.title('With 100% observability of UGVs (1 POI, 1 UAVs, 2 Trucks, coupling-2 Trucks)', fontsize=30)
#plt.legend(["Without UAV: 2018", "Without UAV: 2019 ", "Without UAV: 2020", "Without UAV: 2021", "Average performance Without UAV",
#			"With UAV: 2018", "With UAV: 2019 ", "With UAV: 2020", "With UAV: 2021", "Average performance With UAV (same speed as truck)",
#			"With UAV (faster than truck): 2018"], fontsize=10)
plt.legend(["Evolution"], fontsize=20)


plt.show()

