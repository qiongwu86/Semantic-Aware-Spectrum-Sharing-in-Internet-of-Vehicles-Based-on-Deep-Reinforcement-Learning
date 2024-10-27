from __future__ import division
import numpy as np
import pandas as pd
import time
import random
import math
from sklearn.preprocessing import MinMaxScaler
import scipy.io


np.random.seed(1234)

mat_data = scipy.io.loadmat('sem_table.mat')
# 加载 'new_data_i_need.csv' 文件
table_data = mat_data['sem_table']


class V2Vchannels:
    # Simulator of the V2V Channels

    def __init__(self):
        self.t = 0
        self.h_bs = 1.5
        self.h_ms = 1.5
        self.fc = 2
        self.decorrelation_distance = 10
        self.shadow_std = 3

    def get_path_loss(self, position_A, position_B):
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        d = math.hypot(d1, d2) + 0.001
        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10 ** 9) / (3 * 10 ** 8)

        def PL_Los(d):
            if d <= 3:
                return 22.7 * np.log10(3) + 41 + 20 * np.log10(self.fc / 5)
            else:
                if d < d_bp:
                    return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc / 5)
                else:
                    return 40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(self.h_bs) - 17.3 * np.log10(self.h_ms) + 2.7 * np.log10(self.fc / 5)

        def PL_NLos(d_a, d_b):
            n_j = max(2.8 - 0.0024 * d_b, 1.84)
            return PL_Los(d_a) + 20 - 12.5 * n_j + 10 * n_j * np.log10(d_b) + 3 * np.log10(self.fc / 5)

        if min(d1, d2) < 7:
            PL = PL_Los(d)
        else:
            PL = min(PL_NLos(d1, d2), PL_NLos(d2, d1))
        return PL  # + self.shadow_std * np.random.normal()

    def get_shadowing(self, delta_distance, shadowing):
        return np.exp(-1 * (delta_distance / self.decorrelation_distance)) * shadowing \
               + math.sqrt(1 - np.exp(-2 * (delta_distance / self.decorrelation_distance))) * np.random.normal(0, 3)  # standard dev is 3 db

def calculate_snr_V2I(signal_powers, interference_powers):
    snr_list = []
    for signal_power, interference_power in zip(signal_powers, interference_powers):
        # 将信号和干扰的功率转换成分贝（dB）
        signal_power_db = 10 * np.log10(signal_power)
        interference_power_db = 10 * np.log10(interference_power)
        snr_db = signal_power_db - interference_power_db
        snr_list.append(snr_db)

    return snr_list
def calculate_snr_V2V(signal_powers, interference_powers):
    snr_list = []
    for signal_power, interference_power in zip(signal_powers, interference_powers):
        # 将信号和干扰的功率转换成分贝（dB）
        # 确保signal_power不包含零或负数
        non_zero_signal_power = np.maximum(signal_power, 1e-10)  # 或者选择适当的小正数代替1e-10
        # 计算signal_power_db
        signal_power_db = 10 * np.log10(non_zero_signal_power)
        interference_power_db = 10 * np.log10(interference_power)
        snr_db = signal_power_db - interference_power_db
        snr_list.append(snr_db)

    return snr_list


def limit_array_range(arr, min_value, max_value):
    """
    限制NumPy数组的元素范围。

    Parameters:
    arr (numpy.ndarray): 要限制范围的NumPy数组。
    min_value (float): 允许的最小值。
    max_value (float): 允许的最大值。

    Returns:
    numpy.ndarray: 限制范围后的NumPy数组。
    """
    return np.clip(arr, min_value, max_value)




class V2Ichannels:

    # Simulator of the V2I channels

    def __init__(self):
        self.h_bs = 25
        self.h_ms = 1.5
        self.Decorrelation_distance = 50
        self.BS_position = [750 / 2, 1299 / 2]  # center of the grids
        self.shadow_std = 8  #BS天线的增益gain

    def get_path_loss(self, position_A):
        d1 = abs(position_A[0] - self.BS_position[0])
        d2 = abs(position_A[1] - self.BS_position[1])
        distance = math.hypot(d1, d2) # 计算两个数的欧几里得距离 返回它们的平方和的平方根
        return 128.1 + 37.6 * np.log10(math.sqrt(distance ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000) # + self.shadow_std * np.random.normal()

    def get_shadowing(self, delta_distance, shadowing):
        nVeh = len(shadowing)
        self.R = np.sqrt(0.5 * np.ones([nVeh, nVeh]) + 0.5 * np.identity(nVeh))
        return np.multiply(np.exp(-1 * (delta_distance / self.Decorrelation_distance)), shadowing) \
               + np.sqrt(1 - np.exp(-2 * (delta_distance / self.Decorrelation_distance))) * np.random.normal(0, 8, nVeh)


class Vehicle:
    # Vehicle simulator: include all the information for a vehicle

    def __init__(self, start_position, start_direction, velocity):
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity
        self.neighbors = []
        self.destinations = []


class Environ:

    def __init__(self, down_lane, up_lane, left_lane, right_lane, width, height, n_veh, n_neighbor):
        self.down_lanes = down_lane
        self.up_lanes = up_lane
        self.left_lanes = left_lane
        self.right_lanes = right_lane
        self.width = width
        self.height = height

        self.V2Vchannels = V2Vchannels()
        self.V2Ichannels = V2Ichannels()
        self.vehicles = []

        self.demand = []
        self.V2V_Shadowing = []
        self.V2I_Shadowing = []
        self.delta_distance = []
        self.V2V_channels_abs = []
        self.V2I_channels_abs = []
        self.snr_results_list = []
        self.snr_results_V2I_train_list = []

        self.V2I_power_dB = 60  # dBm
        self.V2V_power_dB_List = [23, 15, 5, -100]

        # self.V2V_power_dB_List = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16,17,18,19,20,21,22,23]  # the power levels
        self.sig2_dB = -114  # dbm
        self.bsAntGain = 8   # dbi
        self.bsNoiseFigure = 5   #db
        self.vehAntGain = 3    #dbi
        self.vehNoiseFigure = 9   #db
        self.sig2 = 10 ** (self.sig2_dB / 10)
        self.u = 20

        self.n_RB = n_veh
        self.n_Veh = n_veh
        self.n_neighbor = n_neighbor
        self.time_fast = 0.001
        self.time_slow = 1  # update slow fading/vehicle position every 100 ms
        self.bandwidth = int(1e6)  # bandwidth per RB, 1 MHz
        # self.bandwidth = 1500
        self.byte_time = 5
        self.demand_size = int((4 * 190 + 300) * self.byte_time) / self.u  # V2V payload: 1060 Bytes every 100 ms
        # self.demand_size = 20

        self.V2V_Interference_all = np.zeros((self.n_Veh, self.n_neighbor, self.n_RB)) + self.sig2
        self.V2V_SNR_all_dB = np.zeros((self.n_Veh, self.n_neighbor, self.n_RB)) + self.sig2

        self.V2V_symbols_of_word_for_train = np.ones((len(self.vehicles)))
        self.V2I_symbols_of_word = np.ones((len(self.vehicles)))


        self.snr_V2I_results = np.full((n_veh, 1), -30)

    def add_new_vehicles(self, start_position, start_direction, start_velocity):
        self.vehicles.append(Vehicle(start_position, start_direction, start_velocity))

    def add_new_vehicles_by_number(self, n):

        for i in range(n):
            ind = np.random.randint(0, len(self.down_lanes))
            start_position = [self.down_lanes[ind], np.random.randint(0, self.height)]
            start_direction = 'd' # velocity: 10 ~ 15 m/s, random
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))

            start_position = [self.up_lanes[ind], np.random.randint(0, self.height)]
            start_direction = 'u'
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))
            start_position = [np.random.randint(0, self.width), self.left_lanes[ind]]
            start_direction = 'l'
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))
            start_position = [np.random.randint(0, self.width), self.right_lanes[ind]]
            start_direction = 'r'
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))

        # initialize channels
        self.V2V_Shadowing = np.random.normal(0, 3, [len(self.vehicles), len(self.vehicles)])
        self.V2I_Shadowing = np.random.normal(0, 8, len(self.vehicles))
        self.delta_distance = np.asarray([c.velocity*self.time_slow for c in self.vehicles])

    def renew_positions(self):
        # ===============
        # This function updates the position of each vehicle
        # ===============

        i = 0
        while (i < len(self.vehicles)):
            delta_distance = self.vehicles[i].velocity * self.time_slow
            change_direction = False
            if self.vehicles[i].direction == 'u':
                # print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):

                    if (self.vehicles[i].position[1] <= self.left_lanes[j]) and ((self.vehicles[i].position[1] + delta_distance) >= self.left_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (delta_distance - (self.left_lanes[j] - self.vehicles[i].position[1])), self.left_lanes[j]]
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if change_direction == False:
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] <= self.right_lanes[j]) and ((self.vehicles[i].position[1] + delta_distance) >= self.right_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (delta_distance + (self.right_lanes[j] - self.vehicles[i].position[1])), self.right_lanes[j]]
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if change_direction == False:
                    self.vehicles[i].position[1] += delta_distance
            if (self.vehicles[i].direction == 'd') and (change_direction == False):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):
                    if (self.vehicles[i].position[1] >= self.left_lanes[j]) and ((self.vehicles[i].position[1] - delta_distance) <= self.left_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (delta_distance - (self.vehicles[i].position[1] - self.left_lanes[j])), self.left_lanes[j]]
                            # print ('down with left', self.vehicles[i].position)
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if change_direction == False:
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] >= self.right_lanes[j]) and (self.vehicles[i].position[1] - delta_distance <= self.right_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (delta_distance + (self.vehicles[i].position[1] - self.right_lanes[j])), self.right_lanes[j]]
                                # print ('down with right', self.vehicles[i].position)
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if change_direction == False:
                    self.vehicles[i].position[1] -= delta_distance
            if (self.vehicles[i].direction == 'r') and (change_direction == False):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.up_lanes)):
                    if (self.vehicles[i].position[0] <= self.up_lanes[j]) and ((self.vehicles[i].position[0] + delta_distance) >= self.up_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (delta_distance - (self.up_lanes[j] - self.vehicles[i].position[0]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if change_direction == False:
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] <= self.down_lanes[j]) and ((self.vehicles[i].position[0] + delta_distance) >= self.down_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (delta_distance - (self.down_lanes[j] - self.vehicles[i].position[0]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                if change_direction == False:
                    self.vehicles[i].position[0] += delta_distance
            if (self.vehicles[i].direction == 'l') and (change_direction == False):
                for j in range(len(self.up_lanes)):

                    if (self.vehicles[i].position[0] >= self.up_lanes[j]) and ((self.vehicles[i].position[0] - delta_distance) <= self.up_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (delta_distance - (self.vehicles[i].position[0] - self.up_lanes[j]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if change_direction == False:
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] >= self.down_lanes[j]) and ((self.vehicles[i].position[0] - delta_distance) <= self.down_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (delta_distance - (self.vehicles[i].position[0] - self.down_lanes[j]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                    if change_direction == False:
                        self.vehicles[i].position[0] -= delta_distance

            # if it comes to an exit
            if (self.vehicles[i].position[0] < 0) or (self.vehicles[i].position[1] < 0) or (self.vehicles[i].position[0] > self.width) or (self.vehicles[i].position[1] > self.height):
                # delete
                #    print ('delete ', self.position[i])
                if (self.vehicles[i].direction == 'u'):
                    self.vehicles[i].direction = 'r'
                    self.vehicles[i].position = [self.vehicles[i].position[0], self.right_lanes[-1]]
                else:
                    if (self.vehicles[i].direction == 'd'):
                        self.vehicles[i].direction = 'l'
                        self.vehicles[i].position = [self.vehicles[i].position[0], self.left_lanes[0]]
                    else:
                        if (self.vehicles[i].direction == 'l'):
                            self.vehicles[i].direction = 'u'
                            self.vehicles[i].position = [self.up_lanes[0], self.vehicles[i].position[1]]
                        else:
                            if (self.vehicles[i].direction == 'r'):
                                self.vehicles[i].direction = 'd'
                                self.vehicles[i].position = [self.down_lanes[-1], self.vehicles[i].position[1]]

            i += 1

    def renew_neighbor(self):
        """ Determine the neighbors of each vehicles """

        for i in range(len(self.vehicles)):
            self.vehicles[i].neighbors = []
            self.vehicles[i].actions = []
        z = np.array([[complex(c.position[0], c.position[1]) for c in self.vehicles]])
        Distance = abs(z.T - z)

        for i in range(len(self.vehicles)):
            sort_idx = np.argsort(Distance[:, i])  # 表示第i辆车辆与其他车辆的距离 对距离矩阵的第i列进行排序，可以确定与第i辆车辆距离最近的车辆的索引排列 从最近的邻居到最远的邻居
            for j in range(self.n_neighbor):
                self.vehicles[i].neighbors.append(sort_idx[j + 1]) # 存储了当前车辆的邻居索引，按照与当前车辆距离的从近到远的顺序排列
            destination = self.vehicles[i].neighbors

            self.vehicles[i].destinations = destination

    def renew_channel(self):
        """ Renew slow fading channel """
# 过np.identity(len(self.vehicles))创建了一个大小为(len(self.vehicles), len(self.vehicles))的单位矩阵。单位矩阵的对角线元素为1，其余元素为0
# 对角线元素表示车辆与自身的路径损耗，设置为50，表示较高的路径损耗
        self.V2V_pathloss = np.zeros((len(self.vehicles), len(self.vehicles))) + 50 * np.identity(len(self.vehicles))
        self.V2I_pathloss = np.zeros((len(self.vehicles)))

        self.V2V_channels_abs = np.zeros((len(self.vehicles), len(self.vehicles)))
        self.V2I_channels_abs = np.zeros((len(self.vehicles)))
        for i in range(len(self.vehicles)):
            for j in range(i + 1, len(self.vehicles)):
                self.V2V_Shadowing[j][i] = self.V2V_Shadowing[i][j] = self.V2Vchannels.get_shadowing(self.delta_distance[i] + self.delta_distance[j], self.V2V_Shadowing[i][j])
                self.V2V_pathloss[j,i] = self.V2V_pathloss[i][j] = self.V2Vchannels.get_path_loss(self.vehicles[i].position, self.vehicles[j].position)

        self.V2V_channels_abs = self.V2V_pathloss + self.V2V_Shadowing

        self.V2I_Shadowing = self.V2Ichannels.get_shadowing(self.delta_distance, self.V2I_Shadowing)
        for i in range(len(self.vehicles)):
            self.V2I_pathloss[i] = self.V2Ichannels.get_path_loss(self.vehicles[i].position)

        self.V2I_channels_abs = self.V2I_pathloss + self.V2I_Shadowing

    def renew_channels_fastfading(self):
        """ Renew fast fading channel """

        # self.V2V_channels_abs是一个二维数组，表示车对车通信的信道增益。通过在第三个轴上插入一个新的维度（np.newaxis），我们可以将其转换为三维数组，使得第三个轴表示资源块（RB）的数量
        V2V_channels_with_fastfading = np.repeat(self.V2V_channels_abs[:, :, np.newaxis], self.n_RB, axis=2) # 通过将self.V2V_channels_abs沿第三个轴（axis=2）重复self.n_RB次得到的数组
        self.V2V_channels_with_fastfading = V2V_channels_with_fastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 1, V2V_channels_with_fastfading.shape) + 1j * np.random.normal(0, 1, V2V_channels_with_fastfading.shape)) / math.sqrt(2))

        V2I_channels_with_fastfading = np.repeat(self.V2I_channels_abs[:, np.newaxis], self.n_RB, axis=1)
        self.V2I_channels_with_fastfading = V2I_channels_with_fastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 1, V2I_channels_with_fastfading.shape) + 1j * np.random.normal(0, 1, V2I_channels_with_fastfading.shape))/ math.sqrt(2))


    def rennew_symbols_of_word(self, action_length):
        """ Renew symbols of every word """

        self.V2I_symbols_of_word = action_length[:, 1]
        self.V2V_symbols_of_word_for_train = action_length[:, 0]
        # 输出结果
        # print("每辆车V2V选择的句子长度列表：", self.V2V_symbols_of_word)
        # print("每辆车V2I选择的句子长度列表：", self.V2V_symbols_of_word)




    def Compute_Performance_Reward_Train(self, actions_power):

        actions = actions_power[:, :, 0]  # the channel_selection_part
        power_selection = actions_power[:, :, 1]  # power selection

        # ------------ Compute V2I rate --------------------
        V2I_Rate = np.zeros(self.n_RB)
        V2I_Interference = np.zeros(self.n_RB)  # V2I interference
        for i in range(len(self.vehicles)):
            for j in range(self.n_neighbor):
                if not self.active_links[i, j]:
                    continue
                V2I_Interference[actions[i][j]] += 10 ** ((self.V2V_power_dB_List[power_selection[i, j]] - self.V2I_channels_with_fastfading[i, actions[i, j]]
                                                           + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        self.V2I_Interference = V2I_Interference + self.sig2
        V2I_Signals = 10 ** ((self.V2I_power_dB - self.V2I_channels_with_fastfading.diagonal() + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)

        # 计算每个信号的信噪比
        snr_results_V2I_train = calculate_snr_V2I(V2I_Signals, self.V2I_Interference)
        # 将列表转换为NumPy数组
        snr_results_V2I_train = np.array(snr_results_V2I_train)
        # 将数组元素类型转换为整数类型
        snr_results_V2I_train = snr_results_V2I_train.astype(int)

        self.snr_V2I_results = np.array(snr_results_V2I_train).reshape(self.n_Veh, 1)
        self.snr_V2I_results = limit_array_range(self.snr_V2I_results, -10, 20)

        semantic_similarity_V2I = []  # 存储每辆车的语义相似度

        # 遍历每辆车的句子长度和信噪比
        for length, snr in zip(self.V2I_symbols_of_word, self.snr_V2I_results):
            row_index = length - 1  # 由于索引从0开始，所以需要减1
            col_index = snr + 10  # 由于snr_results在-60到60之间，需要加60以匹配table_data的列索引
            # 获取对应的语义相似性值，并添加到列表中
            semantic_similarity_V2I.append(table_data[row_index, col_index])
            # print("Current SNR:", snr, "Current Length:", length, "Semantic Similarity:",
            #       table_data[row_index, col_index])

        # 将列表转换为 NumPy 数组，并调整形状为 (4, 1)
        semantic_similarity_V2I = np.array(semantic_similarity_V2I).reshape((self.n_Veh, 1))
        V2I_symbols_new = self.V2I_symbols_of_word.reshape((self.n_Veh, 1))

        # 计算 V2V_SC_SSE，即相除得到的结果
        V2I_SC_SSE = semantic_similarity_V2I / V2I_symbols_new

        # 打印结果
        # print("V2V_SC_SSE:", V2I_SC_SSE)


        V2I_Rate = np.log2(1 + np.divide(V2I_Signals, self.V2I_Interference))
        # print('训练过程中V2I速率', V2I_Rate)


        # ------------ Compute V2V rate -------------------------
        V2V_Interference = np.zeros((len(self.vehicles), self.n_neighbor))
        V2V_Signal = np.zeros((len(self.vehicles), self.n_neighbor))
        actions[(np.logical_not(
            self.active_links))] = -1  # inactive links will not transmit regardless of selected power levels
        for i in range(self.n_RB):  # scanning all bands
            indexes = np.argwhere(actions == i)  # find spectrum-sharing V2Vs
            for j in range(len(indexes)):
                receiver_j = self.vehicles[indexes[j, 0]].destinations[indexes[j, 1]]
                V2V_Signal[indexes[j, 0], indexes[j, 1]] = 10 ** (
                            (self.V2V_power_dB_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                             - self.V2V_channels_with_fastfading[
                                 indexes[j][0], receiver_j, i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                # V2I links interference to V2V links
                V2V_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.V2I_power_dB -
                                                                          self.V2V_channels_with_fastfading[
                                                                              i, receiver_j, i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)

                #  V2V interference
                for k in range(j + 1, len(indexes)):  # spectrum-sharing V2Vs
                    receiver_k = self.vehicles[indexes[k][0]].destinations[indexes[k][1]]
                    V2V_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** (
                                (self.V2V_power_dB_List[power_selection[indexes[k, 0], indexes[k, 1]]]
                                 - self.V2V_channels_with_fastfading[indexes[k][0]][receiver_j][
                                     i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                    V2V_Interference[indexes[k, 0], indexes[k, 1]] += 10 ** (
                                (self.V2V_power_dB_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                                 - self.V2V_channels_with_fastfading[indexes[j][0]][receiver_k][
                                     i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
        self.V2V_Interference = V2V_Interference + self.sig2
        # V2V_Interference_train = 10 * np.log10(self.V2V_Interference)
        # V2V_Signal_train = 10 * np.log10(V2V_Signal)
        # print('训练过程中的V2V干扰', V2V_Interference_train)
        # print('训练过程的V2V信号', V2V_Signal_train)

        # 计算每个信号的信噪比
        snr_results_V2V_train = calculate_snr_V2V(V2V_Signal, self.V2V_Interference)
        # 将列表转换为NumPy数组
        snr_results_V2V_train = np.array(snr_results_V2V_train)
        # 将数组元素类型转换为整数类型
        snr_results_V2V_train = snr_results_V2V_train.astype(int)

        self.snr_V2V_results = np.array(snr_results_V2V_train).reshape(self.n_Veh, 1)
        self.snr_V2V_results = limit_array_range(self.snr_V2V_results, -10, 20)

        semantic_similarity_V2V = []  # 存储每辆车的语义相似度

        # 遍历每辆车的句子长度和信噪比
        for length, snr in zip(self.V2V_symbols_of_word_for_train, self.snr_V2V_results):
            row_index = length - 1  # 由于索引从0开始，所以需要减1
            col_index = snr + 10  # 由于snr_results在-60到60之间，需要加60以匹配table_data的列索引
            # 获取对应的语义相似性值，并添加到列表中
            semantic_similarity_V2V.append(table_data[row_index, col_index])
            # print("Current SNR:", snr, "Current Length:", length, "Semantic Similarity:",
            #       table_data[row_index, col_index])

        # 将列表转换为 NumPy 数组，并调整形状为 (4, 1)
        semantic_similarity_V2V = np.array(semantic_similarity_V2V).reshape((self.n_Veh, 1))
        V2V_symbols_new = self.V2V_symbols_of_word_for_train.reshape((self.n_Veh, 1))

        # 计算 V2V_SC_SSE，即相除得到的结果
        V2V_SC_SSE = semantic_similarity_V2V / V2V_symbols_new

        # 打印结果
        # print("V2V_SC_SSE:", V2V_SC_SSE)





        V2V_Rate = np.log2(1 + np.divide(V2V_Signal, self.V2V_Interference))
        # print('训练过程中V2V速率',V2V_Rate)


        # self.demand -= V2V_Rate * self.time_fast * self.bandwidth
        self.demand -= V2V_SC_SSE * self.time_fast * self.bandwidth
        self.demand[self.demand < 0] = 0 # eliminate negative demands

        self.individual_time_limit -= self.time_fast

        # reward_elements = V2V_Rate/10
        reward_elements = V2V_SC_SSE/1000
        reward_elements[self.demand <= 0] = 1

        self.active_links[np.multiply(self.active_links, self.demand <= 0)] = 0 # transmission finished, turned to "inactive"

        # return V2I_Rate, V2V_Rate, reward_elements
        return V2I_SC_SSE, V2V_SC_SSE, reward_elements

    def Compute_Performance_Reward_Test_rand(self, actions_power):
        """ for random baseline computation """

        actions = actions_power[:, :, 0]  # the channel_selection_part
        power_selection = actions_power[:, :, 1]  # power selection

        # ------------ Compute V2I rate --------------------
        V2I_Rate = np.zeros(self.n_RB)
        V2I_Interference = np.zeros(self.n_RB)  # V2I interference
        for i in range(len(self.vehicles)):
            for j in range(self.n_neighbor):
                if not self.active_links_rand[i, j]:
                    continue
                V2I_Interference[actions[i][j]] += 10 ** ((self.V2V_power_dB_List[power_selection[i, j]] - self.V2I_channels_with_fastfading[i, actions[i, j]]
                                                           + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        self.V2I_Interference_random = V2I_Interference + self.sig2
        V2I_Signals = 10 ** ((self.V2I_power_dB - self.V2I_channels_with_fastfading.diagonal() + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        V2I_Rate = np.log2(1 + np.divide(V2I_Signals, self.V2I_Interference_random))

        # ------------ Compute V2V rate -------------------------
        V2V_Interference = np.zeros((len(self.vehicles), self.n_neighbor))
        V2V_Signal = np.zeros((len(self.vehicles), self.n_neighbor))
        actions[(np.logical_not(self.active_links_rand))] = -1
        for i in range(self.n_RB):  # scanning all bands
            indexes = np.argwhere(actions == i)  # find spectrum-sharing V2Vs
            for j in range(len(indexes)):
                receiver_j = self.vehicles[indexes[j, 0]].destinations[indexes[j, 1]]
                V2V_Signal[indexes[j, 0], indexes[j, 1]] = 10 ** ((self.V2V_power_dB_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                                                                   - self.V2V_channels_with_fastfading[indexes[j][0], receiver_j, i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                # V2I links interference to V2V links
                V2V_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.V2I_power_dB - self.V2V_channels_with_fastfading[i, receiver_j, i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)

                #  V2V interference
                for k in range(j + 1, len(indexes)):  # spectrum-sharing V2Vs
                    receiver_k = self.vehicles[indexes[k][0]].destinations[indexes[k][1]]
                    V2V_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.V2V_power_dB_List[power_selection[indexes[k, 0], indexes[k, 1]]]
                                                                              - self.V2V_channels_with_fastfading[indexes[k][0]][receiver_j][i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                    V2V_Interference[indexes[k, 0], indexes[k, 1]] += 10 ** ((self.V2V_power_dB_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                                                                              - self.V2V_channels_with_fastfading[indexes[j][0]][receiver_k][i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
        self.V2V_Interference_random = V2V_Interference + self.sig2
        V2V_Rate = np.log2(1 + np.divide(V2V_Signal, self.V2V_Interference_random))

        self.demand_rand -= V2V_Rate * self.time_fast * self.bandwidth
        self.demand_rand[self.demand_rand < 0] = 0

        self.individual_time_limit_rand -= self.time_fast

        self.active_links_rand[np.multiply(self.active_links_rand, self.demand_rand <= 0)] = 0 # transmission finished, turned to "inactive"

        return V2I_Rate, V2V_Rate


    def Compute_Interference(self, actions):
        V2V_Interference = np.zeros((len(self.vehicles), self.n_neighbor, self.n_RB)) + self.sig2

        channel_selection = actions.copy()[:, :, 0]
        power_selection = actions.copy()[:, :, 1]
        channel_selection[np.logical_not(self.active_links)] = -1

        # interference from V2I links
        for i in range(self.n_RB):
            for k in range(len(self.vehicles)):
                for m in range(len(channel_selection[k, :])):
                    V2V_Interference[k, m, i] += 10 ** ((self.V2I_power_dB - self.V2V_channels_with_fastfading[i][self.vehicles[k].destinations[m]][i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)

        # interference from peer V2V links
        for i in range(len(self.vehicles)):
            for j in range(len(channel_selection[i, :])):
                for k in range(len(self.vehicles)):
                    for m in range(len(channel_selection[k, :])):
                        # if i == k or channel_selection[i,j] >= 0:
                        if i == k and j == m or channel_selection[i, j] < 0:
                            continue
                        V2V_Interference[k, m, channel_selection[i, j]] += 10 ** ((self.V2V_power_dB_List[power_selection[i, j]]
                                                                                   - self.V2V_channels_with_fastfading[i][self.vehicles[k].destinations[m]][channel_selection[i,j]] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
        self.V2V_Interference_all = 10 * np.log10(V2V_Interference)
        # print('V2V干扰', self.V2V_Interference_all)

    def compute_V2V_SNR(self, actions):

        V2V_Signal = np.zeros((len(self.V2V_channels_with_fastfading), 1, len(self.V2V_channels_with_fastfading[0][0])))

        channel_selection = actions.copy()[:, :, 0]
        power_selection = actions.copy()[:, :, 1]
        channel_selection[np.logical_not(self.active_links)] = -1

        for i in range(len(self.V2V_channels_with_fastfading[0][0])):
            for k in range(len(self.V2V_channels_with_fastfading)):
                for m in range(len(channel_selection[k, :])):
                    V2V_Signal[k, m, i] = 10 ** (
                            (self.V2V_power_dB_List[power_selection[k, m]] -
                             self.V2V_channels_with_fastfading[k][self.vehicles[k].destinations[m]][channel_selection[k,m]] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
        self.V2V_Signal_dB = 10 * np.log10(V2V_Signal)
        # print('V2V信号', self.V2V_Signal_dB)
        self.V2V_SNR_all_dB = self.V2V_Signal_dB - self.V2V_Interference_all
        self.V2V_SNR_all_dB = limit_array_range(self.V2V_SNR_all_dB, -10, 20)
        # print('V2V信噪比', self.V2V_SNR_all_dB)





    def act_for_training(self, actions):

        action_temp = actions.copy()
        V2I_SC_SSE, V2V_SC_SSE, reward_elements = self.Compute_Performance_Reward_Train(action_temp)

        lamda = 0.9


        '''# 判断第二部分是否等于1，并输出提示信息
        if np.sum(reward_elements) / (self.n_Veh * self.n_neighbor) == 1:
            print("第二部分的值等于1！")'''

        reward = lamda * np.sum(V2I_SC_SSE) / (self.n_Veh * 0.1)+ (1-lamda) * np.sum(reward_elements) / (self.n_Veh * self.n_neighbor)

        # print('第一部分奖励' , np.sum(V2I_SC_SSE) / (self.n_Veh* 0.1))
        # print('第二部分奖励' ,np.sum(reward_elements) / (self.n_Veh * self.n_neighbor))

        return reward

    def act_for_testing(self, actions):

        action_temp = actions.copy()
        V2I_SC_SSE, V2V_SC_SSE, _ = self.Compute_Performance_Reward_Train(action_temp)

        lamda = 0.9
        V2V_success = 1 - np.sum(self.active_links) / (self.n_Veh * self.n_neighbor)
        return V2I_SC_SSE, V2V_success, V2V_SC_SSE,


    def act_for_testing_rand(self, actions):

        action_temp = actions.copy()
        V2I_Rate, V2V_Rate = self.Compute_Performance_Reward_Test_rand(action_temp)
        V2V_success = 1 - np.sum(self.active_links_rand) / (self.n_Veh * self.n_neighbor)  # V2V success rates

        return V2I_Rate, V2V_success, V2V_Rate

    def new_random_game(self, n_Veh=0):
        # make a new game

        self.vehicles = []
        if n_Veh > 0:
            self.n_Veh = n_Veh
        self.add_new_vehicles_by_number(int(self.n_Veh / 4))
        self.renew_neighbor()
        self.renew_channel()
        self.renew_channels_fastfading()

        self.demand = self.demand_size * np.ones((self.n_Veh, self.n_neighbor))
        self.individual_time_limit = self.time_slow * np.ones((self.n_Veh, self.n_neighbor))
        self.active_links = np.ones((self.n_Veh, self.n_neighbor), dtype='bool')
        self.V2I_symbols_of_word = np.ones(len(self.vehicles), dtype=int)
        self.V2V_symbols_of_word_for_train = np.ones(len(self.vehicles), dtype=int)

        # random baseline
        self.demand_rand = self.demand_size * np.ones((self.n_Veh, self.n_neighbor))
        self.individual_time_limit_rand = self.time_slow * np.ones((self.n_Veh, self.n_neighbor))
        self.active_links_rand = np.ones((self.n_Veh, self.n_neighbor), dtype='bool')



