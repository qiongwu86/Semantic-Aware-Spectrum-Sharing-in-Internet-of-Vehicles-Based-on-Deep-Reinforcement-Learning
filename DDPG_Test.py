import numpy as np
import os
import scipy.io
import Environment_marl_V2I_and_V2V_SC_for_SAC_initial
from RL_DDPG import Agent
from RL_DDPG import ActorNetwork, CriticNetwork
import matplotlib.pyplot as plt

# 其他设置与训练代码相同
def get_state(env, idx=(0,0), ind_episode=1., epsi=0.02):
    """ Get state from the environment """

    # V2I_channel = (env.V2I_channels_with_fastfading[idx[0], :] - 80) / 60
    V2I_fast = (env.V2I_channels_with_fastfading[idx[0], :] - env.V2I_channels_abs[idx[0]] + 10) / 35

    # V2V_channel = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - 80) / 60
    V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]],
                :] - env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] + 10) / 35

    V2V_interference = (-env.V2V_Interference_all[idx[0], idx[1], :] - 60) / 60
    # print('状态里的V2V干扰', env.V2V_Interference_all[idx[0], idx[1], :])

    V2V_SNR = env.V2V_SNR_all_dB[idx[0], idx[1], :]
    V2V_SNR_normalized = (V2V_SNR + 10) / 30
    # print('V2V_SNR_normalized' , V2V_SNR_normalized)
    V2V_symbol = env.V2V_symbols_of_word_for_train[idx[0]] / 20
    V2I_symbol = env.V2I_symbols_of_word[idx[0]] / 20

    V2I_abs = (env.V2I_channels_abs[idx[0]] - 80) / 60.0
    V2V_abs = (env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] - 80) / 60.0

    load_remaining = np.asarray([env.demand[idx[0], idx[1]] / env.demand_size])
    time_remaining = np.asarray([env.individual_time_limit[idx[0], idx[1]] / env.time_slow])

    # return np.concatenate((np.reshape(V2V_channel, -1), V2V_interference, V2I_abs, V2V_abs, time_remaining, load_remaining, np.asarray([ind_episode, epsi])))
    # return np.concatenate((V2I_fast, np.reshape(V2V_fast, -1), V2V_interference, np.asarray([V2I_abs]), V2V_abs, time_remaining, load_remaining, np.asarray([ind_episode, epsi])))
    return np.concatenate((np.reshape(V2I_fast, -1), np.reshape(V2V_fast, -1), V2V_interference, V2V_SNR_normalized, np.asarray([V2I_abs]), V2V_abs, time_remaining, load_remaining, np.reshape(V2V_symbol, -1), np.reshape(V2I_symbol, -1), np.asarray([ind_episode, epsi])))


# ################## SETTINGS ######################
up_lanes = [i/2.0 for i in [3.5/2,3.5/2 + 3.5,250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]]
down_lanes = [i/2.0 for i in [250-3.5-3.5/2,250-3.5/2,500-3.5-3.5/2,500-3.5/2,750-3.5-3.5/2,750-3.5/2]]
left_lanes = [i/2.0 for i in [3.5/2,3.5/2 + 3.5,433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]]
right_lanes = [i/2.0 for i in [433-3.5-3.5/2,433-3.5/2,866-3.5-3.5/2,866-3.5/2,1299-3.5-3.5/2,1299-3.5/2]]

width = 750/2
height = 1298/2

IS_TRAIN = 0
IS_TEST = 1-IS_TRAIN

label = 'model/ddpg_model/newpowerlist_bs32_bata0.001'
model_path = label + '/agent'

n_veh = 4
n_neighbor = 1
n_RB = n_veh
max_power = 23
length = 20
env = Environment_marl_V2I_and_V2V_SC_for_SAC_initial.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor)
env.new_random_game()  # initialize parameters in env

n_episode = 1000
n_step_per_episode = int(env.time_slow/env.time_fast)
epsi_final = 0.02
epsi_anneal_length = int(0.8*n_episode)
mini_batch_step = n_step_per_episode
target_update_step = n_step_per_episode*4

n_episode_test = 100  # test episodes

######################################################
# ------------------------------------------------------------------------------------------------------------------ #
n_input = len(get_state(env=env))
n_output = 4 * n_RB
action_range = 1.0
# --------------------------------------------------------------
## Initializations ##
# ------- characteristics related to the network -------- #
batch_size = 32
memory_size = 1000000
gamma = 0.99
alpha = 0.001
beta = 0.001
# actor and critic hidden layers
fc1_dims = 512
fc2_dims = 512
fc3_dims = 512
fc4_dims = 512
# ------------------------------

tau = 0.005#参数更新权重
agent = Agent(alpha, beta, n_input, tau, n_output, gamma, memory_size, fc1_dims, fc2_dims, fc3_dims, fc4_dims, batch_size, n_veh)
actor = ActorNetwork(alpha, n_input, fc1_dims, fc2_dims, fc3_dims, fc4_dims, n_veh, n_output, 'actor')
critic = CriticNetwork(beta, n_input, fc1_dims, fc2_dims, fc3_dims, fc4_dims, n_veh, n_output, 'critic')


# ------------------------- Testing -----------------------------
if IS_TEST:
    # Load the trained SAC model
    # agent.load_models()
    # Load the trained SAC model
    actor.load_checkpoint()
    critic.load_checkpoint()

    SC_SSE_LIST = []
    V2I_SC_SSE_LIST = []
    V2V_SC_SSE_LIST = []
    V2V_success_list = []


    record_reward = np.zeros(n_step_per_episode)
    n_episode = n_episode_test  # 设置测试的总episode数
    reward_test_list = []  # 存储测试结果的列表
    test_action_all_training = np.zeros([n_veh, n_neighbor, 2], dtype='int32')

    for i_episode in range(n_episode):
        print("-------------------------")
        print('Testing Episode:', i_episode)
        epsi = 0.02  # 在测试期间不需要epsilon贪心策略


        env.renew_positions()  # 更新车辆位置
        env.renew_neighbor()
        env.renew_channel()  # 更新信道的慢衰落
        env.renew_channels_fastfading()

        env.demand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
        env.individual_time_limit = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
        env.active_links = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

        state_old_all = []
        for i in range(n_RB):
            for j in range(n_neighbor):
                state = get_state(env, [i, j], i_episode / (n_episode - 1), epsi)
                state_old_all.append(state)

        Sum_rate_per_episode = []
        average_reward = 0
        V2I_SC_SSE_per_episode = []
        V2V_SC_SSE_per_episode = []

        for i_step in range(n_step_per_episode):
            done = 0
            state_new_all = []
            action_all = []

            test_action_all_training = np.zeros([n_RB, n_neighbor, 2], dtype=int)  # 子载波，功率
            test_action_all_training_length = np.zeros([n_veh, 2], dtype=int)  # 长度

            action = agent.choose_action(np.asarray(state_old_all).flatten())  # 使用智能体的策略选择动作
            action = np.clip(action, -0.999, 0.999)
            action_all.append(action)

            for i in range(n_RB):
                for j in range(n_neighbor):
                    test_action_all_training[i, j, 0] = ((action[0 + i * 2] + 1) / 2) * n_RB  # 选择的RB
                    test_action_all_training[i, j, 1] = ((action[1 + i * 2] + 1) / 2) * n_RB  # 由PL选择的功率
            for i in range(n_RB):
                test_action_all_training_length[i, 0] = 1 + ((action[2 * n_RB + i] + 1) / 2) * length
                test_action_all_training_length[i, 1] = 1 + ((action[2 * n_RB + 1 + i] + 1) / 2) * length
            action_temp = test_action_all_training.copy()
            env.rennew_symbols_of_word(test_action_all_training_length)
            V2I_SC_SSE, V2V_success_rate, V2V_SC_SSE = env.act_for_testing(action_temp)
            V2I_SC_SSE_per_episode.append(np.sum(V2I_SC_SSE))
            V2V_SC_SSE_per_episode.append(np.sum(V2V_SC_SSE))

            env.renew_channels_fastfading()
            env.Compute_Interference(action_temp)
            env.compute_V2V_SNR(action_temp)

            if i_step == n_step_per_episode - 1:
                V2V_success_list.append(V2V_success_rate)

        V2I_SC_SSE_LIST.append(np.mean(V2I_SC_SSE_per_episode))
        V2V_SC_SSE_LIST.append(np.mean(V2V_SC_SSE_per_episode))
        print('DDPG_with_sc_V2I_SC_SSE', round(np.average(V2I_SC_SSE_per_episode), 4))
        print('DDPG_with_sc_V2I_SC_SSE', round(np.average(V2V_SC_SSE_per_episode), 4))
        print('DDPG_with_sc_V2V_success', V2V_success_list[i_episode])

    # 计算平均值
    average_V2I_SC_SSE = np.mean(V2I_SC_SSE_LIST)
    average_V2V_SC_SSE = np.mean(V2V_SC_SSE_LIST)

    average_V2V_success = np.mean(V2V_success_list)
    # 打印结果
    print('Average V2I_SC_SSE (DDPG_with_sc):', round(average_V2I_SC_SSE, 4))
    print('Average V2V_SC_SSE (DDPG_with_sc):', round(average_V2V_SC_SSE, 4))

    print('Average V2V_success (DDPG_with_sc):', round(average_V2V_success, 4))

    print('the demand size, 1060*', str(env.byte_time))
    print('V2I_power_dB ', str(env.V2I_power_dB) )

# 打开一个文件以写入模式
with open('result/ddpg_vehicle4_V2Ipower_change_u_20_demand_25_results.txt', 'a') as file:
    # 将结果追加到文件
    file.write('Average V2I_SC_SSE (DDPG_with_sc): ' + str(round(average_V2I_SC_SSE, 4)) + '\n')
    file.write('Average V2V_SC_SSE (DDPG_with_sc): ' + str(round(average_V2V_SC_SSE, 4)) + '\n')
    file.write('Average V2V_success (DDPG_with_sc): ' + str(round(average_V2V_success, 4)) + '\n')
    file.write('the u: ' + str(env.u) + '\n')
    file.write('the demand size: 1060 * ' + str(env.byte_time) + '\n')
    file.write('V2I_power_dB ' + str(env.V2I_power_dB) + '\n')

