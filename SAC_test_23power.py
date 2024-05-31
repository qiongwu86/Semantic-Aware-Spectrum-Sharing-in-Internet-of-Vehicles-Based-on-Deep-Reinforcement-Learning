import numpy as np
import os
import scipy.io
import Environment_marl_V2I_and_V2V_SC_for_SAC_initial_23
from RL_SAC import SAC_Trainer
from  RL_SAC import ReplayBuffer
import matplotlib.pyplot as plt


def get_state(env, idx=(0,0), ind_episode=1., epsi=0.02):
    """ Get state from the environment """

    # V2I_channel = (env.V2I_channels_with_fastfading[idx[0], :] - 80) / 60
    V2I_fast = (env.V2I_channels_with_fastfading[idx[0], :] - env.V2I_channels_abs[idx[0]] + 10)/35

    # V2V_channel = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - 80) / 60
    V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] + 10)/35

    V2V_interference = (-env.V2V_Interference_all[idx[0], idx[1], :] - 60) / 60
    # print('状态里的V2V干扰', env.V2V_Interference_all[idx[0], idx[1], :])

    V2V_SNR = env.V2V_SNR_all_dB[idx[0], idx[1], :]
    V2V_SNR_normalized = (V2V_SNR + 10) / 30
    # print('V2V_SNR_normalized' , V2V_SNR_normalized)
    V2V_symbol = env.V2V_symbols_of_word_for_train[idx[0]] / 20
    V2I_symbol = env.V2I_symbols_of_word[idx[0]] / 20

    V2I_abs = (env.V2I_channels_abs[idx[0]] - 80) / 60.0
    V2V_abs = (env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] - 80)/60.0

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


label = 'model/sac_model/with_sc_n_veh_8_lamda0.9_lr3e-4_hs_512_bs64_newpowerlist'
model_path = label + '/agent'



n_veh = 8
n_neighbor = 1
n_RB = n_veh
max_power = 23
length = 20

test_env = Environment_marl_V2I_and_V2V_SC_for_SAC_initial_23.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor)
test_env.new_random_game()  # initialize parameters in env


n_episode = 1000
n_step_per_episode = int(test_env.time_slow/test_env.time_fast)

epsi_final = 0.02
epsi_anneal_length = int(0.8*n_episode)

mini_batch_step = n_step_per_episode
target_update_step = n_step_per_episode*4



n_episode_test = 100  # test episodes

######################################################
# ------------------------------------------------------------------------------------------------------------------ #
n_input = len(get_state(env=test_env))*n_RB


n_output = 4 * n_RB

action_range = 1.0
# --------------------------------------------------------------
#agent = SAC_Trainer(alpha, beta, n_input, tau, gamma, 12 ,memory_size, fc1_dims, fc2_dims, fc3_dims, fc4_dims, batch_size, 2, 'OU')
replay_buffer_size = 1e6
batch_size = 64
replay_buffer = ReplayBuffer(replay_buffer_size)
hidden_dim = 512
agent = SAC_Trainer(replay_buffer, n_input, n_output, hidden_dim=hidden_dim, action_range=action_range)


update_itr = 1
AUTO_ENTROPY=True
DETERMINISTIC=False
frame_idx = 0
explore_steps = 0 # for random action sampling in the beginning of training

# ------------------------- Testing -----------------------------
if IS_TEST:
    # Load the trained SAC model
    agent.load_model(model_path)


    SC_SSE_LIST = []
    V2I_SC_SSE_LIST = []
    V2V_SC_SSE_LIST = []
    V2V_success_list = []

    record_reward = np.zeros(n_step_per_episode)
    reward_average_list = []
    record_loss = []
    test_action_all_training = np.zeros([n_veh, n_neighbor, 2], dtype='int32')


    time_step = 0
    for i_episode in range(n_episode_test):
        print("-------------------------")
        print('Episode:', i_episode)


        test_env.renew_positions() # update vehicle position
        test_env.renew_neighbor()
        test_env.renew_channel() # update channel slow fading
        test_env.renew_channels_fastfading()
        test_env.demand = test_env.demand_size * np.ones((test_env.n_Veh, test_env.n_neighbor))
        test_env.individual_time_limit = test_env.time_slow * np.ones((test_env.n_Veh, test_env.n_neighbor))
        test_env.active_links = np.ones((test_env.n_Veh, test_env.n_neighbor), dtype='bool')

        state_old_all = []


        for i in range(n_RB):
            for j in range(n_neighbor):
                state = get_state(test_env, [i, j], i_episode / (n_episode - 1), epsi_final)
                state_old_all.append(state)



        Sum_rate_per_episode = []
        average_reward = 0
        V2I_SC_SSE_per_episode = []
        V2V_SC_SSE_per_episode = []

        for test_step in range(n_step_per_episode):
            done = 0
            state_new_all = []
            action_all = []
            test_action_all_training = np.zeros([n_RB, n_neighbor, 2], dtype=int)  # sub, power
            test_action_all_training_length = np.zeros([n_veh, 2], dtype=int)  # length

            action = agent.policy_net.get_action(np.asarray(state_old_all).flatten(), deterministic=DETERMINISTIC)
            action = np.clip(action, -0.999, 0.999)
            action_all.append(action)

            for i in range(n_RB):
                for j in range(n_neighbor):
                    test_action_all_training[i, j, 0] = ((action[0 + i * 2] + 1) / 2) * n_RB  # chosen RB
                    test_action_all_training[i, j, 1] = ((action[1 + i * 2] + 1) / 2) * n_RB  # power selected by PL
            for i in range(n_RB):
                test_action_all_training_length[i, 0] = 1 + ((action[2 * n_RB + i] + 1) / 2) * length
                test_action_all_training_length[i, 1] = 1 + ((action[2 * n_RB + 1 + i] + 1) / 2) * length
            action_temp = test_action_all_training.copy()
            test_env.rennew_symbols_of_word(test_action_all_training_length)
            V2I_SC_SSE, V2V_success_rate, V2V_SC_SSE = test_env.act_for_testing(action_temp)
            V2I_SC_SSE_per_episode.append(np.sum(V2I_SC_SSE))
            V2V_SC_SSE_per_episode.append(np.sum(V2V_SC_SSE))


            test_env.renew_channels_fastfading()
            test_env.Compute_Interference(action_temp)
            test_env.compute_V2V_SNR(action_temp)

            if test_step == n_step_per_episode - 1:
                V2V_success_list.append(V2V_success_rate)

        V2I_SC_SSE_LIST.append(np.mean(V2I_SC_SSE_per_episode))
        V2V_SC_SSE_LIST.append(np.mean(V2V_SC_SSE_per_episode))
        print('SAC_with_sc_V2I_SC_SSE', round(np.average(V2I_SC_SSE_per_episode), 4))
        print('SAC_with_sc_V2I_SC_SSE', round(np.average(V2V_SC_SSE_per_episode), 4))
        print('SAC_with_sc_V2V_success', V2V_success_list[i_episode])

    # 计算平均值
    average_V2I_SC_SSE = np.mean(V2I_SC_SSE_LIST)
    average_V2V_SC_SSE = np.mean(V2V_SC_SSE_LIST)

    average_V2V_success = np.mean(V2V_success_list)

    # 打印结果
    print('Average V2I_SC_SSE (SAC_with_sc):', round(average_V2I_SC_SSE, 4))
    print('Average V2V_SC_SSE (SAC_with_sc):', round(average_V2V_SC_SSE, 4))

    print('Average V2V_success (SAC_with_sc):', round(average_V2V_success, 4))
    print('the demand size, 1060*', str(test_env.byte_time))


# 打开一个文件以写入模式
with open('result/sac_vehicle_change_u_20_demand_25_results.txt', 'a') as file:
    # 将结果追加到文件
    file.write('Average V2I_SC_SSE (SAC_with_sc): ' + str(round(average_V2I_SC_SSE, 4)) + '\n')
    file.write('Average V2V_SC_SSE (SAC_with_sc): ' + str(round(average_V2V_SC_SSE, 4)) + '\n')
    file.write('Average V2V_success (SAC_with_sc): ' + str(round(average_V2V_success, 4)) + '\n')
    file.write('the u: ' + str(test_env.u) + '\n')
    file.write('the demand size: 1060 * ' + str(test_env.byte_time) + '\n')
    file.write('V2I_power_dB ' + str(test_env.V2I_power_dB) + '\n')


