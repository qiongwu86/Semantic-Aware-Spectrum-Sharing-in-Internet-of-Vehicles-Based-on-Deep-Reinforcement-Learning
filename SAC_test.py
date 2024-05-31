import numpy as np
import os
import scipy.io
import Environment_marl_V2I_and_V2V_SC_for_SAC_initial
import Environment_marl_initial_for_sac
from RL_SAC import SAC_Trainer
from  RL_SAC import ReplayBuffer
import matplotlib.pyplot as plt


def get_state1(env, idx=(0,0), ind_episode=1., epsi=0.02):
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

def get_state2(env, idx=(0,0), ind_episode=1., epsi=0.02):
    """ Get state from the environment """

    # V2I_channel = (env.V2I_channels_with_fastfading[idx[0], :] - 80) / 60
    V2I_fast = (env.V2I_channels_with_fastfading[idx[0], :] - env.V2I_channels_abs[idx[0]] + 10)/35

    # V2V_channel = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - 80) / 60
    V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] + 10)/35

    V2V_interference = (-env.V2V_Interference_all[idx[0], idx[1], :] - 60) / 60
    # print('状态里的V2V干扰', env.V2V_Interference_all[idx[0], idx[1], :])

    V2I_abs = (env.V2I_channels_abs[idx[0]] - 80) / 60.0
    V2V_abs = (env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] - 80)/60.0

    load_remaining = np.asarray([env.demand[idx[0], idx[1]] / env.demand_size])
    time_remaining = np.asarray([env.individual_time_limit[idx[0], idx[1]] / env.time_slow])

    # return np.concatenate((np.reshape(V2V_channel, -1), V2V_interference, V2I_abs, V2V_abs, time_remaining, load_remaining, np.asarray([ind_episode, epsi])))
    # return np.concatenate((V2I_fast, np.reshape(V2V_fast, -1), V2V_interference, np.asarray([V2I_abs]), V2V_abs, time_remaining, load_remaining, np.asarray([ind_episode, epsi])))
    return np.concatenate((np.reshape(V2I_fast, -1), np.reshape(V2V_fast, -1), V2V_interference,  np.asarray([V2I_abs]), V2V_abs, time_remaining, load_remaining,  np.asarray([ind_episode, epsi])))

# ################## SETTINGS ######################
up_lanes = [i/2.0 for i in [3.5/2,3.5/2 + 3.5,250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]]
down_lanes = [i/2.0 for i in [250-3.5-3.5/2,250-3.5/2,500-3.5-3.5/2,500-3.5/2,750-3.5-3.5/2,750-3.5/2]]
left_lanes = [i/2.0 for i in [3.5/2,3.5/2 + 3.5,433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]]
right_lanes = [i/2.0 for i in [433-3.5-3.5/2,433-3.5/2,866-3.5-3.5/2,866-3.5/2,1299-3.5-3.5/2,1299-3.5/2]]

width = 750/2
height = 1298/2

IS_TRAIN = 0
IS_TEST = 1-IS_TRAIN


label1 = 'model/sac_model/with_sc_lamda0.9_lr3e-4_hs_512_bs64_newpowerlist'
model_path1 = label1 + '/agent'
label2 = 'model/sac_model/without_sc/lamda0_lr3e-4_hs_512_bs64_newpowerlist_done'
model_path2 = label2 + '/agent'



n_veh = 4
n_neighbor = 1
n_RB = n_veh
max_power = 23
length = 20

test_env1 = Environment_marl_V2I_and_V2V_SC_for_SAC_initial.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor)
test_env1.new_random_game()  # initialize parameters in env

test_env2 = Environment_marl_initial_for_sac.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor)
test_env2.new_random_game()

n_episode = 1000
n_step_per_episode1 = int(test_env1.time_slow/test_env1.time_fast)
n_step_per_episode2 = int(test_env2.time_slow/test_env2.time_fast)

epsi_final = 0.02
epsi_anneal_length = int(0.8*n_episode)

mini_batch_step1 = n_step_per_episode1
target_update_step1 = n_step_per_episode1*4

mini_batch_step2 = n_step_per_episode1
target_update_step2 = n_step_per_episode2*4

n_episode_test = 100  # test episodes

######################################################
# ------------------------------------------------------------------------------------------------------------------ #
n_input1 = len(get_state1(env=test_env1))*n_RB
n_input2 = len(get_state2(env=test_env2))*n_RB

n_output1 = 4 * n_RB
n_output2 = 2 * n_RB

action_range = 1.0
# --------------------------------------------------------------
#agent = SAC_Trainer(alpha, beta, n_input, tau, gamma, 12 ,memory_size, fc1_dims, fc2_dims, fc3_dims, fc4_dims, batch_size, 2, 'OU')
replay_buffer_size = 1e6
batch_size = 64
replay_buffer = ReplayBuffer(replay_buffer_size)
hidden_dim = 512
agent1 = SAC_Trainer(replay_buffer, n_input1, n_output1, hidden_dim=hidden_dim, action_range=action_range)
agent2 = SAC_Trainer(replay_buffer, n_input2, n_output2, hidden_dim=hidden_dim, action_range=action_range)



update_itr = 1
AUTO_ENTROPY=True
DETERMINISTIC=False
frame_idx = 0
explore_steps = 0 # for random action sampling in the beginning of training

# ------------------------- Testing -----------------------------
if IS_TEST:
    # Load the trained SAC model
    agent1.load_model(model_path1)
    agent2.load_model(model_path2)


    SC_SSE_LIST1 = []
    V2I_SC_SSE_LIST1 = []
    V2V_SC_SSE_LIST1 = []
    V2V_success_list1 = []

    SC_SSE_LIST2 = []
    V2I_SC_SSE_LIST2 = []
    V2V_SC_SSE_LIST2 = []
    V2V_success_list2 = []

    record_reward1 = np.zeros(n_step_per_episode1)
    reward_average_list1 = []
    record_loss1 = []
    test_action_all_training1 = np.zeros([n_veh, n_neighbor, 2], dtype='int32')

    record_reward2 = np.zeros(n_step_per_episode2)
    reward_average_list2 = []
    record_loss2 = []
    test_action_all_training2 = np.zeros([n_veh, n_neighbor, 2], dtype='int32')

    demand_sac_with_sc = test_env1.demand_size * np.ones([n_episode_test, n_step_per_episode1 + 1, n_veh, n_neighbor])
    demand_sac_without_sc = test_env2.demand_size * np.ones([n_episode_test, n_step_per_episode2 + 1, n_veh, n_neighbor])

    time_step = 0
    for i_episode in range(n_episode_test):
        print("-------------------------")
        print('Episode:', i_episode)


        test_env1.renew_positions() # update vehicle position
        test_env1.renew_neighbor()
        test_env1.renew_channel() # update channel slow fading
        test_env1.renew_channels_fastfading()
        test_env1.demand = test_env1.demand_size * np.ones((test_env1.n_Veh, test_env1.n_neighbor))
        test_env1.individual_time_limit = test_env1.time_slow * np.ones((test_env1.n_Veh, test_env1.n_neighbor))
        test_env1.active_links = np.ones((test_env1.n_Veh, test_env1.n_neighbor), dtype='bool')

        test_env2.renew_positions()  # update vehicle position
        test_env2.renew_neighbor()
        test_env2.renew_channel()  # update channel slow fading
        test_env2.renew_channels_fastfading()
        test_env2.demand = test_env2.demand_size * np.ones((test_env2.n_Veh, test_env2.n_neighbor))
        test_env2.individual_time_limit = test_env2.time_slow * np.ones((test_env2.n_Veh, test_env2.n_neighbor))
        test_env2.active_links = np.ones((test_env2.n_Veh, test_env1.n_neighbor), dtype='bool')

        state_old_all1 = []
        state_old_all2 = []

        for i in range(n_RB):
            for j in range(n_neighbor):
                state1 = get_state1(test_env1, [i, j], i_episode / (n_episode - 1), epsi_final)
                state_old_all1.append(state1)
                state2 = get_state2(test_env2, [i, j], i_episode / (n_episode - 1), epsi_final)
                state_old_all2.append(state2)


        Sum_rate_per_episode1 = []
        average_reward1 = 0
        V2I_SC_SSE_per_episode1 = []
        V2V_SC_SSE_per_episode1 = []

        Sum_rate_per_episode2 = []
        average_reward2 = 0
        V2I_SC_SSE_per_episode2 = []
        V2V_SC_SSE_per_episode2 = []

        for test_step in range(n_step_per_episode1):
            done = 0
            state_new_all1 = []
            action_all1 = []
            test_action_all_training1 = np.zeros([n_RB, n_neighbor, 2], dtype=int)  # sub, power
            test_action_all_training_length1 = np.zeros([n_veh, 2], dtype=int)  # length

            action1 = agent1.policy_net.get_action(np.asarray(state_old_all1).flatten(), deterministic=DETERMINISTIC)
            action1 = np.clip(action1, -0.999, 0.999)
            action_all1.append(action1)

            for i in range(n_RB):
                for j in range(n_neighbor):
                    test_action_all_training1[i, j, 0] = ((action1[0 + i * 2] + 1) / 2) * n_RB  # chosen RB
                    test_action_all_training1[i, j, 1] = ((action1[1 + i * 2] + 1) / 2) * n_RB  # power selected by PL
            for i in range(n_RB):
                test_action_all_training_length1[i, 0] = 1 + ((action1[2 * n_RB + i] + 1) / 2) * length
                test_action_all_training_length1[i, 1] = 1 + ((action1[2 * n_RB + 1 + i] + 1) / 2) * length
            action_temp1 = test_action_all_training1.copy()
            test_env1.rennew_symbols_of_word(test_action_all_training_length1)
            V2I_SC_SSE1, V2V_success_rate1, V2V_SC_SSE1 = test_env1.act_for_testing(action_temp1)
            V2I_SC_SSE_per_episode1.append(np.sum(V2I_SC_SSE1))
            V2V_SC_SSE_per_episode1.append(np.sum(V2V_SC_SSE1))


            test_env1.renew_channels_fastfading()
            test_env1.Compute_Interference(action_temp1)
            test_env1.compute_V2V_SNR(action_temp1)

            if test_step == n_step_per_episode1 - 1:
                V2V_success_list1.append(V2V_success_rate1)
            demand_sac_with_sc[i_episode, test_step + 1, :, :] = test_env1.demand

        for test_step in range(n_step_per_episode2):
            done = 0
            state_new_all2 = []
            action_all2 = []
            test_action_all_training2 = np.zeros([n_RB, n_neighbor, 2], dtype=int)  # sub, power
            test_action_all_training_length2 = np.zeros([n_veh, 2], dtype=int)  # length

            action2 = agent2.policy_net.get_action(np.asarray(state_old_all2).flatten(), deterministic=DETERMINISTIC)
            action2 = np.clip(action2, -0.999, 0.999)
            action_all2.append(action2)

            for i in range(n_RB):
                for j in range(n_neighbor):
                    test_action_all_training2[i, j, 0] = ((action2[0 + i * 2] + 1) / 2) * n_RB  # chosen RB
                    test_action_all_training2[i, j, 1] = ((action2[1 + i * 2] + 1) / 2) * n_RB  # power selected by PL

            action_temp2 = test_action_all_training2.copy()

            V2I_rate2, V2V_success_rate2, V2V_rate2 = test_env2.act_for_testing(action_temp2)
            V2I_SC_SSE_per_episode2.append(np.sum(V2I_rate2))
            V2V_SC_SSE_per_episode2.append(np.sum(V2V_rate2))


            test_env2.renew_channels_fastfading()
            test_env2.Compute_Interference(action_temp2)


            if test_step == n_step_per_episode2 - 1:
                V2V_success_list2.append(V2V_success_rate2)
            demand_sac_without_sc[i_episode, test_step + 1, :, :] = test_env2.demand

        V2I_SC_SSE_LIST1.append(np.mean(V2I_SC_SSE_per_episode1))
        V2V_SC_SSE_LIST1.append(np.mean(V2V_SC_SSE_per_episode1))
        print('SAC_with_sc_V2I_SC_SSE', round(np.average(V2I_SC_SSE_per_episode1), 4))
        print('SAC_with_sc_V2I_SC_SSE', round(np.average(V2V_SC_SSE_per_episode1), 4))
        print('SAC_with_sc_V2V_success', V2V_success_list1[i_episode])

        V2I_SC_SSE_LIST2.append(np.mean(V2I_SC_SSE_per_episode2))
        print('SAC_without_sc_V2I_SC_SSE', round(np.average(V2I_SC_SSE_per_episode2), 4)/test_env1.u)
        print('SAC_without_sc_V2V_success', V2V_success_list2[i_episode])
    # 计算平均值
    average_V2I_SC_SSE1 = np.mean(V2I_SC_SSE_LIST1)
    average_V2V_SC_SSE1 = np.mean(V2V_SC_SSE_LIST1)

    average_V2I_SC_SSE2 = np.mean(V2I_SC_SSE_LIST2)

    average_V2V_success1 = np.mean(V2V_success_list1)
    average_V2V_success2 = np.mean(V2V_success_list2)

    demand_sac_with_sc_subset = demand_sac_with_sc[99, :, :, :]
    demand_sac_without_sc_subset = demand_sac_without_sc[99, :, :, :]
    np.save("demand_sac_with_sc_episode_99_subset.npy", demand_sac_with_sc_subset)
    np.save("demand_sac_without_sc_episode_99_subset.npy", demand_sac_without_sc_subset)

    # 打印结果
    print('Average V2I_SC_SSE (SAC_with_sc):', round(average_V2I_SC_SSE1, 4))
    print('Average V2V_SC_SSE (SAC_with_sc):', round(average_V2V_SC_SSE1, 4))
    print('Average V2I_SC_SSE (SAC_without_sc):', round((average_V2I_SC_SSE2 / test_env1.u), 4))

    print('Average V2V_success (SAC_with_sc):', round(average_V2V_success1, 4))
    print('Average V2V_success (SAC_without_sc):', round(average_V2V_success2, 4))
    print('the demand size, 1060*', str(test_env1.byte_time))


# 打开一个文件以写入模式
with open('result/sac_vehicle_change_u_20_demand_25_results.txt', 'a') as file:
    # 将结果追加到文件
    file.write('Average V2I_SC_SSE (SAC_with_sc): ' + str(round(average_V2I_SC_SSE1, 4)) + '\n')
    file.write('Average V2V_SC_SSE (SAC_with_sc): ' + str(round(average_V2V_SC_SSE1, 4)) + '\n')
    file.write('Average V2I_SC_SSE (SAC_without_sc): ' + str(round(average_V2I_SC_SSE2 / test_env1.u, 4)) + '\n')
    file.write('Average V2V_success (SAC_with_sc): ' + str(round(average_V2V_success1, 4)) + '\n')
    file.write('Average V2V_success (SAC_without_sc): ' + str(round(average_V2V_success2, 4)) + '\n')
    file.write('the u: ' + str(test_env1.u) + '\n')
    file.write('the demand size: 1060 * ' + str(test_env1.byte_time) + '\n')
    file.write('V2I_power_dB ' + str(test_env1.V2I_power_dB) + '\n')


