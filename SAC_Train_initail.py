import numpy as np
import os
import scipy.io
import Environment_marl_V2I_and_V2V_SC_for_SAC_initial
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

IS_TRAIN = 1
IS_TEST = 1-IS_TRAIN

label = 'model/sac_model/with_sc_n_veh_12_lamda0.9_lr3e-4_hs_512_bs64_newpowerlist'
model_path = label + '/agent'
n_veh = 4
# n_veh = 12

n_neighbor = 1
n_RB = n_veh
# max_power = 23
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
n_input = len(get_state(env=env))*n_RB
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

# ------------------------- Training -----------------------------
if IS_TRAIN:
    record_reward = np.zeros(n_step_per_episode)
    reward_average_list = []
    record_loss = []
    action_all_training = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
    time_step = 0
    for i_episode in range(n_episode):
        print("-------------------------")
        print('Episode:', i_episode)
        if i_episode < epsi_anneal_length:
            epsi = 1 - i_episode * (1 - epsi_final) / (epsi_anneal_length - 1)  # epsilon decreases over each episode
        else:
            epsi = epsi_final
        if i_episode%20 == 0:
            env.renew_positions() # update vehicle position
            env.renew_neighbor()
            env.renew_channel() # update channel slow fading
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
        for i_step in range(n_step_per_episode):
            done = 0
            state_new_all = []
            action_all = []
            action_all_training = np.zeros([n_RB, n_neighbor, 2], dtype=int)  # sub, power
            action_all_training_length = np.zeros([n_veh, 2], dtype=int)  # length
            # receive observation
            if frame_idx > explore_steps:
                action = agent.policy_net.get_action(np.asarray(state_old_all).flatten(), deterministic=DETERMINISTIC)
            else:
                action = agent.policy_net.sample_action()
            action = np.clip(action, -0.999, 0.999)
            action_all.append(action)
            for i in range(n_RB):
                for j in range(n_neighbor):
                    action_all_training[i, j, 0] = ((action[0 + i * 2] + 1) / 2) * n_RB  # chosen RB
                    action_all_training[i, j, 1] = ((action[1 + i * 2] + 1) / 2) * 4  # power selected by PL
            for i in range(n_RB):
                action_all_training_length[i, 0] = 1 + ((action[2 * n_RB + i] + 1) / 2) * length
                action_all_training_length[i, 1] = 1 + ((action[2 * n_RB + 1 + i] + 1) / 2) * length
            action_temp = action_all_training.copy()
            env.rennew_symbols_of_word(action_all_training_length)
            train_reward = env.act_for_training(action_temp)
            record_reward[i_step] = train_reward

            env.renew_channels_fastfading()
            env.Compute_Interference(action_temp)
            env.compute_V2V_SNR(action_temp)

            for i in range(n_RB):
                for j in range(n_neighbor):
                    state_new = get_state(env, [i, j], i_episode / (n_episode - 1), epsi)
                    state_new_all.append((state_new))
             # taking the agents actions, states and reward
            replay_buffer.push(np.asarray(state_old_all).flatten(), np.asarray(action_all).flatten(),
                                       train_reward, np.asarray(state_new_all).flatten(), done)

            # agents take random samples and learn
            if len(replay_buffer) > batch_size:
                for i in range(update_itr):
                    _ = agent.update(batch_size, reward_scale=10., auto_entropy=AUTO_ENTROPY,
                                             target_entropy=-1. * n_output)

            # old observation = new_observation
            state_old_all = state_new_all
            frame_idx += 1

        average_reward = np.mean(record_reward)
        print('reward' , average_reward)
        reward_average_list.append(average_reward)

        if (i_episode+1) % 100 == 0 and i_episode != 0:
            agent.save_model(model_path)

    x = np.linspace(0, n_episode - 1, n_episode, dtype=int)
    y1 = reward_average_list
    plt.figure(1)
    plt.plot(x, y1)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    print('Training Done. Saving models...')
    np.save('Data/SAC_with_sc1000_n_veh_8_lamda0.9_lr3e-4_hs_512_bs64_newpowerlist.npy', reward_average_list)

    # print("Sorted SNR results have been saved to", output_file)
