import numpy as np
import os
import scipy.io
import Environment_marl_initial_for_sac
from RL_DDPG import Agent
import matplotlib.pyplot as plt


def get_state(env, idx=(0,0), ind_episode=1., epsi=0.02):
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

IS_TRAIN = 1
IS_TEST = 1-IS_TRAIN

label = 'model/ddpg_model/without_sc/3000_newpowerlist_1024'
model_path = label + '/agent'
n_veh = 4
n_neighbor = 1
n_RB = n_veh
# max_power = 23
length = 20
env = Environment_marl_initial_for_sac.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor)
env.new_random_game()  # initialize parameters in env

n_episode = 3000
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
batch_size = 64
memory_size = 1000000
gamma = 0.99
alpha = 0.0001
beta = 0.001
# actor and critic hidden layers
fc1_dims = 1024
fc2_dims = 1024
fc3_dims = 1024
fc4_dims = 1024
# ------------------------------

tau = 0.005#参数更新权重
agent = Agent(alpha, beta, n_input, tau, n_output, gamma, memory_size, fc1_dims, fc2_dims, fc3_dims, fc4_dims, batch_size, n_veh)

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
            # receive observation
            action = agent.choose_action(np.asarray(state_old_all).flatten())
            action = np.clip(action, -0.999, 0.999)
            action_all.append(action)
            for i in range(n_RB):
                for j in range(n_neighbor):
                    action_all_training[i, j, 0] = ((action[0 + i * 2] + 1) / 2) * n_RB  # chosen RB
                    action_all_training[i, j, 1] = ((action[1 + i * 2] + 1) / 2) * n_RB  # power selected by PL

            action_temp = action_all_training.copy()
            train_reward = env.act_for_training(action_temp)
            record_reward[i_step] = train_reward

            # env.renew_channel()  # update channel slow fading
            env.renew_channels_fastfading()
            env.Compute_Interference(action_temp)

            for i in range(n_RB):
                for j in range(n_neighbor):
                    state_new = get_state(env, [i, j], i_episode / (n_episode - 1), epsi)
                    state_new_all.append((state_new))

            if i_step == n_step_per_episode - 1:
                done = True

                # taking the agents actions, states and reward
            agent.remember(np.asarray(state_old_all).flatten(), np.asarray(action_all).flatten(),
                           train_reward, np.asarray(state_new_all).flatten(), done)

            # agents take random samples and learn
            agent.learn()

            # old observation = new_observation
            state_old_all = state_new_all

        average_reward = np.mean(record_reward)
        print('reward' , average_reward)
        reward_average_list.append(average_reward)

        if (i_episode+1) % 100 == 0 and i_episode != 0:
            agent.save_models()

    x = np.linspace(0, n_episode - 1, n_episode, dtype=int)
    y1 = reward_average_list
    plt.figure(1)
    plt.plot(x, y1)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    print('Training Done. Saving models...')
    np.save('Data/ddpg_3000_without_newpowerlist_1024.npy', reward_average_list)

    # print("Sorted SNR results have been saved to", output_file)
