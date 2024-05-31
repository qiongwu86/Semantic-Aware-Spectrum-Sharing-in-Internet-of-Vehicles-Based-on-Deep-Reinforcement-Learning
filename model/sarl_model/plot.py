import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Load data from train_loss.mat and reward.mat
train_loss_data = scipy.io.loadmat("train_loss.mat")
reward_data = scipy.io.loadmat("reward.mat")

# Extract train loss values
train_loss_values = train_loss_data['train_loss']
train_loss_values = np.transpose(train_loss_values)

# Extract reward values
reward_values = reward_data['reward']
reward_values = np.transpose(reward_values)


# Create x-axis values for the plots (number of episodes)
n_episodes_train = len(train_loss_values)
n_episodes_reward = len(reward_values)
episodes_train = np.arange(1, n_episodes_train + 1)
episodes_reward = np.arange(1, n_episodes_reward + 1)

# Plot Training Loss
plt.figure(figsize=(10, 6))
plt.plot(episodes_train, train_loss_values, label='Training Loss')
plt.xlabel('Episodes')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot Reward over Episodes
plt.figure(figsize=(10, 6))
plt.plot(episodes_reward, reward_values, label='Reward')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Reward ')
plt.legend()
plt.grid(True)
plt.show()
