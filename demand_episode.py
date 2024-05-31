import numpy as np
import matplotlib.pyplot as plt

# 从文件加载数据
demand_sac_with_sc_subset = np.load("demand_sac_with_sc_episode_99_subset.npy")
demand_sac_without_sc_subset = np.load("demand_sac_without_sc_episode_99_subset.npy")

# 获取车辆数量
num_vehicles_sac_with_sc = demand_sac_with_sc_subset.shape[1]
num_vehicles_sac_without_sc = demand_sac_without_sc_subset.shape[1]

# 获取时间步数
time_steps_sac_with_sc = demand_sac_with_sc_subset.shape[0]
time_steps_sac_without_sc = demand_sac_without_sc_subset.shape[0]


# 创建时间步数组（假设时间步是从0到time_steps-1）
time_array_sac_with_sc = np.arange(time_steps_sac_with_sc)
time_array_sac_without_sc = np.arange(time_steps_sac_without_sc)


# 画出每一辆车的曲线
for vehicle_idx in range(num_vehicles_sac_with_sc):
    plt.plot(time_array_sac_with_sc, (demand_sac_with_sc_subset[:, vehicle_idx, 0]) * 20, marker='o', markersize=3, label=f'Vehicle {vehicle_idx + 1}')

# 添加标题和标签
# plt.title('sac_with_sc Demand Variation for Each Vehicle Over Time')
plt.xlabel('Time Step/ms')
plt.ylabel('Demand')
plt.legend()
plt.grid(True)
# 显示图形
plt.savefig('fig_sac_with_sc_demand.pdf')
plt.show()

for vehicle_idx in range(num_vehicles_sac_without_sc):
    plt.plot(time_array_sac_without_sc, demand_sac_without_sc_subset[:, vehicle_idx, 0], marker='o', markersize=3, label=f'Vehicle {vehicle_idx + 1}')

# 添加标题和标签
# plt.title('sac_without_sc Demand Variation for Each Vehicle Over Time')
plt.xlabel('Time Step/ms')
plt.ylabel('Demand')
plt.legend()
plt.grid(True)
# 显示图形
plt.savefig('fig_sac_without_sc_demand.pdf')
plt.show()
