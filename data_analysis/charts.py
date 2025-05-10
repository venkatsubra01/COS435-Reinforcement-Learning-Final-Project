import pandas as pd
import matplotlib.pyplot as plt


# FIG 1 -- Ant CRL vs ICRL
# CRL # https://wandb.ai/vs9542-princeton-university/jaxgcrl/runs/f6f0v64p?nw=nwuservs9542
crl_data_frame = pd.read_csv('Input_Data/CRL_Ant.csv')
crl_data_frame = crl_data_frame[["Step", "run - eval/episode_success"]]
crl_data_frame.rename(columns={"run - eval/episode_success": "reward"}, inplace=True)

# Load ICRL Data and rename column
icrl_data_frame = pd.read_csv('Input_Data/ICRL_Ant.csv') #  https://wandb.ai/vs9542-princeton-university/TEST_WANDB/runs/0i5mb8fn?nw=nwusersaij10
icrl_data_frame = icrl_data_frame[["Step", "mabrax_ant__train_icrl__1__1746058437 - eval/episode_success"]]
icrl_data_frame.rename(columns={"mabrax_ant__train_icrl__1__1746058437 - eval/episode_success": "reward"}, inplace=True)

# Smooth rewards
crl_data_frame["Smoothed"] = crl_data_frame["reward"].rolling(window=8, min_periods=1).mean()
icrl_data_frame["Smoothed"] = icrl_data_frame["reward"].rolling(window=8, min_periods=1).mean()

# Limit steps to below a certain value (like 6.5 M)
crl_data_frame = crl_data_frame[crl_data_frame["Step"] < 6500000]
icrl_data_frame = icrl_data_frame[icrl_data_frame["Step"] < 6500000]


# Plot data
plt.figure(figsize=(6, 4))
plt.plot(crl_data_frame["Step"], crl_data_frame["Smoothed"], label="CRL", color='red')
plt.plot(icrl_data_frame["Step"], icrl_data_frame["Smoothed"], label="ICRL", color='green')
plt.title("CRL/ICRL Smoothed Episode Success on Ant")
plt.xlabel("Timesteps")
plt.ylabel("Episode Success (Smoothed)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("ant_crl_vs_icrl.png")

# FIG 2 -- Ant PPO Dense vs PPO Sparse

# PPO Dense Reward - https://wandb.ai/vs9542-princeton-university/jaxgcrl/runs/ofw1mgcv?nw=nwuservs9542
ppo_dense_reward_frame = pd.read_csv('Input_Data/PPO_Dense_Reward_Ant.csv')
ppo_dense_reward_frame = ppo_dense_reward_frame[["Step", "run - eval/episode_success"]]
ppo_dense_reward_frame.rename(columns={"run - eval/episode_success": "reward"}, inplace=True)

# PPO Sparse Reward--https://wandb.ai/vs9542-princeton-university/jaxgcrl/runs/6hdszwdl?nw=nwuservs9542
ppo_sparse_reward_frame = pd.read_csv('Input_Data/PPO_Sparse_Reward_Ant.csv')
ppo_sparse_reward_frame = ppo_sparse_reward_frame[["Step", "run - eval/episode_success"]]
ppo_sparse_reward_frame.rename(columns={"run - eval/episode_success": "reward"}, inplace=True)

ppo_dense_reward_frame["Smoothed"] = ppo_dense_reward_frame["reward"].rolling(window=8, min_periods=1).mean()
ppo_sparse_reward_frame["Smoothed"] = ppo_sparse_reward_frame["reward"].rolling(window=8, min_periods=1).mean()

ppo_dense_reward_frame = ppo_dense_reward_frame[ppo_dense_reward_frame["Step"] < 50000000]
ppo_sparse_reward_frame = ppo_sparse_reward_frame[ppo_sparse_reward_frame["Step"] < 50000000]

# Plot data
plt.figure(figsize=(6, 4))
plt.plot(ppo_dense_reward_frame["Step"], ppo_dense_reward_frame["Smoothed"], label="PPO Dense Reward")
plt.plot(ppo_sparse_reward_frame["Step"], ppo_sparse_reward_frame["Smoothed"], label="PPO Sparse Reward")
plt.title("PPO Smoothed Episode Success on Ant")
plt.xlabel("Timesteps")
plt.ylabel("Episode Success (Smoothed)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("dense_ppo_vs_sparse_ppo.png")

# FIG 3 -- AntSoccerV1


# Default PPO - 0 reward -- https://wandb.ai/vs9542-princeton-university/jaxgcrl/runs/hpr2mot8/overview
ppo_default_frame = pd.read_csv('Input_Data/PPO_Default_AntSoccerV1.csv')
ppo_default_frame = ppo_default_frame[["Step", "run - eval/episode_success"]]
ppo_default_frame.rename(columns={"run - eval/episode_success": "reward"}, inplace=True)


# CRL -- https://wandb.ai/vs9542-princeton-university/jaxgcrl/runs/ct7uik87/workspace?nw=nwuservs9542
crl_data_frame = pd.read_csv('Input_data/CRL_AntSoccerV1.csv')
crl_data_frame = crl_data_frame[["Step", "run - eval/episode_success"]]
crl_data_frame.rename(columns={"run - eval/episode_success": "reward"}, inplace=True)

# Modified Reward Function of PPO -- https://wandb.ai/vs9542-princeton-university/jaxgcrl/runs/zkmadbao?nw=nwuservs9542
ppo_modified_frame = pd.read_csv('Input_Data/PPO_Modified_AntSoccerV1.csv')
ppo_modified_frame = ppo_modified_frame[["Step", "run - eval/episode_success"]]
ppo_modified_frame.rename(columns={"run - eval/episode_success": "reward"}, inplace=True)


# Smooth rewards over window
ppo_default_frame["Smoothed"] = ppo_default_frame["reward"].rolling(window=8, min_periods=1).mean()
crl_data_frame["Smoothed"] = crl_data_frame["reward"].rolling(window=8, min_periods=1).mean()
ppo_modified_frame["Smoothed"] = ppo_modified_frame["reward"].rolling(window=8, min_periods=1).mean()

# Plot figures
plt.figure(figsize=(6, 4))
plt.plot(ppo_default_frame["Step"], ppo_default_frame["Smoothed"], label="PPO Default AntSoccerV1")
plt.plot(crl_data_frame["Step"], crl_data_frame["Smoothed"], label="CRL Default AntSoccerV1")
plt.plot(ppo_modified_frame["Step"], ppo_modified_frame["Smoothed"], label="PPO Modified AntSoccerV1")
plt.title("PPO/CRL Smoothed Episode Success on AntSoccerV1")
plt.xlabel("Timesteps")
plt.ylabel("Episode Success (Smoothed)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("AntSoccerv1_ppo_default_vs_modified_ppo_vs_crl.png")


# FIG 4 -- AntSoccerV2

# Modified Reward Function of PPO -- https://wandb.ai/vs9542-princeton-university/jaxgcrl/runs/2w6f94nm?nw=nwuservs9542
ppo_modified_frame = pd.read_csv('Input_Data/PPO_Modified_AntSoccerV2.csv')
ppo_modified_frame = ppo_modified_frame[["Step", "run - eval/episode_success"]]
ppo_modified_frame.rename(columns={"run - eval/episode_success": "reward"}, inplace=True)


# CRL 2 hidden layers on AntSoccerV2 environment with distance as 20 and episode length as 1001, 500 million training steps, no random target – boundaries don’t do anything
# -- https://wandb.ai/vs9542-princeton-university/jaxgcrl/runs/lvs0jo1w?nw=nwuservs9542
crl_data_frame = pd.read_csv('Input_Data/CRL_AntSoccerV2.csv')
crl_data_frame = crl_data_frame[["Step", "run - eval/episode_success"]]
crl_data_frame.rename(columns={"run - eval/episode_success": "reward"}, inplace=True)

# Smooth rewards
crl_data_frame["Smoothed"] = crl_data_frame["reward"].rolling(window=8, min_periods=1).mean()
ppo_modified_frame["Smoothed"] = ppo_modified_frame["reward"].rolling(window=8, min_periods=1).mean()

# Plot rewards
plt.figure(figsize=(6, 4))
plt.plot(crl_data_frame["Step"], crl_data_frame["Smoothed"], label="CRL Default AntSoccerV2")
plt.plot(ppo_modified_frame["Step"], ppo_modified_frame["Smoothed"], label="PPO Modified AntSoccerV2")
plt.title("CRL/PPO Smoothed Episode Success on AntSoccerV2")
plt.xlabel("Timesteps")
plt.ylabel("Episode Success (Smoothed)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("AntSoccerv2_ppo_modified_vs_crl.png")



# ICRL on AntSoccer V1  -- https://wandb.ai/vs9542-princeton-university/TEST_WANDB/runs/ork68dcx?nw=nwusersaij10
icrl_data_frame = pd.read_csv("Input_Data/ICRL_AntSoccerV1.csv")
icrl_data_frame = icrl_data_frame[["Step", "mabrax_ant_soccer__train_icrl__1__1746579734 - eval/episode_success"]]
icrl_data_frame.rename(columns={"mabrax_ant_soccer__train_icrl__1__1746579734 - eval/episode_success": "reward"}, inplace=True)
icrl_data_frame["Smoothed"] = icrl_data_frame["reward"].rolling(window=8, min_periods=1).mean()

plt.figure(figsize=(6, 4))
plt.plot(icrl_data_frame["Step"], icrl_data_frame["Smoothed"], label="ICRL AntSoccerV1")
plt.title("ICRL Smoothed Episodes Success on AntSoccerV1")
plt.xlabel("Timesteps")
plt.ylabel("Episode Success (Smoothed)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("AntSoccerv1_icrl.png")


# ICRL on AntSoccerV2  -- https://wandb.ai/vs9542-princeton-university/TEST_WANDB/runs/4dblsd0t?nw=nwusersaij10
icrl_data_frame = pd.read_csv("Input_Data/ICRL_AntSoccerV2.csv")
icrl_data_frame = icrl_data_frame[["Step", "mabrax_ant_soccerv2__train_icrl__1__1746579584 - eval/episode_success"]]
icrl_data_frame.rename(columns={"mabrax_ant_soccerv2__train_icrl__1__1746579584 - eval/episode_success": "reward"}, inplace=True)
icrl_data_frame["Smoothed"] = icrl_data_frame["reward"].rolling(window=8, min_periods=1).mean()

plt.figure(figsize=(6, 4))
plt.plot(icrl_data_frame["Step"], icrl_data_frame["Smoothed"], label="ICRL AntSoccerV2")
plt.title("ICRL Smoothed Episode Success on AntSoccerV2")
plt.xlabel("Timesteps")
plt.ylabel("Episode Success (Smoothed)")
plt.ylim(bottom=0)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("AntSoccerv2_icrl.png")





#
