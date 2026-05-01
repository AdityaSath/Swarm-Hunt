import torch
from torchrl.envs import VmasEnv

device = "cuda" if torch.cuda.is_available() else "cpu"
env = VmasEnv(
    scenario="simple_tag",
    num_envs=1,
    continuous_actions=True,
    max_steps=50,
    device=device,
    seed=0,
    # scenario specific (TorchRL tutorial)
    num_good_agents=1,
    num_adversaries=2,
    num_landmarks=2,
)

# TorchRL examples show env.rollout(...) usage for quick checks.
td = env.rollout(10)
print("Rollout keys:", td.keys(True, True))
print("Done shape:", td.get("done").shape)
env.close()