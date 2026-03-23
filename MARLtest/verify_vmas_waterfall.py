import vmas

env = vmas.make_env(
    scenario="waterfall",
    num_envs=1,
    device="cpu",
    continuous_actions=True,
    max_steps=50,
    n_agents=3
)
obs = env.reset()
for t in range(50):
    obs, rews, dones, info = env.step(env.get_random_actions())
    if dones.all():
        break
print("VMAS episode ran. Steps:", t + 1)