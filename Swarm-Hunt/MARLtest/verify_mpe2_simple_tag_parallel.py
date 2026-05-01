from mpe2 import simple_tag_v3
from pettingzoo.test import parallel_api_test

env = simple_tag_v3.parallel_env(
    num_good=1,
    num_adversaries=2,
    num_obstacles=2,
    max_cycles=50,
    continuous_actions=True,
    render_mode=None
)
# 1) API sanity check (recommended when wiring wrappers/algorithms)
parallel_api_test(env, num_cycles=25)
# 2) Run a single episode with random actions
obs, infos = env.reset(seed=0)
total_reward = {a: 0.0 for a in env.agents}

done = {a: False for a in env.agents}
while env.agents:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
    for a, r in rewards.items():
        total_reward[a] += float(r)
        done = {a: terminations[a] or truncations[a] for a in env.agents}
    if all(done.values()):
        break
    print("Episode complete. Total reward:", total_reward)
    env.close()
