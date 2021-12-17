# Imports
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import gym

# khởi tạo môi trường
eval_env = gym.make('LunarLander-v2')
# khởi tạo DQN
nn_layers = [64,64]
policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=nn_layers)
model = DQN('MlpPolicy', 'LunarLander-v2', verbose=1, exploration_final_eps=0.1, target_update_interval=250, policy_kwargs=policy_kwargs, learning_rate=0.0001)

# Training model
model.learn(total_timesteps=100000, log_interval=10)
# Save the agent
model.save("dqn_lunar")

del model  # delete trained model to demonstrate loading


# nhận xét kết quả
obs = eval_env.reset()
model = DQN.load("dqn_lunar")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(mean_reward, std_reward)
total_reward = 0


while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)
    eval_env.render()
    total_reward += reward
    print(reward)
    if done:
      print(total_reward)
      obs = eval_env.reset()
      total_reward = 0