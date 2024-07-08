import numpy as np
import matplotlib.pyplot as plt
import pickle
from pendulum_env_extended import PendulumEnvExtended

with open('best_q_agent.pkl', 'rb') as f:
    Q = pickle.load(f)

x_space = np.linspace(-1, 1, 10)
y_space = np.linspace(-1, 1, 10)
vel_space = np.linspace(-8, 8, 50)
actions = list(np.linspace(-2, 2, 5))

def get_state(obs):
    x, y, vel = obs
    x_bin = np.digitize(x, x_space)
    y_bin = np.digitize(y, y_space)
    vel_bin = np.digitize(vel, vel_space)
    return x_bin, y_bin, vel_bin

def test_agent(env, Q, num_episodes=100):
    total_rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = get_state(state)
        done = False
        total_reward = 0
        while not done:
            action = np.argmax(Q[state])
            continuous_action = [actions[action]]
            next_state, reward, done, _, _ = env.step(continuous_action)
            next_state = get_state(next_state)
            state = next_state
            total_reward += reward
        total_rewards.append(total_reward)
    return total_rewards

env = PendulumEnvExtended()
num_test_episodes = 100
rewards = test_agent(env, Q, num_test_episodes)

average_reward = np.mean(rewards)
print(f"Average Reward over {num_test_episodes} episodes: {average_reward:.2f}")