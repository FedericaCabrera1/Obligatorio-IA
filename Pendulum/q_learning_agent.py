import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pendulum_env_extended import PendulumEnvExtended
import pickle

x_space = np.linspace(-1, 1, 10)
y_space = np.linspace(-1, 1, 10)
vel_space= np.linspace(-8, 8, 50)
action_space= list(np.linspace(-2, 2, 5))

alpha = 0.3
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.99
num_episodes = 5000

def get_state(obs, x_space, y_space, vel_space):
    x, y, vel = obs
    x_bin = np.digitize(x, x_space)
    y_bin = np.digitize(y, y_space)
    vel_bin = np.digitize(vel, vel_space)
    return x_bin, y_bin, vel_bin

def epsilon_greedy_policy(state, Q, epsilon=0.1):
    if np.random.random() < epsilon:
        return np.random.randint(len(Q[state]))
    else:
        return np.argmax(Q[state])

class QLearningAgent:
    def __init__(self, env, state_bins, action_bins, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.env = env
        self.state_bins = state_bins
        self.action_bins = action_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.Q = np.random.uniform(low=-1, high=0, size=(*[len(b) + 1 for b in state_bins], len(action_bins)))
        self.episode_rewards = []
        self.steps_per_episode = []

    def learn(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state][best_next_action]
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

    def q_learning(self, num_episodes):
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state = get_state(state, self.state_bins[0], self.state_bins[1], self.state_bins[2])
            done = False
            total_reward = 0
            steps = 0

            while not done:
                action = epsilon_greedy_policy(state, self.Q, self.epsilon)
                continuous_action = [self.action_bins[action]]
                next_state, reward, done, _, _ = self.env.step(continuous_action)
                next_state = get_state(next_state, self.state_bins[0], self.state_bins[1], self.state_bins[2])
                self.learn(state, action, reward, next_state)
                state = next_state
                total_reward += reward
                steps += 1
            self.episode_rewards.append(total_reward)
            self.steps_per_episode.append(steps)
            self.epsilon *= self.epsilon_decay

def run_learning(alpha, gamma, epsilon, epsilon_decay, x_space, y_space, vel_space, actions, num_episodes=3000):
    env = PendulumEnvExtended()
    state_bins = [x_space, y_space, vel_space]
    action_bins = actions
    
    agent = QLearningAgent(env, state_bins, action_bins, alpha, gamma, epsilon, epsilon_decay)
    agent.q_learning(num_episodes)

    q_learning_rewards = agent.episode_rewards[-100:]
    print(f"Average Reward per Episode: {np.mean(q_learning_rewards):.2f}")
    return agent

best_agent = run_learning(alpha, gamma, epsilon, epsilon_decay, x_space, y_space, vel_space, action_space, num_episodes) 

with open('best_q_agent.pkl', 'wb') as f:
    pickle.dump(best_agent.Q, f)

plt.figure(figsize=(14, 7))

plt.plot(range(num_episodes), best_agent.episode_rewards, label='Q-Learning')
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Total Rewards per Episode for Q-Learning')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()