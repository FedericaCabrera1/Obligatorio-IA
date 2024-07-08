import numpy as np
from taxi_env_extended import TaxiEnvExtended
import matplotlib.pyplot as plt
import pickle

def epsilon_greedy_policy(state, Q, env, epsilon=0.1):
    explore = np.random.binomial(1, epsilon)
    if explore:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state])
    return action

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        self.episode_rewards = []
        self.steps_per_episode = []
        self.penalties_per_episode = []

    def learn(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state][best_next_action]
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

    def q_learning(self, num_episodes):
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            penalties = 0
            while not done:
                action = epsilon_greedy_policy(state, self.Q, self.env, self.epsilon)
                next_state, reward, done, _, _ = self.env.step(action)
                if reward == -10:
                    penalties += 1
                self.learn(state, action, reward, next_state)
                state = next_state
                total_reward += reward
                steps += 1
            self.episode_rewards.append(total_reward)
            self.steps_per_episode.append(steps)
            self.penalties_per_episode.append(penalties)
            self.epsilon *= self.epsilon_decay

def run_learning(alpha, gamma, epsilon, epsilon_decay, num_episodes=1000):
    env = TaxiEnvExtended()
    agent = QLearningAgent(env, alpha, gamma, epsilon, epsilon_decay)
    agent.q_learning(num_episodes)

    q_learning_rewards = agent.episode_rewards[-100:]
    q_learning_steps = agent.steps_per_episode[-100:]
    q_learning_penalties = agent.penalties_per_episode[-100:]

    label = f"alpha={alpha}, gamma={gamma}, epsilon={epsilon}, epsilon_decay={epsilon_decay} num_episodes={num_episodes}"

    print(f"*** {label} ***")
    print(f"Average Reward per Episode: {np.mean(q_learning_rewards):.2f}")
    print(f"Average Penalties per Episode: {np.mean(q_learning_penalties):.2f}")
    print(f"Average Steps per Episode: {np.mean(q_learning_steps):.2f}")

    return agent

alpha = 0.6
gamma = 0.8
epsilon = 1.0
epsilon_decay = 0.8
episodes = 21000

best_agent = run_learning(alpha, gamma, epsilon, epsilon_decay, episodes) 

with open('best_q_agent.pkl', 'wb') as f:
    pickle.dump(best_agent.Q, f)