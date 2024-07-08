import numpy as np
import pickle
from taxi_env_extended import TaxiEnvExtended

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

    def set_Q(self, Q):
        self.Q = Q

    def epsilon_greedy_policy(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state])

    def evaluate(self, num_episodes=100):
        self.epsilon = 0 
        total_rewards = 0
        total_penalties = 0
        total_steps = 0

        for _ in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0
            episode_penalties = 0

            while not done:
                action = self.epsilon_greedy_policy(state)
                next_state, reward, done, _, _ = self.env.step(action)
                if reward == -10:
                    episode_penalties += 1
                episode_reward += reward
                state = next_state
                episode_steps += 1

            total_rewards += episode_reward
            total_steps += episode_steps
            total_penalties += episode_penalties

        avg_reward = total_rewards / num_episodes
        avg_steps = total_steps / num_episodes
        avg_penalties = total_penalties / num_episodes

        print(f"Average Reward per Episode: {avg_reward:.2f}")
        print(f"Average Steps per Episode: {avg_steps:.2f}")
        print(f"Average Penalties per Episode: {avg_penalties:.2f}")

        return avg_reward, avg_steps, avg_penalties


with open('best_q_agent.pkl', 'rb') as f:
    loaded_Q = pickle.load(f)

env = TaxiEnvExtended()
agent = QLearningAgent(env)
agent.set_Q(loaded_Q)

agent.evaluate(num_episodes=100)
