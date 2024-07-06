from datetime import datetime
from board import Board as GameBoard
from random_agent import RandomAgent
from minimax_agent import MinimaxAgent
from coin_game_env import CoinGameEnv

def play_vs_other_agent(env, agent1, agent2, render=False):
    done = False
    obs = env.reset()
    winner = 0
    while not done:
        if render: env.render()
        action = agent1.next_action(obs)
        obs, _, done, winner, _ = env.step(action)

        if render: env.render()
        if not done:
            next_action = agent2.next_action(obs)
            _, _, done, winner, _ = env.step(next_action)

    if render: env.render()
    return winner

if __name__ == "__main__":
    env = CoinGameEnv(grid_size=3)
    agent1 = MinimaxAgent(player=1)
    agent2 = RandomAgent(player=2)
    winner = play_vs_other_agent(env, agent1, agent2, render=True)
    print(f"El ganador es el jugador: {winner}")
