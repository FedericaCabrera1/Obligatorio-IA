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

def run_simulation(num_games=100):
    env = CoinGameEnv(grid_size=3)
    agent1 = MinimaxAgent(player=1)
    agent2 = RandomAgent(player=2)
    
    wins_player1 = 0
    wins_player2 = 0
    
    for _ in range(num_games):
        winner = play_vs_other_agent(env, agent1, agent2, render=False)
        if winner == 1:
            wins_player1 += 1
        elif winner == 2:
            wins_player2 += 1
    
    return wins_player1, wins_player2

if __name__ == "__main__":
    num_games = 100
    wins_player1, wins_player2 = run_simulation(num_games)
    print(f"En {num_games} partidas:")
    print(f"Jugador 1 (MinimaxAgent) ganó {wins_player1} veces.")
    print(f"Jugador 2 (RandomAgent) ganó {wins_player2} veces.")
