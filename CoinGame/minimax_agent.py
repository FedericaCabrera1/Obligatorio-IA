from agent import Agent
from board import Board
import math

class MinimaxAgent(Agent):
    def __init__(self, player=1):
        super().__init__(player)

    def next_action(self, obs):
        best_action, _ = self.minimax(obs, self.player, depth=5, alpha=-math.inf, beta=math.inf)
        return best_action


    def minimax(self, board, player, depth, alpha, beta):
        is_terminal, winner = board.is_end(player)
        if is_terminal:
            return None, 1 if winner == self.player else -1

        if depth == 0:
            return None, self.heuristic_utility(board)

        best_action = None
        if player == self.player:
            max_eval = -math.inf
            for action in board.get_possible_actions():
                new_board = board.clone()
                new_board.play(action)
                _, eval = self.minimax(new_board, 3 - player, depth-1, alpha, beta)
                if eval > max_eval:
                    max_eval = eval
                    best_action = action
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return best_action, max_eval
        else:
            min_eval = math.inf
            for action in board.get_possible_actions():
                new_board = board.clone()
                new_board.play(action)
                _, eval = self.minimax(new_board, 3 - player, depth-1, alpha, beta)
                if eval < min_eval:
                    min_eval = eval
                    best_action = action
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return best_action, min_eval

    def heuristic_utility(self, board):
        coin_count = sum(row.sum() for row in board.grid)
        max_row_length = max(sum(row) for row in board.grid)
        return -(coin_count + max_row_length)

