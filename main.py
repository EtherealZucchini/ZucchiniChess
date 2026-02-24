import chess
import torch

import MCTS
import encoder
from model import ChessNetBody, ChessNetValue, ChessNetPolicy


def play_match(mcts: MCTS):
    board = chess.Board()

    print("Initializing AlphaZero Prototype...")
    print(board)
    print("-" * 30)

    while not board.is_game_over(claim_draw=True):
        # 1. MCTS runs its 800 simulations
        print(f"Thinking for {"WHITE" if board.turn == chess.WHITE else "BLACK"}... ", end="", flush=True)
        best_move = mcts.search(board)
        board.push(best_move)
        mcts.update_with_move(best_move)  # Keep the tree synced!


        # 3. Display the board state
        print(f"Played {best_move}")
        print(board)
        print("-" * 30)

    # 4. Game Over Evaluation
    print("Game Over!")
    outcome = board.outcome(claim_draw=True)

    if outcome.winner == chess.WHITE:
        print("Result: 1-0 (White Wins)")
    elif outcome.winner == chess.BLACK:
        print("Result: 0-1 (Black Wins)")
    else:
        print("Result: 1/2-1/2 (Draw)")
        print(f"Reason: {outcome.termination.name}")


if __name__ == "__main__":
    # Initialize your PyTorch networks
    # (Assuming you have a master ChessNet or the three separate modules)
    # body = ...
    # value_head = ...
    # policy_head = ...

    # Initialize the MCTS wrapper
    # mcts = MCTS(body, value_head, policy_head, num_simulations=800)

    # Start the game
    # play_match(mcts)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    body = ChessNetBody().to(device)
    value = ChessNetValue().to(device)
    policy = ChessNetPolicy().to(device)
    board = chess.Board()
    print(board.fen())

    mcts = MCTS.MCTS(body=body, policy_head=policy, value_head=value)
    play_match(mcts)
