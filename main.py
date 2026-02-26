import chess
import torch

import mcts
import encoder
from model import ChessNet


def play_match(engine1: mcts.MCTS, engine2: mcts.MCTS = None):
    board = chess.Board()

    print("Initializing AlphaZero Prototype...")
    print(board)
    print("-" * 30)


    while not board.is_game_over(claim_draw=True):
        # 1. MCTS runs its 800 simulations
        print(f"Thinking for {"WHITE" if board.turn == chess.WHITE else "BLACK"}... ", end="", flush=True)
        if engine1.terminate_early():
            break

        player = engine2 if engine2 is not None and board.turn == chess.BLACK else engine1

        best_move = player.search(board)
        board.push(best_move)
        engine1.update_with_move(best_move)  # Keep the tree synced!
        if engine2:
            engine2.update_with_move(best_move)


        # 3. Display the board state
        print(f"Played {best_move}")
        print(board)
        print("-" * 30)

    # 4. Game Over Evaluation
    print("Game Over!")
    outcome = board.outcome(claim_draw=True)
    if not outcome:
        print("Result: 1/2-1/2 (Draw)")
        print(f"Reason: early termination")
        return

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

    nn = ChessNet().to(device)
    nn.load_state_dict(torch.load("chessnet_v2.pth"))

    nn2 = ChessNet().to(device)
    nn2.load_state_dict(torch.load("chessnet_v3.pth"))

    board = chess.Board()

    tree = mcts.MCTS(nn=nn)
    tree2 = mcts.MCTS(nn=nn2)
    play_match(tree, tree2)
