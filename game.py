import chess
import torch.nn as nn

board = chess.Board()

print(board)

moves = board.legal_moves

for move in moves:
    print(move)