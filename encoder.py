import torch
import chess
import numpy as np


def board_to_tensor(board):
    # 12 layers (6 pieces x 2 colors) on an 8x8 grid
    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    # Mapping piece types to layer indices
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }

    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        # Shift index by 6 for black pieces
        index = piece_map[piece.piece_type] + (6 if piece.color == chess.BLACK else 0)
        tensor[index][row][col] = 1.0

    return torch.from_numpy(tensor)
