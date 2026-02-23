import torch
import chess
import numpy as np


def board_to_tensor(board):
    # 19 layers: 12 (pieces) + 1 (turn) + 4 (castling) + 1 (50-move) + 1 (en passant)
    tensor = np.zeros((19, 8, 8), dtype=np.float32)

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

    # Layer 12: Whose turn is it? (All 1s if White, 0s if Black)
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    # Layers 13-16: Castling Rights (All 1s if true)
    if board.has_kingside_castling_rights(chess.WHITE):  tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE): tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):  tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK): tensor[16, :, :] = 1.0

    # Layer 17: Half-move clock (normalized to 0.0 - 1.0)
    tensor[17, :, :] = board.halfmove_clock / 100.0

    # Layer 18: En Passant Target Square
    if board.ep_square is not None:
        row, col = divmod(board.ep_square, 8)
        tensor[18, row, col] = 1.0

    return torch.from_numpy(tensor)
