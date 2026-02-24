import torch
import chess
import numpy as np


def board_to_tensor(board):
    # 21 layers: 12 (pieces) + 1 (turn) + 4 (castling) + 1 (50-move) + 1 (en passant) + 2 (repetition)
    tensor = np.zeros((21, 8, 8), dtype=np.float32)

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

    # Layer 19: First Repetition (Warning: One more time makes it a draw)
    # is_repetition(2) means the current position has now appeared twice.
    if board.is_repetition(2):
        tensor[19, :, :] = 1.0

    # Layer 20: Threefold Repetition (The game is currently a draw)
    # is_repetition(3) means the position has appeared three times.
    if board.is_repetition(3):
        tensor[20, :, :] = 1.0

    return torch.from_numpy(tensor)

from chess import polyglot

def get_state_key(board) -> tuple[int, int, int]:
    """
    Creates a perfectly unique MCTS key combining the physical state
    (Zobrist), the 50-move rule, and the repetition count.
    """
    physical_hash: int = polyglot.zobrist_hash(board)
    halfmove_clock: int = board.halfmove_clock

    # Check repetition status
    rep_count = 1
    if board.is_repetition(3):
        rep_count = 3
    elif board.is_repetition(2):
        rep_count = 2

    # A tuple key is fast, memory-efficient, and perfectly state-aware
    return physical_hash, halfmove_clock, rep_count


def move_to_plane(move):
    """Maps a python-chess move to its AlphaZero 73-plane index (0-72)."""

    # Extract geometric deltas
    dx = chess.square_file(move.to_square) - chess.square_file(move.from_square)
    dy = chess.square_rank(move.to_square) - chess.square_rank(move.from_square)

    # 1. UNDERPROMOTIONS (Planes 64-72)
    # Note: Queen promotions are treated as standard directional moves below
    if move.promotion and move.promotion != chess.QUEEN:
        # Map dx (-1, 0, 1) to Direction Index (0, 1, 2)
        dir_idx = dx + 1

        # Map piece (KNIGHT=2, BISHOP=3, ROOK=4) to Piece Index (0, 1, 2)
        piece_idx = move.promotion - 2

        # 64 + (Piece * 3) + Direction
        return 64 + (piece_idx * 3) + dir_idx

    # 2. KNIGHT MOVES (Planes 56-63)
    if abs(dx) in [1, 2] and abs(dy) in [1, 2] and abs(dx) != abs(dy):
        # A rigid, clockwise mapping of the 8 knight jumps
        knight_moves = [
            (1, 2), (2, 1), (2, -1), (1, -2),
            (-1, -2), (-2, -1), (-2, 1), (-1, 2)
        ]
        return 56 + knight_moves.index((dx, dy))

    # 3. QUEEN MOVES (Planes 0-55)
    # Includes all standard pawn, king, and sliding piece moves
    distance = max(abs(dx), abs(dy))

    # Normalize the delta to get a 1-unit step direction
    step_x = dx // distance
    step_y = dy // distance

    # Map the 8 compass directions to indices 0-7
    compass = {
        (0, 1): 0,  # North
        (1, 1): 1,  # North-East
        (1, 0): 2,  # East
        (1, -1): 3,  # South-East
        (0, -1): 4,  # South
        (-1, -1): 5,  # South-West
        (-1, 0): 6,  # West
        (-1, 1): 7  # North-West
    }
    dir_idx = compass[(step_x, step_y)]

    # (Direction * 7) + (Distance - 1)
    return (dir_idx * 7) + (distance - 1)

def get_legal_move_mask(board):
    """
    Creates a boolean tensor of shape (4672,) where True indicates a legal move.
    """
    # Initialize a mask of all False
    mask = torch.zeros(4672, dtype=torch.bool)

    for move in board.legal_moves:
        plane_idx = move_to_plane(move)
        from_sq = move.from_square

        # Calculate the 1D index
        flat_index = (plane_idx * 64) + from_sq

        # Unmask the legal move
        mask[flat_index] = True

    return mask

