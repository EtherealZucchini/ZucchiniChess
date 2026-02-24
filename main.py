import chess
import numpy as np
import torch
import torch.nn.functional as F

import encoder
from model import ChessNetBody, ChessNetValue, ChessNetPolicy

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    body = ChessNetBody().to(device)
    value = ChessNetValue().to(device)
    policy = ChessNetPolicy().to(device)
    board = chess.Board()
    print(board.fen())

    state = encoder.board_to_tensor(board).view(-1, 21, 8, 8).to(device)
    print(state.shape)
    fwd = body.forward(state)
    v = value.forward(fwd)
    print(v)

    key = encoder.get_state_key(board)
    print(key)
