import chess
import numpy as np
import torch
import torch.nn.functional as F

import encoder
from model import ChessNet


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    m = ChessNet().to(device)
    board = chess.Board()
    state = encoder.board_to_tensor(board).to(device)
    print(state.shape)
    fwd = m.forward(state)
    print(fwd)
