import chess
import torch.nn as nn


class ChessGame():
    def __init__(self):
        self.board = chess.Board()