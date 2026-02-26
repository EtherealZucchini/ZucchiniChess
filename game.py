#!/home/william/PycharmProjects/zucchinichess/.venv/bin/python
import chess
import torch

import mcts
import model
import sys

class ZucchiniChess():
    def __init__(self, model_path='chessnet_v1.pth'):
        self.model = model.ChessNet().to('cuda')
        self.model.load_state_dict(torch.load(model_path))
        self.board = chess.Board()
        self.engine = mcts.MCTS(nn=self.model)



    def uci_loop(self):

        while True:
            # Read a line from the GUI/User
            line = sys.stdin.readline()
            if not line:
                break

            line = line.strip()
            tokens = line.split()
            if not tokens:
                continue

            command = tokens[0].lower()

            # Handle Commands
            if command == "uci":
                print("id name Zucchinichess")
                print("id author William Eliot")
                print("uciok")
            elif command == "isready":
                print("readyok")
            elif command == "ucinewgame":
                self.board = chess.Board()
                self.engine = mcts.MCTS(nn=self.model)
                print(f"info string new game", flush=True)


            elif command == "position":
                self.handle_position(tokens[1:])
            elif command == "go":
                self.handle_go(tokens[1:])
            elif command == "quit":
                break

            # Ensure the GUI sees your response immediately
            sys.stdout.flush()


    def handle_go(self, params):
        # Default search settings
        search_params = {
            'depth': None,
            'wtime': None,
            'btime': None,
            'movetime': None,
            'infinite': False
        }

        print(f"info string search params:", flush=True)
        print(f"info string {search_params}")
        print(f"info string starting position \n{self.board}", flush=True)

        # Iterate through tokens and grab the value after the keyword
        i = 0
        while i < len(params):
            token = params[i]
            if token == "depth":
                search_params['depth'] = int(params[i + 1])
                i += 1
            elif token == "wtime":
                search_params['wtime'] = int(params[i + 1])
                i += 1
            elif token == "infinite":
                search_params['infinite'] = True
            # ... add other UCI flags similarly
            i += 1
        print(f"info string pondering move...", flush=True)
        # Now trigger your search algorithm using search_params
        try:
            best_move = self.engine.search(self.board)
            print(f'bestmove {best_move.uci()}', flush=True)
        except Exception as e:
            print(f"info string {e}", flush=True)
        self.board.push(best_move)
        self.engine.update_with_move(best_move)



    def handle_position(self, params):
        # Example using string joining to reconstruct FEN
        print(f"info string handling position...", flush=True)

        try:

            if params[0] == "startpos":
                # setup_start_board()
                pointer = 1
            elif params[0] == "fen":
                # FEN strings have spaces, so we grab the next 6 tokens
                fen_str = " ".join(params[1:7])
                # setup_fen_board(fen_str)
                pointer = 7

            # Check if there are moves to play

            #if pointer < len(params) and params[pointer] == "moves":
            #    for move in params[pointer + 1:]:
            #        self.board.push(chess.Move.from_uci(move))
            # just play the last move, we maintain state internally

            move = chess.Move.from_uci(params[-1])
            self.board.push(move)
            self.engine.update_with_move(move)

        except Exception as e:
            print(f"info string {e}", flush=True)
        print(f"info string done", flush=True)


if __name__ == "__main__":
    game = ZucchiniChess()
    game.uci_loop()