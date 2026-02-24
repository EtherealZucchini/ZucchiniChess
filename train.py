import os

import chess.pgn
import torch
import torch.nn as nn
import torch.optim as optim
import encoder
from model import ChessNet


def generate_batches_from_pgn(pgn_path: str, batch_size: int = 32):
    """
    Reads a PGN file and yields batched tensors for training.
    """
    states, masks, target_values, target_policies = [], [], [], []

    with open(pgn_path, "r") as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break  # End of file

            # 1. Determine the game outcome (z) from White's perspective
            result = game.headers.get("Result", "*")
            if result == "1-0":
                game_outcome = 1.0
            elif result == "0-1":
                game_outcome = -1.0
            elif result == "1/2-1/2":
                game_outcome = 0.0
            else:
                continue  # Skip games with unknown results

            board = game.board()

            # 2. Iterate through the game moves
            for move in game.mainline_moves():
                # --- State & Mask ---
                state_tensor = encoder.board_to_tensor(board)
                legal_mask = encoder.get_legal_move_mask(board)

                # --- Target Value ---
                # Value must be from the perspective of the player whose turn it is
                target_v = game_outcome if board.turn == chess.WHITE else -game_outcome

                # --- Target Policy ---
                # Create a one-hot vector for the move the GM played
                target_p = torch.zeros(4672, dtype=torch.float32)
                plane_idx = encoder.move_to_plane(move)
                flat_index = (plane_idx * 64) + move.from_square
                target_p[flat_index] = 1.0

                # Append to our current batch
                states.append(state_tensor)
                masks.append(legal_mask)
                target_values.append(torch.tensor([target_v], dtype=torch.float32))
                target_policies.append(target_p)

                # Step the board forward
                board.push(move)

                # If we hit the batch size, yield the tensors and clear the lists
                if len(states) >= batch_size:
                    yield (
                        torch.stack(states),
                        torch.stack(masks),
                        torch.stack(target_values),
                        torch.stack(target_policies)
                    )
                    states, masks, target_values, target_policies = [], [], [], []


def train_network(model_path="chessnet_v1.pth", pgn_path="training_data.pgn"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}...")

    # Initialize the master network and optimizer
    model = ChessNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # weight_decay is L2 regularization
    mse_loss_fn = nn.MSELoss()

    # Put model in training mode (activates BatchNorm tracking)
    model.train()

    # Make sure you have a file named 'training_data.pgn' in your directory
    # You can download database dumps from Lichess or KingBase
    batch_generator = generate_batches_from_pgn(pgn_path=pgn_path, batch_size=64)

    for batch_idx, (states, masks, target_values, target_policies) in enumerate(batch_generator):
        states = states.to(device)
        masks = masks.to(device)
        target_values = target_values.to(device)
        target_policies = target_policies.to(device)

        # 1. Zero gradients
        optimizer.zero_grad()

        # 2. Forward pass

        pred_values, pred_policies = model(states, masks)

        # 3. Calculate Value Loss (Mean Squared Error)
        value_loss = mse_loss_fn(pred_values, target_values)

        # 4. Calculate Policy Loss (Cross Entropy)
        # Because your policy head outputs softmax probabilities, we use: -sum(target * log(pred))
        # We add 1e-8 to prevent taking the log of absolute zero, which returns NaN
        policy_loss = -torch.sum(target_policies * torch.log(pred_policies + 1e-8), dim=1).mean()

        # 5. Combine losses
        total_loss = value_loss + policy_loss

        # 6. Backpropagate and step
        total_loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(
                f"Batch {batch_idx} | Total Loss: {total_loss.item():.4f} (V_loss: {value_loss.item():.4f}, P_loss: {policy_loss.item():.4f})")

    # Save the trained weights!
    torch.save(model.state_dict(), model_path)
    print("Training complete. Model saved.")


if __name__ == "__main__":
    path = os.path.join(os.getcwd(), "Grischuk.pgn")
    train_network(pgn_path=path)