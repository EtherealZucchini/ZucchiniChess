import collections

import torch

import encoder
import chess
import torch.nn as nn
import chess.polyglot as polyglot

class MCTS():
    def __init__(self, body: nn.Module, value_head: nn.Module, policy_head: nn.module, num_simulations=800):
        self.nn_body = body
        self.policy_head = policy_head
        self.value_head = value_head
        self.num_simulations = num_simulations
        self.transposition_table = {}  # Your global cache of MCTSNodes

    def search(self, initial_board: chess.Board):
        """
        The main entry point. Runs the simulation loop and returns the best move.
        """
        # Create the root node for the current real-world board state
        root_node: MCTSNode = self._get_or_create_node(initial_board)

        for _ in range(self.num_simulations):
            # 1. We must use a COPY of the board for the search phase,
            # so we don't mess up the actual game board!
            search_board = initial_board.copy(stack=False)

            # 2. SELECTION: Walk down the tree, pushing moves to search_board
            leaf_node, search_path = self.select_leaf(root_node, search_board)

            # 3. EXPANSION & EVALUATION: Pass the synced search_board to the NN
            value = self.evaluate_and_expand(leaf_node, search_board)

            # 4. BACKPROPAGATION: Walk back up the search_path updating N and Q
            self.backpropagate(search_path, value)

        # After 800 simulations, pick the child of the root with the highest N (visit count)
        return self.get_best_move(root_node)

    def select_leaf(self, root_node: MCTSNode, global_board: chess.Board) -> tuple[MCTSNode, list[MCTSNode]]:
        """
        Traverses down the tree using PUCT, syncing the global_board as it goes.
        Returns the leaf node and the path taken to get there.
        """
        current_node = root_node
        search_path = [current_node]

        while current_node.children:
            # Pick best child via PUCT formula
            best_move, next_node = self._get_best_puct_child(current_node)

            # Keep our single board in sync with our mathematical traversal
            global_board.push(best_move)

            current_node = next_node
            search_path.append(current_node)

        return current_node, search_path

    def backpropagate(self, search_path: list[MCTSNode], value: float):
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value

    def evaluate_and_expand(self, leaf_node: MCTSNode, search_board: chess.Board) -> float:
        """
            Evaluates the leaf node using the Neural Network and expands its legal children.
            Returns the value (v) of the state from the perspective of the player to move.
            """
        # 1. TERMINAL CHECK: Is the game over?
        # claim_draw=True forces python-chess to recognize 50-move and 3-fold rules
        outcome = search_board.outcome(claim_draw=True) # TODO: this may be inefficient, check options
        if outcome is not None:
            if outcome.winner is None:
                return 0.0  # Draw
            elif outcome.winner == search_board.turn:
                return 1.0  # Current player won
            else:
                return -1.0  # Current player lost

        # 2. EVALUATE: Query the Neural Network
        # Convert the python-chess board into our 19-plane tensor
        state_tensor = encoder.board_to_tensor(search_board).unsqueeze(0)  # Add batch dimension
        legal_mask = encoder.get_legal_move_mask(search_board).unsqueeze(0)

        # Disable gradient tracking for massive speedup during self-play
        with torch.no_grad():
            encoding = self.nn_body(state_tensor)
            policy_logits = self.policy_head(encoding, legal_mask)
            value_tensor = self.value_head(encoding)

        # Extract the scalar value for backpropagation
        value = value_tensor.item()

        # 3. EXPAND: Create the children nodes
        # (Assume decode_policy converts the 4672-length tensor into a {chess.Move: float} dict)

        action_probs = encoder.decode_policy(policy_logits, search_board)

        for move, prob in action_probs.items():
            # Get the new repetition state for this child (using our O(1) logic)
            child_node = expand_node(search_board, prior=prob, parent=leaf_node, move=move, parent_rep_counter=leaf_node.rep_counter)

            # Attach it to the tree
            leaf_node.children[move] = child_node

        # 4. Return the predicted value to be sent up the tree
        return value

    def _get_best_puct_child(self, current_node):
        pass


def expand_node(global_board: chess.Board, parent: MCTSNode, move: chess.Move, parent_rep_counter: collections.Counter, prior) :
    """
    Executes a move and calculates the new repetition state in O(1) time.
    :returns: child_hash, child_counter, is_twofold, is_threefold
    """


    # 1. Inherit the parent's repetition history
    child_counter = parent_rep_counter.copy()

    # 2. Make the move
    global_board.push(move)


    # 3. The Golden Rule: Clear the dictionary on irreversible moves!
    if global_board.halfmove_clock == 0:
        #TODO: this doesn't account for castling (needed for threefold repetition), but we'll leave it for now
        child_counter.clear()

    # 4. Generate the current state's unique integer
    current_hash = polyglot.zobrist_hash(global_board)

    # 5. Increment the count for this specific position
    # we are explicitly counting repeated positions, not nodes, so it is ok that we don't include
    child_counter[current_hash] += 1

    to_play = 1 if global_board.turn == chess.WHITE else -1
    child = MCTSNode(prior=prior, physical_hash=current_hash, halfmove_clock=global_board.halfmove_clock, to_play=to_play, rep_counter=child_counter, parent=parent, move_from_parent=move)

    # Undo the move to leave the original board intact for other branches
    global_board.pop()

    return child


class MCTSNode(object):
    def __init__(self, *, prior: float, physical_hash: int, halfmove_clock: int, to_play: int, rep_counter: collections.Counter, parent: MCTSNode,
                 move_from_parent: chess.Move):
        self.parent = parent
        self.move_from_parent = move_from_parent
        self.children = {}

        # Calculate the unique state identifiers once upon creation
        self.physical_hash = physical_hash
        self.halfmove_clock = halfmove_clock


        #TODO: problem; we are trying to hash with the repcount, but we need the hash to get the repcount unless it is passed!
        self._hash = hash((self.physical_hash, self.halfmove_clock, rep_counter[self.physical_hash]))

        self.prior = prior
        self.visit_count = 0
        self.to_play = to_play
        self.value_sum = 0
        self.rep_counter = rep_counter # counter for visited nodes in the tree


    def expanded(self):
        return len(self.children)

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def __hash__(self):
        return self._hash

    def __eq__(self, other):

        if not isinstance(other, MCTSNode):
            return False

        return self._state_tuple == other._state_tuple