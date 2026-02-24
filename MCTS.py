import collections
import math

import torch

import encoder
import chess
import torch.nn as nn
import chess.polyglot as polyglot

class MCTS:
    def __init__(self, *,
                 nn: nn.Module,
                 num_simulations=800,
                 max_turns=200,
                 device='cuda'):
        self.nn = nn
        self.num_simulations = num_simulations
        self.transposition_table = {}
        self.device = device
        self.root: MCTSNode | None = None
        self._turn = 1
        self._max_turns = max_turns

    def terminate_early(self):
        return self._turn >= self._max_turns

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

    def get_best_move(self, root_node: MCTSNode):
        max_visit_count = 0
        candidate = None
        for move, (child, prior) in root_node.children.items():
            if child.visit_count > max_visit_count:
                max_visit_count = child.visit_count
                candidate = move
        return candidate

    def select_leaf(self, root_node: MCTSNode, global_board: chess.Board) -> tuple[MCTSNode, list[MCTSNode]]:
        """
        Traverses down the tree using PUCT, syncing the global_board as it goes.
        Returns the leaf node and the path taken to get there.
        """
        current_node = root_node
        search_path = [current_node]

        while current_node.children:
            # Pick best child via PUCT formula
            best_move, next_node = current_node.get_best_puct_child()

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
        if (leaf_node.rep_counter[leaf_node.physical_hash] >= 3
                or search_board.halfmove_clock >= 100
                or self._turn >= self._max_turns):
            return 0.0
        outcome = search_board.outcome(claim_draw=False)
        if outcome is not None:
            if outcome.winner is None:
                return 0.0  # Draw
            elif outcome.winner == search_board.turn:
                return 1.0  # Current player won
            else:
                return -1.0  # Current player lost

        # 2. EVALUATE: Query the Neural Network
        # Convert the python-chess board into our 19-plane tensor
        state_tensor = encoder.board_to_tensor(search_board, leaf_node.rep_count()).unsqueeze(0).to(self.device) # Add batch dimension
        legal_mask = encoder.get_legal_move_mask(search_board).unsqueeze(0).to(self.device)

        # Disable gradient tracking for massive speedup during self-play
        with torch.no_grad():
            value_tensor, policy_logits = self.nn(state_tensor, legal_mask)

        # Extract the scalar value for backpropagation
        value = value_tensor.item()

        # 3. EXPAND: Create the children nodes
        # (Assume decode_policy converts the 4672-length tensor into a {chess.Move: float} dict)

        action_probs = encoder.decode_policy(policy_logits, search_board)

        for move, prob in action_probs.items():
            # Pass the transposition table to the expansion function
            child_node = expand_node(
                global_board=search_board,
                prior=prob,
                move=move,
                parent_rep_counter=leaf_node.rep_counter,
                transposition_table=self.transposition_table
            )

            # Attach it to the tree
            leaf_node.children[move] = (child_node, prob)

        # 4. Return the predicted value to be sent up the tree
        return value

    def update_with_move(self, move: chess.Move):
        """
        Steps the MCTS root forward along the played move, preserving history.
        """
        self._turn += 1
        if self.root is not None and move in self.root.children:
            self.root, _ = self.root.children[move]
        else:
            # If the move wasn't in our tree (e.g., turn 1, or an unsearched opponent move),
            # we force a hard reset.
            self.root = None
        self.transposition_table.clear()

    def _get_or_create_node(self, initial_board: chess.Board):
        if self.root is not None:
            return self.root

        physical_hash = polyglot.zobrist_hash(initial_board)
        root_counter = collections.Counter()
        root_counter[physical_hash] = 1

        root = MCTSNode(physical_hash=physical_hash, halfmove_clock=initial_board.halfmove_clock,
                        rep_counter=root_counter)

        self.root = root

        # Register the new root in the transposition table
        self.transposition_table[root._state_tuple] = root

        return root


def expand_node(global_board: chess.Board,
                move: chess.Move,
                parent_rep_counter: collections.Counter,
                prior: float,
                transposition_table: dict):
    """
    Executes a move, checks the transposition table, and either returns the cached node
    or creates a new one in O(1) time.
    """
    # 1. Inherit the parent's repetition history
    child_counter = parent_rep_counter.copy()

    # 2. Make the move
    global_board.push(move)

    # 3. Clear the dictionary on irreversible moves
    if global_board.halfmove_clock == 0:
        child_counter.clear()

    # 4. Generate state identifiers
    current_hash = polyglot.zobrist_hash(global_board)
    child_counter[current_hash] += 1
    halfmove_clock = global_board.halfmove_clock

    # 5. Check the Transposition Table!
    state_tuple = (current_hash, halfmove_clock, child_counter[current_hash])
    if state_tuple in transposition_table:
        global_board.pop()
        return transposition_table[state_tuple]

    # 6. If not found, create the new node
    child = MCTSNode(physical_hash=current_hash, halfmove_clock=halfmove_clock, rep_counter=child_counter)

    # 7. Add the new node to the cache before returning
    transposition_table[state_tuple] = child

    global_board.pop()
    return child


class MCTSNode(object):
    def __init__(self, *, physical_hash: int, halfmove_clock: int, rep_counter: collections.Counter):
        self.children: dict[chess.Move, tuple[MCTSNode, float]] = {}

        # Calculate the unique state identifiers once upon creation
        self.physical_hash = physical_hash
        self.halfmove_clock = halfmove_clock

        self._state_tuple = (self.physical_hash, self.halfmove_clock, rep_counter[self.physical_hash])
        self._hash = hash(self._state_tuple)

        self.visit_count = 0
        self.value_sum = 0
        self.rep_counter = rep_counter # counter for visited nodes in the tree

    def rep_count(self):
        """
        Helper function to calculate get the number of repetitions using own physical hash
        :return:
        """
        return self.rep_counter[self.physical_hash]

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

    def get_best_puct_child(self) -> tuple[chess.Move, MCTSNode]:
        best_puct = -float('inf')
        best_move = None
        best_child = None

        # DeepMind's dynamic PUCT constants for chess (from the pseudocode)
        pb_c_base = 19652.0
        pb_c_init = 1.25


        # Calculate the dynamic exploration base factor once for the parent
        pb_c = math.log((self.visit_count + pb_c_base + 1.0) / pb_c_base) + pb_c_init
        pb_c *= math.sqrt(self.visit_count)

        for move, (child, prior) in self.children.items():
            # 1. EXPLOITATION: Q-Value
            # We negate the child's value because it is from the opponent's perspective.
            # If the child has 0 visits, its Q-value is 0.0.
            q_value = -child.value() if child.visit_count > 0 else 0.0

            # 2. EXPLORATION: U-Value
            # Prior * sqrt(Parent Visits) / (1 + Child Visits)
            u_value = pb_c * (prior / (child.visit_count + 1.0))

            # 3. COMBINE
            puct_score = q_value + u_value

            # Keep track of the highest score
            if puct_score > best_puct:
                best_puct = puct_score
                best_move = move
                best_child = child

        return best_move, best_child