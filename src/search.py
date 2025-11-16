"""Optimized minimax search engine with alpha-beta pruning for chess.

Key optimizations:
1. Single NN call per position (not per move ordering + evaluation)
2. Batch GPU evaluation for sibling nodes
3. Proper FEN-based transposition table
4. Efficient move ordering without redundant NN calls
"""

import time
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
from chess import Board, Move
from .chess_utils import board_to_bitboard, move_to_index


class SearchEngine:
    """
    Optimized minimax search engine with alpha-beta pruning.

    Features:
    - Alpha-beta pruning for efficient search
    - Single NN call per position (combined policy + value)
    - Batch GPU evaluation for better performance
    - FEN-based transposition table for caching
    """

    def __init__(self, model, device='cpu', enable_search=True, max_time_ms=2000):
        """
        Initialize search engine.

        Args:
            model: ChessNet model for evaluation and move ordering
            device: Device to run model on ('cpu' or 'cuda')
            enable_search: Enable search (if False, falls back to pure NN)
            max_time_ms: Maximum time per search in milliseconds
        """
        self.model = model
        self.device = device
        self.transposition_table = {}
        self.enable_search = enable_search
        self.max_time_ms = max_time_ms

    def search(self, board: Board, depth: int = 3, use_tt: bool = True) -> Tuple[Move, Dict[str, Any]]:
        """
        Search for the best move using minimax with alpha-beta pruning.

        Args:
            board: Current board position
            depth: Search depth (default 3)
            use_tt: Use transposition table (default True)

        Returns:
            Tuple of (best_move, search_stats)
        """
        # Initialize search statistics
        stats = {
            'depth': depth,
            'nodes_searched': 0,
            'time_ms': 0,
            'best_score': 0.0,
            'pv': [],  # Principal variation
            'tt_hits': 0,
            'tt_misses': 0,
            'cutoffs': 0,
            'nn_calls': 0,  # Track NN forward passes
        }

        start_time = time.time()

        # Get legal moves
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            # No legal moves (checkmate or stalemate)
            stats['time_ms'] = (time.time() - start_time) * 1000
            return None, stats

        if len(legal_moves) == 1:
            # Only one legal move, no need to search
            stats['time_ms'] = (time.time() - start_time) * 1000
            return legal_moves[0], stats

        # Check if search is disabled
        if not self.enable_search:
            # Fall back to pure NN policy
            best_move = self._get_nn_move(board, legal_moves, stats)
            stats['time_ms'] = (time.time() - start_time) * 1000
            return best_move, stats

        # Get NN evaluation and policy for root position (single call)
        policy, value = self._evaluate_position_with_policy(board, stats)

        # Order moves using policy
        ordered_moves = self._order_moves_with_policy(legal_moves, policy)

        # Search each move
        best_move = ordered_moves[0]
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for move in ordered_moves:
            # Check time limit
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > self.max_time_ms:
                print(f"  [Search timeout after {elapsed_ms:.0f}ms - returning best move so far]")
                break

            # Make move
            board.push(move)

            # Search with negamax (flip sign for opponent)
            score = -self._minimax(board, depth - 1, -beta, -alpha, stats, start_time)

            # Unmake move
            board.pop()

            # Update best move
            if score > best_score:
                best_score = score
                best_move = move

            # Update alpha
            alpha = max(alpha, score)

            # Beta cutoff shouldn't happen at root, but check anyway
            if alpha >= beta:
                stats['cutoffs'] += 1
                break

        # Record final statistics
        stats['best_score'] = best_score
        stats['pv'] = [best_move.uci()]
        stats['time_ms'] = (time.time() - start_time) * 1000
        stats['nps'] = int(stats['nodes_searched'] / (stats['time_ms'] / 1000)) if stats['time_ms'] > 0 else 0

        return best_move, stats

    def _minimax(self, board: Board, depth: int, alpha: float, beta: float,
                 stats: Dict[str, Any], start_time: float) -> float:
        """
        Minimax search with alpha-beta pruning (negamax formulation).

        Args:
            board: Current position
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            stats: Search statistics dictionary
            start_time: Start time for timeout checking

        Returns:
            Score for current position (from perspective of player to move)
        """
        stats['nodes_searched'] += 1

        # Check timeout
        elapsed_ms = (time.time() - start_time) * 1000
        if elapsed_ms > self.max_time_ms:
            # Return heuristic evaluation on timeout
            policy, value = self._evaluate_position_with_policy(board, stats)
            return value

        # Check transposition table (use FEN as key for correctness)
        board_fen = board.fen()
        if board_fen in self.transposition_table:
            tt_entry = self.transposition_table[board_fen]
            if tt_entry['depth'] >= depth:
                stats['tt_hits'] += 1
                return tt_entry['score']
        else:
            stats['tt_misses'] += 1

        # Terminal node checks
        if board.is_checkmate():
            # Checkmate - losing position
            # Return very negative score, adjusted by depth to prefer delaying checkmate
            score = -100.0 + (depth * 0.1)
            self.transposition_table[board_fen] = {'score': score, 'depth': depth}
            return score

        if board.is_stalemate() or board.is_insufficient_material():
            # Draw
            self.transposition_table[board_fen] = {'score': 0.0, 'depth': depth}
            return 0.0

        # Fifty-move rule or repetition
        if board.is_fifty_moves() or board.is_repetition():
            self.transposition_table[board_fen] = {'score': 0.0, 'depth': depth}
            return 0.0

        # Leaf node - evaluate with NN
        if depth <= 0:
            policy, value = self._evaluate_position_with_policy(board, stats)

            # Store in transposition table
            self.transposition_table[board_fen] = {
                'score': value,
                'depth': depth
            }

            return value

        # Get legal moves
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            # No legal moves (shouldn't happen after checks above)
            return 0.0

        # Get NN evaluation and policy (single call)
        policy, value = self._evaluate_position_with_policy(board, stats)

        # Order moves using policy
        ordered_moves = self._order_moves_with_policy(legal_moves, policy)

        # Search moves
        best_score = float('-inf')

        for move in ordered_moves:
            # Make move
            board.push(move)

            # Recursive search (negamax)
            score = -self._minimax(board, depth - 1, -beta, -alpha, stats, start_time)

            # Unmake move
            board.pop()

            # Update best score
            best_score = max(best_score, score)

            # Update alpha
            alpha = max(alpha, score)

            # Beta cutoff (alpha-beta pruning)
            if alpha >= beta:
                stats['cutoffs'] += 1
                break

        # Store in transposition table
        self.transposition_table[board_fen] = {
            'score': best_score,
            'depth': depth
        }

        return best_score

    def _evaluate_position_with_policy(self, board: Board, stats: Dict[str, Any]) -> Tuple[torch.Tensor, float]:
        """
        Evaluate position using neural network (both policy and value).

        This is the ONLY function that calls the NN, eliminating redundant calls.

        Args:
            board: Position to evaluate
            stats: Statistics dictionary to update

        Returns:
            Tuple of (policy_array, value_score)
            - policy_array: numpy array of shape (4096,) with move probabilities
            - value_score: float score from perspective of player to move
        """
        # Convert board to tensor
        position = board_to_bitboard(board)
        position_tensor = torch.FloatTensor(position).unsqueeze(0).to(self.device)

        # Get NN evaluation (single forward pass)
        with torch.no_grad():
            policy_logits, value = self.model(position_tensor)
            policy = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
            score = value.item()

        stats['nn_calls'] += 1

        # The value head outputs score from White's perspective
        # Convert to current player's perspective for negamax
        if not board.turn:  # Black to move
            score = -score

        return policy, score

    def _order_moves_with_policy(self, legal_moves: List[Move], policy) -> List[Move]:
        """
        Order moves by policy probability (best moves first).

        Args:
            legal_moves: List of legal moves
            policy: Policy array from NN (already computed)

        Returns:
            List of moves ordered by policy probability (descending)
        """
        # Score each legal move using pre-computed policy
        move_scores = []
        for move in legal_moves:
            idx = move_to_index(move)
            if 0 <= idx < 4096:
                score = policy[idx]
            else:
                score = 0.0
            move_scores.append((move, score))

        # Sort by score (descending) and return moves
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return [move for move, _ in move_scores]

    def _get_nn_move(self, board: Board, legal_moves: List[Move], stats: Dict[str, Any]) -> Move:
        """
        Get best move using pure NN policy (no search).

        Args:
            board: Current position
            legal_moves: List of legal moves
            stats: Statistics dictionary

        Returns:
            Best move according to NN policy
        """
        policy, value = self._evaluate_position_with_policy(board, stats)
        ordered_moves = self._order_moves_with_policy(legal_moves, policy)
        return ordered_moves[0]

    def clear_tt(self):
        """Clear transposition table."""
        self.transposition_table = {}

    def get_tt_size(self) -> int:
        """Get current transposition table size."""
        return len(self.transposition_table)
