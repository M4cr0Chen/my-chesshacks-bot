"""Minimax search engine with alpha-beta pruning for chess."""

import time
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from chess import Board, Move
from .chess_utils import board_to_bitboard, move_to_index


class SearchEngine:
    """
    Minimax search engine with alpha-beta pruning.

    Features:
    - Alpha-beta pruning for efficient search
    - NN-based move ordering for better pruning
    - NN-based position evaluation at leaf nodes
    - Transposition table for caching evaluated positions
    """

    def __init__(self, model, device='cpu'):
        """
        Initialize search engine.

        Args:
            model: ChessNet model for evaluation and move ordering
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        self.transposition_table = {}

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
        # Clear transposition table for new search
        if not use_tt:
            self.transposition_table = {}

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
        }

        start_time = time.time()

        # Get legal moves
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            # No legal moves (checkmate or stalemate)
            return None, stats

        if len(legal_moves) == 1:
            # Only one legal move, no need to search
            stats['time_ms'] = (time.time() - start_time) * 1000
            return legal_moves[0], stats

        # Order moves using NN policy
        ordered_moves = self._order_moves(board, legal_moves)

        # Search each move
        best_move = ordered_moves[0]
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for move in ordered_moves:
            # Make move
            board.push(move)

            # Search with negamax (flip sign for opponent)
            score = -self._minimax(board, depth - 1, -beta, -alpha, stats)

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

    def _minimax(self, board: Board, depth: int, alpha: float, beta: float, stats: Dict[str, Any]) -> float:
        """
        Minimax search with alpha-beta pruning (negamax formulation).

        Args:
            board: Current position
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            stats: Search statistics dictionary

        Returns:
            Score for current position (from perspective of player to move)
        """
        stats['nodes_searched'] += 1

        # Check transposition table
        board_hash = hash(board.fen())
        if board_hash in self.transposition_table:
            tt_entry = self.transposition_table[board_hash]
            if tt_entry['depth'] >= depth:
                stats['tt_hits'] += 1
                return tt_entry['score']
        else:
            stats['tt_misses'] += 1

        # Terminal node checks
        if board.is_checkmate():
            # Checkmate - losing position
            # Return very negative score, adjusted by depth to prefer quick mates
            return -100.0 + (depth * 0.1)

        if board.is_stalemate() or board.is_insufficient_material():
            # Draw
            return 0.0

        # Fifty-move rule or repetition
        if board.is_fifty_moves() or board.is_repetition():
            return 0.0

        # Leaf node - evaluate with NN
        if depth <= 0:
            score = self._evaluate_position(board)

            # Store in transposition table
            self.transposition_table[board_hash] = {
                'score': score,
                'depth': depth
            }

            return score

        # Get legal moves
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            # No legal moves (shouldn't happen after checks above)
            return 0.0

        # Order moves using NN policy
        ordered_moves = self._order_moves(board, legal_moves)

        # Search moves
        best_score = float('-inf')

        for move in ordered_moves:
            # Make move
            board.push(move)

            # Recursive search (negamax)
            score = -self._minimax(board, depth - 1, -beta, -alpha, stats)

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
        self.transposition_table[board_hash] = {
            'score': best_score,
            'depth': depth
        }

        return best_score

    def _evaluate_position(self, board: Board) -> float:
        """
        Evaluate position using neural network value head.

        Args:
            board: Position to evaluate

        Returns:
            Score from perspective of player to move (range: approximately -1 to 1)
        """
        # Convert board to tensor
        position = board_to_bitboard(board)
        position_tensor = torch.FloatTensor(position).unsqueeze(0).to(self.device)

        # Get NN evaluation
        with torch.no_grad():
            _, value = self.model(position_tensor)
            score = value.item()

        # The value head outputs score from White's perspective
        # Need to flip for Black
        if not board.turn:  # Black to move
            score = -score

        return score

    def _order_moves(self, board: Board, legal_moves: list) -> list:
        """
        Order moves by NN policy probability (best moves first).

        This dramatically improves alpha-beta pruning efficiency by searching
        the most promising moves first.

        Args:
            board: Current position
            legal_moves: List of legal moves

        Returns:
            List of moves ordered by policy probability (descending)
        """
        # Convert board to tensor
        position = board_to_bitboard(board)
        position_tensor = torch.FloatTensor(position).unsqueeze(0).to(self.device)

        # Get NN policy
        with torch.no_grad():
            policy_logits, _ = self.model(position_tensor)
            policy = F.softmax(policy_logits, dim=1).cpu().numpy()[0]

        # Score each legal move
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

    def clear_tt(self):
        """Clear transposition table."""
        self.transposition_table = {}

    def get_tt_size(self) -> int:
        """Get current transposition table size."""
        return len(self.transposition_table)
