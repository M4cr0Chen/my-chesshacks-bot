"""Utility functions for chess training."""

import numpy as np
import chess
import torch


def board_to_bitboard(board):
    """
    Convert python-chess Board to bitboard tensor representation.

    Args:
        board: chess.Board object

    Returns:
        numpy array of shape (14, 8, 8)
        - Planes 0-11: Pieces (6 types × 2 colors)
        - Plane 12: Castling rights and turn
        - Plane 13: En passant square
    """
    bitboard = np.zeros((14, 8, 8), dtype=np.float32)

    # Encode pieces (12 planes)
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

    for color_idx, color in enumerate([chess.WHITE, chess.BLACK]):
        for piece_idx, piece_type in enumerate(piece_types):
            piece_mask = board.pieces(piece_type, color)
            plane_idx = color_idx * 6 + piece_idx

            for square in piece_mask:
                rank = square // 8
                file = square % 8
                bitboard[plane_idx, rank, file] = 1.0

    # Encode castling rights and turn (plane 12)
    bitboard[12, 0, 0] = float(board.has_kingside_castling_rights(chess.WHITE))
    bitboard[12, 0, 1] = float(board.has_queenside_castling_rights(chess.WHITE))
    bitboard[12, 0, 2] = float(board.has_kingside_castling_rights(chess.BLACK))
    bitboard[12, 0, 3] = float(board.has_queenside_castling_rights(chess.BLACK))
    bitboard[12, 0, 4] = float(board.turn)  # 1 for White, 0 for Black

    # Encode en passant (plane 13)
    if board.ep_square is not None:
        ep_rank = board.ep_square // 8
        ep_file = board.ep_square % 8
        bitboard[13, ep_rank, ep_file] = 1.0

    return bitboard


def fen_to_bitboard(fen):
    """Convert FEN string to bitboard tensor."""
    board = chess.Board(fen)
    return board_to_bitboard(board)


def move_to_index(move):
    """
    Convert chess.Move to policy index (0-4095).

    Encoding scheme:
    - Normal moves: from_square * 64 + to_square (0-4095)

    Note: Promotions are handled as normal moves with the promotion
    flag set in the chess.Move object. The encoding only uses the
    from_square and to_square, resulting in indices 0-4095.
    """
    from_square = move.from_square
    to_square = move.to_square

    # Simple encoding: all moves (including promotions) use from*64+to
    # Promotion information is stored in the move object itself
    index = from_square * 64 + to_square

    return index


def index_to_move(index, board):
    """
    Convert policy index to chess.Move.

    Args:
        index: Policy index (0-4095)
        board: chess.Board for validation

    Returns:
        chess.Move object or None if invalid
    """
    # Decode index to from and to squares
    from_square = index // 64
    to_square = index % 64

    # Check all legal moves for a match
    # This handles promotions correctly since we compare the actual move
    for legal_move in board.legal_moves:
        if legal_move.from_square == from_square and legal_move.to_square == to_square:
            return legal_move

    return None


def create_legal_move_mask(board):
    """
    Create binary mask for legal moves.

    Args:
        board: chess.Board object

    Returns:
        numpy array of shape (4096,) with 1 for legal moves, 0 otherwise
    """
    mask = np.zeros(4096, dtype=np.float32)

    for move in board.legal_moves:
        move_idx = move_to_index(move)
        if 0 <= move_idx < 4096:
            mask[move_idx] = 1.0

    return mask


def centipawn_to_win_prob(cp_score, scale=400):
    """
    Convert centipawn score to win probability using logistic function.

    Args:
        cp_score: Centipawn score (positive favors white)
        scale: Scaling factor (default 400)

    Returns:
        Win probability in [0, 1]
    """
    return 1.0 / (1.0 + 10 ** (-cp_score / scale))


def mate_to_value(mate_in_n):
    """
    Convert mate-in-N to a value score.

    Args:
        mate_in_n: Moves until mate (positive if we're delivering, negative if receiving)

    Returns:
        Value in [-1, 1]
    """
    if mate_in_n > 0:
        # We're delivering checkmate
        return 0.99
    elif mate_in_n < 0:
        # We're getting checkmated
        return -0.99
    else:
        return 0.0


def parse_stockfish_eval(eval_str):
    """
    Parse Stockfish evaluation string to value.

    Args:
        eval_str: String like "+150" (centipawns) or "#5" (mate in 5)

    Returns:
        Value in [-1, 1]
    """
    eval_str = str(eval_str).strip()

    if eval_str.startswith('#'):
        # Mate score
        try:
            mate_in = int(eval_str[1:])
            return mate_to_value(mate_in)
        except ValueError:
            return 0.0
    else:
        # Centipawn score
        try:
            cp = float(eval_str)
            # Convert to win probability, then scale to [-1, 1]
            win_prob = centipawn_to_win_prob(cp)
            return 2 * win_prob - 1  # Map [0,1] to [-1,1]
        except ValueError:
            return 0.0


def create_policy_target(board, best_move_uci):
    """
    Create one-hot policy target for the best move.

    Args:
        board: chess.Board object
        best_move_uci: Best move in UCI format (e.g., "e2e4")

    Returns:
        numpy array of shape (4096,) with 1 at best move index, 0 elsewhere
    """
    policy = np.zeros(4096, dtype=np.float32)

    try:
        move = chess.Move.from_uci(best_move_uci)
        from_square = move.from_square
        to_square = move.to_square

        # FIX: Some datasets use incorrect castling notation (e.g., "e1h1" instead of "e1g1")
        # Check if this looks like incorrect castling notation (king to rook on original square)
        # This is ALWAYS wrong - proper castling is king two squares toward the rook
        piece = board.piece_at(from_square)
        target = board.piece_at(to_square)

        if (piece and piece.piece_type == chess.KING and
            target and target.piece_type == chess.ROOK and
            piece.color == target.color):

            # This is incorrect castling notation - convert to correct notation:
            # e1h1 -> e1g1 (white kingside)
            # e1a1 -> e1c1 (white queenside)
            # e8h8 -> e8g8 (black kingside)
            # e8a8 -> e8c8 (black queenside)

            if from_square == chess.E1:  # White king
                if to_square == chess.H1:  # Kingside
                    move = chess.Move.from_uci("e1g1")
                elif to_square == chess.A1:  # Queenside
                    move = chess.Move.from_uci("e1c1")
            elif from_square == chess.E8:  # Black king
                if to_square == chess.H8:  # Kingside
                    move = chess.Move.from_uci("e8g8")
                elif to_square == chess.A8:  # Queenside
                    move = chess.Move.from_uci("e8c8")

        # Now check if the (possibly corrected) move is legal
        if move in board.legal_moves:
            move_idx = move_to_index(move)
            if 0 <= move_idx < 4096:
                policy[move_idx] = 1.0
    except (ValueError, AssertionError):
        # Invalid move format, return uniform distribution over legal moves
        mask = create_legal_move_mask(board)
        num_legal = mask.sum()
        if num_legal > 0:
            policy = mask / num_legal

    return policy
