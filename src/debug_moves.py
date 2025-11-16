"""Debug script to check move encoding."""

import chess
from chess_utils import move_to_index, create_legal_move_mask

# Test with starting position
board = chess.Board()

print("Debugging move encoding for starting position:")
print(f"FEN: {board.fen()}\n")

legal_moves = list(board.legal_moves)
print(f"Total legal moves: {len(legal_moves)}\n")

# Check move indices
moves_in_range = 0
moves_out_range = 0

for move in legal_moves:
    idx = move_to_index(move)
    in_range = 0 <= idx < 1968

    if in_range:
        moves_in_range += 1
    else:
        moves_out_range += 1
        print(f"⚠️  Move {move.uci()} -> index {idx} (OUT OF RANGE!)")

print(f"\n✓ Moves in range [0, 1968): {moves_in_range}")
print(f"✗ Moves out of range: {moves_out_range}")

# Check mask
mask = create_legal_move_mask(board)
num_legal_in_mask = int(mask.sum())
print(f"\nLegal moves in mask: {num_legal_in_mask}")
print(f"Expected: {len(legal_moves)}")

if num_legal_in_mask != len(legal_moves):
    print("\n❌ MISMATCH! Some moves are not being encoded correctly!")
else:
    print("\n✅ All moves encoded correctly!")
