"""Quick debug of move indices."""

import chess

def move_to_index(move):
    from_square = move.from_square
    to_square = move.to_square

    if move.promotion is None:
        index = from_square * 64 + to_square
    else:
        promotion_offset = {
            chess.QUEEN: 0,
            chess.ROOK: 64,
            chess.BISHOP: 128,
            chess.KNIGHT: 192
        }
        base_index = 4096
        offset = promotion_offset.get(move.promotion, 0)
        promotion_index = from_square * 8 + (to_square % 8)
        index = base_index + offset + promotion_index

    return index

# Test
board = chess.Board()
print("Starting position legal moves:")

for i, move in enumerate(list(board.legal_moves)[:5]):
    idx = move_to_index(move)
    print(f"{move.uci()}: index={idx}, in_range(0,1968)={0 <= idx < 1968}")

print(f"\nMaximum normal move index: 63*64+63 = {63*64+63}")
print(f"This is {'OUTSIDE' if 63*64+63 >= 1968 else 'inside'} range [0, 1968)")
print("\n❌ BUG FOUND: Normal moves go up to index 4095, but we check for < 1968!")
