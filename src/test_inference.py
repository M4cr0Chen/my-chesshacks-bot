"""Test inference directly."""

import chess
import torch
import torch.nn.functional as F
from model import ChessNet
from chess_utils import board_to_bitboard, create_legal_move_mask, move_to_index
import os

# Load model
device = 'cpu'
model = ChessNet(num_residual_blocks=5, num_filters=64)
model_path = os.path.join(os.path.dirname(__file__), 'weights', 'model.pt')
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print("Model loaded successfully!")
print(f"Validation accuracy: {checkpoint.get('val_accuracy', 0):.2%}\n")

# Test position
board = chess.Board()
print(f"Testing position: {board.fen()}")
print(f"Legal moves: {len(list(board.legal_moves))}\n")

# Convert board to tensor
position = board_to_bitboard(board)
position_tensor = torch.FloatTensor(position).unsqueeze(0).to(device)

# Get NN prediction
with torch.no_grad():
    policy_logits, value = model(position_tensor)

    # Mask illegal moves
    legal_mask = torch.FloatTensor(create_legal_move_mask(board)).to(device)
    policy_logits = policy_logits.masked_fill(legal_mask == 0, -1e9)

    # Softmax to get probabilities
    policy = F.softmax(policy_logits, dim=1).cpu().numpy()[0]

print(f"Policy output shape: {policy.shape}")
print(f"Policy sum: {policy.sum():.6f}")
print(f"Policy min/max: [{policy.min():.6f}, {policy.max():.6f}]")
print(f"Value: {value.item():.4f}\n")

# Get move probabilities
legal_moves = list(board.legal_moves)
move_probs = {}

for move in legal_moves:
    idx = move_to_index(move)
    if 0 <= idx < 1968:
        move_probs[move] = float(policy[idx])

print(f"Move probabilities collected: {len(move_probs)}")
print(f"Total probability mass: {sum(move_probs.values()):.6f}\n")

if move_probs:
    print("Top 5 moves:")
    for i, (move, prob) in enumerate(sorted(move_probs.items(), key=lambda x: x[1], reverse=True)[:5], 1):
        print(f"  {i}. {move.uci()}: {prob:.6f}")

    best_move = max(move_probs.items(), key=lambda x: x[1])[0]
    print(f"\nBest move: {best_move.uci()}")
else:
    print("❌ NO MOVES FOUND - This is the bug!")
