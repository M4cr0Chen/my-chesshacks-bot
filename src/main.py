from .utils import chess_manager, GameContext
from .model import ChessNet
from .chess_utils import board_to_bitboard, create_legal_move_mask, move_to_index
from .model_loader import load_model_checkpoint
from chess import Move
import torch
import torch.nn.functional as F
import os
import time

# Load model at startup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ChessNet(num_residual_blocks=15, num_filters=256)

# Download model from Hugging Face if needed, then load
checkpoint = load_model_checkpoint(device=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print(f"✓ Model loaded! Accuracy: {checkpoint.get('val_accuracy', 0):.2%}")
print(f"✓ Using pure neural network policy (no search) for maximum speed!")

@chess_manager.entrypoint
def neural_chess_bot(ctx: GameContext) -> Move:
    """Pure neural network chess bot - no search for maximum speed."""
    start_time = time.time()
    board = ctx.board
    time_left = ctx.timeLeft  # in milliseconds

    # Get legal moves
    legal_moves = list(board.legal_moves)
    
    if not legal_moves:
        print("No legal moves available - game over")
        return None
    
    if len(legal_moves) == 1:
        # Only one legal move, return immediately
        print(f"Only one legal move: {legal_moves[0].uci()}")
        return legal_moves[0]

    # Get NN policy and value with single forward pass
    position = board_to_bitboard(board)
    position_tensor = torch.FloatTensor(position).unsqueeze(0).to(device)

    with torch.no_grad():
        policy_logits, value = model(position_tensor)

        # Mask illegal moves
        legal_mask = torch.FloatTensor(create_legal_move_mask(board)).to(device)
        policy_logits = policy_logits.masked_fill(legal_mask == 0, -1e9)

        # Softmax to get probabilities
        policy = F.softmax(policy_logits, dim=1).cpu().numpy()[0]

    # Calculate move probabilities
    move_probs = {}
    for move in legal_moves:
        idx = move_to_index(move)
        if 0 <= idx < 4096:
            move_probs[move] = float(policy[idx])

    # Normalize probabilities
    total = sum(move_probs.values())
    if total > 0:
        move_probs = {m: p/total for m, p in move_probs.items()}

    # Select best move (highest probability)
    best_move = max(move_probs, key=move_probs.get)
    
    # Log probabilities for visualization
    ctx.logProbabilities(move_probs)

    # Calculate inference time
    inference_time_ms = (time.time() - start_time) * 1000

    # Print detailed information
    print(f"\n{'='*60}")
    print(f"🤖 Bot is making a move with PURE NN POLICY (no search)!")
    print(f"FEN: {board.fen()}")
    print(f"\n⏱️  Time Management:")
    print(f"  Time remaining: {time_left/1000:.1f}s")
    print(f"  Inference time: {inference_time_ms:.1f}ms")
    print(f"  Move: {board.fullmove_number}")
    print(f"\n🧠 Neural Network Evaluation:")
    print(f"  Position value: {value.item():.4f}")
    print(f"  Legal moves: {len(legal_moves)}")
    print(f"\n🎯 Selected move: {best_move.uci()}")
    print(f"  NN policy probability: {move_probs.get(best_move, 0):.2%}")
    print(f"\n📋 Top 5 moves by NN policy:")
    for i, (move, prob) in enumerate(sorted(move_probs.items(), key=lambda x: x[1], reverse=True)[:5], 1):
        marker = "★" if move == best_move else " "
        print(f"  {marker} {i}. {move.uci()}: {prob:.2%}")
    print(f"{'='*60}\n")

    return best_move

@chess_manager.reset
def reset_state(ctx: GameContext):
    """Reset state between games (no search state to clear)."""
    print("Game reset - ready for next game!")