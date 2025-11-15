from .utils import chess_manager, GameContext
from .model import ChessNet
from .chess_utils import board_to_bitboard, create_legal_move_mask, move_to_index
from chess import Move
import torch
import torch.nn.functional as F
import os

# Load model at startup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ChessNet(num_residual_blocks=5, num_filters=64)

model_path = os.path.join(os.path.dirname(__file__), 'weights', 'model.pt')
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print(f"✓ Model loaded! Accuracy: {checkpoint.get('val_accuracy', 0):.2%}")

@chess_manager.entrypoint
def neural_chess_bot(ctx: GameContext) -> Move:
    """Neural network chess bot."""
    board = ctx.board

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

    # Select best move
    legal_moves = list(board.legal_moves)
    move_probs = {}

    # DEBUG: Check what's happening
    debug_info = {
        'total_legal': len(legal_moves),
        'indices_in_range': 0,
        'indices_out_range': 0,
        'total_prob': 0.0
    }

    for move in legal_moves:
        idx = move_to_index(move)
        if 0 <= idx < 4096:
            move_probs[move] = float(policy[idx])
            debug_info['indices_in_range'] += 1
            debug_info['total_prob'] += float(policy[idx])
        else:
            debug_info['indices_out_range'] += 1
            print(f"⚠️  Move {move.uci()} has index {idx} (out of range)")

    print(f"\n🔍 DEBUG INFO:")
    print(f"  Legal moves: {debug_info['total_legal']}")
    print(f"  Indices in range: {debug_info['indices_in_range']}")
    print(f"  Indices out of range: {debug_info['indices_out_range']}")
    print(f"  Total probability mass: {debug_info['total_prob']:.6f}")
    print(f"  Policy output shape: {policy.shape}")
    print(f"  Policy min/max: [{policy.min():.6f}, {policy.max():.6f}]")
    print(f"  Policy sum: {policy.sum():.6f}")

    # Log probabilities for visualization
    total = sum(move_probs.values())
    if total > 0:
        move_probs = {m: p/total for m, p in move_probs.items()}

    ctx.logProbabilities(move_probs)

    # Return best move
    if move_probs:
        best_move = max(move_probs.items(), key=lambda x: x[1])[0]
        print(f"\n{'='*60}")
        print(f"🤖 Bot is making a move!")
        print(f"FEN: {board.fen()}")
        print(f"Selected move: {best_move.uci()} (probability: {move_probs[best_move]:.2%})")
        print(f"Position value: {value.item():.4f}")
        print(f"Top 5 moves:")
        for i, (move, prob) in enumerate(sorted(move_probs.items(), key=lambda x: x[1], reverse=True)[:5], 1):
            print(f"  {i}. {move.uci()}: {prob:.2%}")
        print(f"{'='*60}\n")
        return best_move
    else:
        fallback = legal_moves[0]
        print(f"⚠️  No valid moves found in policy, using fallback: {fallback.uci()}")
        return fallback

@chess_manager.reset
def reset_state(ctx: GameContext):
    """Reset any state (not needed for pure NN)."""
    pass