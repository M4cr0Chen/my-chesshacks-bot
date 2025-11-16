from .utils import chess_manager, GameContext
from .model import ChessNet
from .chess_utils import board_to_bitboard, create_legal_move_mask, move_to_index
from .search import SearchEngine
from .model_loader import load_model_checkpoint
from chess import Move
import torch
import torch.nn.functional as F
import os

# Load model at startup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ChessNet(num_residual_blocks=15, num_filters=256)

# Download model from Hugging Face if needed, then load
checkpoint = load_model_checkpoint(device=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print(f"✓ Model loaded! Accuracy: {checkpoint.get('val_accuracy', 0):.2%}")

# Initialize search engine
search_engine = SearchEngine(model, device=device)
SEARCH_DEPTH = 3  # Configurable search depth

print(f"✓ Search engine initialized! Depth: {SEARCH_DEPTH}")

@chess_manager.entrypoint
def neural_chess_bot(ctx: GameContext) -> Move:
    """Neural network chess bot with minimax search."""
    board = ctx.board

    # Use minimax search to find best move
    best_move, search_stats = search_engine.search(board, depth=SEARCH_DEPTH)

    # Handle case where no moves available
    if best_move is None:
        legal_moves = list(board.legal_moves)
        if legal_moves:
            best_move = legal_moves[0]
        else:
            # Game over
            print("No legal moves available - game over")
            return None

    # Get NN policy for visualization (still log probabilities)
    position = board_to_bitboard(board)
    position_tensor = torch.FloatTensor(position).unsqueeze(0).to(device)

    with torch.no_grad():
        policy_logits, value = model(position_tensor)

        # Mask illegal moves
        legal_mask = torch.FloatTensor(create_legal_move_mask(board)).to(device)
        policy_logits = policy_logits.masked_fill(legal_mask == 0, -1e9)

        # Softmax to get probabilities
        policy = F.softmax(policy_logits, dim=1).cpu().numpy()[0]

    # Calculate move probabilities for visualization
    legal_moves = list(board.legal_moves)
    move_probs = {}

    for move in legal_moves:
        idx = move_to_index(move)
        if 0 <= idx < 4096:
            move_probs[move] = float(policy[idx])

    # Normalize probabilities
    total = sum(move_probs.values())
    if total > 0:
        move_probs = {m: p/total for m, p in move_probs.items()}

    # Log probabilities for visualization
    ctx.logProbabilities(move_probs)

    # Print detailed information
    print(f"\n{'='*60}")
    print(f"🤖 Bot is making a move with SEARCH!")
    print(f"FEN: {board.fen()}")
    print(f"\n📊 Search Statistics:")
    print(f"  Depth: {search_stats['depth']}")
    print(f"  Nodes searched: {search_stats['nodes_searched']:,}")
    print(f"  NN calls: {search_stats.get('nn_calls', 0):,}")
    print(f"  Time: {search_stats['time_ms']:.1f}ms")
    print(f"  Speed: {search_stats.get('nps', 0):,} nodes/sec")
    print(f"  Best score: {search_stats['best_score']:.4f}")
    print(f"  TT hits/misses: {search_stats['tt_hits']}/{search_stats['tt_misses']}")
    print(f"  Cutoffs: {search_stats['cutoffs']}")
    print(f"\n🎯 Selected move: {best_move.uci()}")
    print(f"  NN policy probability: {move_probs.get(best_move, 0):.2%}")
    print(f"  NN position value: {value.item():.4f}")
    print(f"\n📋 Top 5 moves by NN policy:")
    for i, (move, prob) in enumerate(sorted(move_probs.items(), key=lambda x: x[1], reverse=True)[:5], 1):
        marker = "★" if move == best_move else " "
        print(f"  {marker} {i}. {move.uci()}: {prob:.2%}")
    print(f"{'='*60}\n")

    return best_move

@chess_manager.reset
def reset_state(ctx: GameContext):
    """Reset search state between games."""
    search_engine.clear_tt()
    print("Search state reset - transposition table cleared")