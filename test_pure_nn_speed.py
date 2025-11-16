#!/usr/bin/env python3
"""
Quick test to verify the pure NN policy is fast enough for bullet games.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import chess
import torch
import time
from src.model import ChessNet
from src.chess_utils import board_to_bitboard, create_legal_move_mask, move_to_index
from src.model_loader import load_model_checkpoint

def test_inference_speed():
    print("="*70)
    print("PURE NEURAL NETWORK POLICY - SPEED TEST")
    print("="*70)

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device.upper()}")

    model = ChessNet(num_residual_blocks=15, num_filters=256)
    checkpoint = load_model_checkpoint(device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded: {checkpoint.get('val_accuracy', 0):.2%} accuracy\n")

    # Test positions
    test_positions = [
        ("Starting position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Middlegame", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
        ("Complex position", "r1bq1rk1/pp2bppp/2n1pn2/2pp4/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 9"),
        ("Endgame", "8/5pk1/6p1/8/3K4/8/6P1/8 w - - 0 1"),
    ]

    total_time = 0
    num_tests = 0

    for name, fen in test_positions:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        
        print(f"\n{name}")
        print(f"FEN: {fen}")
        print(f"Legal moves: {len(legal_moves)}")
        
        # Warm-up run (first run may be slower due to GPU initialization)
        _ = get_best_move(model, board, device)
        
        # Timed runs
        times = []
        for _ in range(5):
            start = time.time()
            best_move = get_best_move(model, board, device)
            elapsed_ms = (time.time() - start) * 1000
            times.append(elapsed_ms)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"Best move: {best_move.uci()}")
        print(f"Inference time:")
        print(f"  Average: {avg_time:.1f}ms")
        print(f"  Min: {min_time:.1f}ms")
        print(f"  Max: {max_time:.1f}ms")
        
        total_time += avg_time
        num_tests += 1

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Average time per move: {total_time/num_tests:.1f}ms")
    print(f"Moves per second: {1000/(total_time/num_tests):.1f}")
    print(f"\n✓ For 1-minute games (60 seconds):")
    moves_possible = (60 * 1000) / (total_time/num_tests)
    print(f"  Theoretical max moves: {moves_possible:.0f}")
    print(f"  Typical game length: ~40 moves")
    print(f"  Time usage: {(40 * total_time/num_tests)/1000:.1f}s / 60s")
    
    if total_time/num_tests < 200:
        print(f"\n✅ EXCELLENT - Bot is fast enough for bullet chess!")
    elif total_time/num_tests < 500:
        print(f"\n✅ GOOD - Bot should work fine for bullet games")
    elif total_time/num_tests < 1000:
        print(f"\n⚠️  BORDERLINE - May have some time pressure in bullet")
    else:
        print(f"\n❌ TOO SLOW - Risk of timeout in bullet games")

def get_best_move(model, board, device):
    """Get best move using pure NN policy."""
    legal_moves = list(board.legal_moves)
    
    if len(legal_moves) == 1:
        return legal_moves[0]
    
    # Get NN evaluation
    position = board_to_bitboard(board)
    position_tensor = torch.FloatTensor(position).unsqueeze(0).to(device)
    
    with torch.no_grad():
        policy_logits, value = model(position_tensor)
        
        # Mask illegal moves
        legal_mask = torch.FloatTensor(create_legal_move_mask(board)).to(device)
        policy_logits = policy_logits.masked_fill(legal_mask == 0, -1e9)
        
        # Get probabilities
        policy = torch.nn.functional.softmax(policy_logits, dim=1).cpu().numpy()[0]
    
    # Find best legal move
    move_probs = {}
    for move in legal_moves:
        idx = move_to_index(move)
        if 0 <= idx < 4096:
            move_probs[move] = policy[idx]
    
    return max(move_probs, key=move_probs.get)

if __name__ == "__main__":
    test_inference_speed()
