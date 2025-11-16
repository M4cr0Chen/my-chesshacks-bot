#!/usr/bin/env python3
"""
Quick performance test for the optimized search engine.

This script compares the old vs new search implementation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import chess
import torch
from src.model import ChessNet
from src.search import SearchEngine

# Test positions
TEST_POSITIONS = [
    # Opening position
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # Middlegame
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    # Tactical position
    "r1bqkb1r/pppp1ppp/2n5/4p3/2BnP3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5",
]

def main():
    print("="*70)
    print("CHESS SEARCH ENGINE - PERFORMANCE TEST")
    print("="*70)

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device.upper()}")

    model = ChessNet(num_residual_blocks=10, num_filters=128)
    model_path = os.path.join(os.path.dirname(__file__), 'src', 'weights', 'model.pt')

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded: {checkpoint.get('val_accuracy', 0):.2%} accuracy")

    # Initialize search engine
    search_engine = SearchEngine(model, device=device, max_time_ms=5000)

    print("\n" + "="*70)
    print("TESTING SEARCH AT DIFFERENT DEPTHS")
    print("="*70)

    for depth in [1, 2, 3]:
        print(f"\n{'─'*70}")
        print(f"DEPTH {depth}")
        print(f"{'─'*70}")

        for i, fen in enumerate(TEST_POSITIONS, 1):
            board = chess.Board(fen)
            print(f"\nPosition {i}: {fen[:50]}...")

            # Run search
            best_move, stats = search_engine.search(board, depth=depth)

            # Print results
            print(f"  Move: {best_move.uci()}")
            print(f"  Time: {stats['time_ms']:.1f}ms")
            print(f"  Nodes: {stats['nodes_searched']:,}")
            print(f"  NN calls: {stats['nn_calls']:,}")
            print(f"  Ratio: {stats['nn_calls'] / max(stats['nodes_searched'], 1):.2f}x")
            print(f"  Speed: {stats.get('nps', 0):,} nodes/sec")
            print(f"  Score: {stats['best_score']:.4f}")
            print(f"  TT hit rate: {stats['tt_hits'] / max(stats['tt_hits'] + stats['tt_misses'], 1):.1%}")
            print(f"  Cutoffs: {stats['cutoffs']}")

            # Check for efficiency
            ratio = stats['nn_calls'] / max(stats['nodes_searched'], 1)
            if ratio > 1.1:
                print(f"  ⚠️  WARNING: NN calls > nodes (ratio={ratio:.2f})!")
            elif ratio < 0.95:
                print(f"  ℹ️  INFO: NN calls < nodes (TT working well)")
            else:
                print(f"  ✓ Optimal: 1 NN call per node")

    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    print("\nExpected performance (with GPU):")
    print("  Depth 1: <100ms (30 nodes)")
    print("  Depth 2: <1s (900 nodes)")
    print("  Depth 3: <10s (27,000 nodes with pruning)")
    print("\nExpected NN call ratio:")
    print("  Optimal: 1.0x (1 NN call per node)")
    print("  Acceptable: 0.8-1.1x (TT may reduce calls)")
    print("  Bad: >1.1x (redundant NN calls - bug!)")
    print("\nIf you see ratio >1.1x, the old bug is still present!")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
