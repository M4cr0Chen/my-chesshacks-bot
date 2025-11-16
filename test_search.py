#!/usr/bin/env python3
"""Test script to verify minimax search implementation."""

import torch
import chess
from src.model import ChessNet
from src.search import SearchEngine
import os

def test_search():
    """Test the search engine with various positions."""

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = ChessNet(num_residual_blocks=10, num_filters=128)
    model_path = os.path.join('src', 'weights', 'model.pt')

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded! Accuracy: {checkpoint.get('val_accuracy', 0):.2%}\n")

    # Initialize search engine
    search_engine = SearchEngine(model, device=device)

    # Test positions
    test_positions = [
        ("Starting position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Mid-game position", "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
        ("Endgame position", "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1"),
    ]

    for name, fen in test_positions:
        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print(f"FEN: {fen}")
        print(f"{'='*70}")

        board = chess.Board(fen)

        # Test different depths
        for depth in [2, 3]:
            print(f"\n--- Depth {depth} ---")
            best_move, stats = search_engine.search(board, depth=depth)

            print(f"Best move: {best_move.uci() if best_move else 'None'}")
            print(f"Score: {stats['best_score']:.4f}")
            print(f"Nodes searched: {stats['nodes_searched']:,}")
            print(f"Time: {stats['time_ms']:.1f}ms")
            print(f"Speed: {stats.get('nps', 0):,} nodes/sec")
            print(f"TT hits/misses: {stats['tt_hits']}/{stats['tt_misses']}")
            print(f"Cutoffs: {stats['cutoffs']}")

        # Clear TT between positions
        search_engine.clear_tt()

    print(f"\n{'='*70}")
    print("All tests completed successfully!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    test_search()
