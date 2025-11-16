"""
Model loading utilities for chess bot.

Handles downloading model weights from Hugging Face Hub and loading them locally.
"""

import os
import torch
from huggingface_hub import hf_hub_download

# Configuration
REPO_ID = "macrochen1/ChessNet"  # Your HuggingFace username/repo
MODEL_FILENAME = "model.pt"
LOCAL_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
LOCAL_MODEL_PATH = os.path.join(LOCAL_WEIGHTS_DIR, MODEL_FILENAME)

# Use a cache directory outside of src/ to avoid triggering hot reloads
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".model_cache")


def ensure_model_downloaded(force_download=False):
    """
    Ensure model weights are available locally.

    Downloads from Hugging Face Hub if not present locally, or if force_download=True.

    Args:
        force_download: If True, re-download even if local file exists

    Returns:
        Path to local model file
    """
    # Check if model exists locally and force_download is False
    if os.path.exists(LOCAL_MODEL_PATH) and not force_download:
        print(f"✓ Using cached model from: {LOCAL_MODEL_PATH}")
        return LOCAL_MODEL_PATH

    # Ensure weights directory exists
    os.makedirs(LOCAL_WEIGHTS_DIR, exist_ok=True)

    print(f"📥 Downloading model from Hugging Face Hub...")
    print(f"   Repository: {REPO_ID}")
    print(f"   This may take a few minutes for first-time download...")

    try:
        # Download from Hugging Face Hub to cache outside src/
        # This prevents triggering hot reloads during download
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_FILENAME,
            cache_dir=CACHE_DIR,
        )

        print(f"✓ Model downloaded to cache!")
        print(f"   Copying to: {LOCAL_MODEL_PATH}")

        # Copy from cache to local weights directory
        import shutil
        shutil.copy2(downloaded_path, LOCAL_MODEL_PATH)

        print(f"✓ Model ready at: {LOCAL_MODEL_PATH}")

        return LOCAL_MODEL_PATH

    except Exception as e:
        print(f"❌ Error downloading model from Hugging Face: {e}")
        print(f"\n💡 Possible solutions:")
        print(f"   1. Check if the repository exists: https://huggingface.co/{REPO_ID}")
        print(f"   2. Ensure you've uploaded the model using upload_model_to_hf.py")
        print(f"   3. If the repository is private, set HF_TOKEN environment variable")
        print(f"   4. For local development, place model.pt in {LOCAL_WEIGHTS_DIR}/")

        # Check if we have a local fallback
        if os.path.exists(LOCAL_MODEL_PATH):
            print(f"\n⚠️  Using existing local model despite download failure")
            return LOCAL_MODEL_PATH

        raise RuntimeError(f"Model not available locally and download failed: {e}")


def load_model_checkpoint(model_path=None, device='cpu'):
    """
    Load model checkpoint from file.

    Args:
        model_path: Path to model file (if None, uses default location)
        device: Device to load model on

    Returns:
        Loaded checkpoint dictionary
    """
    if model_path is None:
        model_path = ensure_model_downloaded()

    try:
        checkpoint = torch.load(model_path, map_location=device)
        print(f"✓ Checkpoint loaded from: {model_path}")

        # Print checkpoint info if available
        if 'epoch' in checkpoint:
            print(f"   Epoch: {checkpoint['epoch']}")
        if 'val_accuracy' in checkpoint:
            print(f"   Validation Accuracy: {checkpoint['val_accuracy']:.2%}")

        return checkpoint

    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {model_path}: {e}")
