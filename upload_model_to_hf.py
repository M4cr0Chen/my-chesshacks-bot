#!/usr/bin/env python3
"""
Upload chess model weights to Hugging Face Hub.

This script uploads the trained model to Hugging Face Hub for deployment.
Run this script once after training to make weights available for the bot.

Usage:
    python upload_model_to_hf.py

You'll need to:
1. Create a Hugging Face account at https://huggingface.co
2. Create an access token at https://huggingface.co/settings/tokens (write access)
3. Set the HF_TOKEN environment variable or enter it when prompted
"""

from huggingface_hub import HfApi, create_repo, login
import os
import torch
import sys

# Configuration
REPO_ID = "macrochen1/ChessNet"  # Change to your username
MODEL_PATH = "src/weights/model.pt"
REPO_TYPE = "model"

def main():
    print("=" * 60)
    print("Chess Model Upload to Hugging Face Hub")
    print("=" * 60)

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print("   Please ensure you have trained weights in src/weights/model.pt")
        return

    # Get model info
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)

    print(f"\n📊 Model Information:")
    print(f"  Path: {MODEL_PATH}")
    print(f"  Size: {model_size_mb:.1f} MB")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'val_accuracy' in checkpoint:
        print(f"  Validation Accuracy: {checkpoint['val_accuracy']:.2%}")

    # Check for HF token and login if needed
    token = os.environ.get("HF_TOKEN")
    if not token:
        print(f"\n🔑 Hugging Face Authentication Required")
        print(f"   Get your token at: https://huggingface.co/settings/tokens")
        print(f"   Make sure to create a token with WRITE access")
        try:
            login()  # This will prompt for token
        except Exception as e:
            print(f"❌ Login failed: {e}")
            print(f"\n💡 Alternative: Set HF_TOKEN environment variable:")
            print(f"   export HF_TOKEN=your_token_here")
            print(f"   python upload_model_to_hf.py")
            return

    # Initialize Hugging Face API
    api = HfApi()

    print(f"\n🚀 Uploading to Hugging Face Hub...")
    print(f"  Repository: {REPO_ID}")

    try:
        # Create repository if it doesn't exist
        print(f"\n📦 Creating repository (if needed)...")
        create_repo(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            exist_ok=True,
            private=False  # Set to True if you want a private repo
        )
        print(f"  ✓ Repository ready")

        # Upload the model file
        print(f"\n📤 Uploading model file...")
        api.upload_file(
            path_or_fileobj=MODEL_PATH,
            path_in_repo="model.pt",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )
        print(f"  ✓ Model uploaded successfully!")

        # Create README
        readme_content = f"""---
license: mit
tags:
- chess
- reinforcement-learning
- alphazero
- chesshacks
---

# ChessHacks Chess Bot Model

This is a trained chess neural network model for the ChessHacks competition.

## Model Details

- **Architecture**: AlphaZero-style ResNet
- **Size**: {model_size_mb:.1f} MB
- **Residual Blocks**: 15
- **Filters**: 256
- **Parameters**: ~26M

## Training

- **Dataset**: Lichess master games (Hugging Face dataset)
- **Training Positions**: 500K+
- **Validation Accuracy**: {checkpoint.get('val_accuracy', 0):.2%}

## Usage

```python
import torch
from huggingface_hub import hf_hub_download

# Download model
model_path = hf_hub_download(
    repo_id="{REPO_ID}",
    filename="model.pt"
)

# Load checkpoint
checkpoint = torch.load(model_path, map_location='cpu')

# Use with your ChessNet model
# model.load_state_dict(checkpoint['model_state_dict'])
```

## Competition

Built for [ChessHacks](https://chesshacks.dev) - a competitive chess AI hackathon.
"""

        print(f"\n📝 Creating README...")
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )
        print(f"  ✓ README created")

        print(f"\n" + "=" * 60)
        print(f"✅ SUCCESS!")
        print(f"=" * 60)
        print(f"\n🔗 Your model is now available at:")
        print(f"   https://huggingface.co/{REPO_ID}")
        print(f"\n💡 Next steps:")
        print(f"   1. The bot will automatically download this model on startup")
        print(f"   2. Add src/weights/ to .gitignore to avoid committing large files")
        print(f"   3. Deploy your bot!")

    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        print(f"\n💡 Make sure you're logged in to Hugging Face:")
        print(f"   huggingface-cli login")
        print(f"\n   Get your token at: https://huggingface.co/settings/tokens")

if __name__ == "__main__":
    main()
