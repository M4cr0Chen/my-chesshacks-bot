# Model Deployment Guide

This document explains how to deploy your trained chess model using Hugging Face Hub.

## Why Hugging Face?

Your trained model (`model.pt`) is **302MB**, which exceeds GitHub's recommended file size limit of 100MB. Instead of committing large model files to Git, we:

1. Upload model weights to Hugging Face Hub (free hosting)
2. Bot automatically downloads weights on startup
3. Keep the repository lightweight and fast to clone

## One-Time Setup: Upload Model to Hugging Face

### 1. Create Hugging Face Account

- Sign up at https://huggingface.co
- Create an access token at https://huggingface.co/settings/tokens
  - Click "New token"
  - Select "Write" access
  - Copy the token

### 2. Update Repository Name

Edit `upload_model_to_hf.py` line 23:

```python
REPO_ID = "your-username/chesshacks-chess-bot"  # Change to your HF username
```

Also update `src/model_loader.py` line 10:

```python
REPO_ID = "your-username/chesshacks-chess-bot"  # Change to match
```

### 3. Run Upload Script

```bash
# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Set your Hugging Face token (or enter when prompted)
export HF_TOKEN=your_token_here

# Run upload script
python upload_model_to_hf.py
```

The script will:
- Create a repository on Hugging Face (if it doesn't exist)
- Upload your model weights
- Create a README with model details
- Print the repository URL

### 4. Verify Upload

Visit https://huggingface.co/your-username/chesshacks-chess-bot to confirm your model is uploaded.

## How the Bot Uses the Model

When your bot starts up (`src/main.py`):

1. Checks if `src/weights/model.pt` exists locally
2. If not, downloads from Hugging Face Hub
3. Caches the download to avoid re-downloading
4. Loads the model and starts playing

The download happens automatically - no manual intervention needed!

## Local Development

For local testing, you can keep using the existing `src/weights/model.pt` file:

- The model loader first checks for a local file
- Only downloads from HuggingFace if local file is missing
- This means local development works even without uploading

## Deployment

When deploying to ChessHacks platform:

1. The bot will download weights from Hugging Face on first startup
2. Subsequent restarts use the cached version
3. No need to commit large files to Git

## File Organization

```
my-chesshacks-bot/
├── src/
│   ├── main.py              # Bot entrypoint (uses model_loader)
│   ├── model_loader.py      # Downloads from HuggingFace
│   └── weights/
│       ├── model.pt         # Local copy (gitignored)
│       └── .cache/          # HuggingFace download cache (gitignored)
├── upload_model_to_hf.py    # Upload script (run once)
├── requirements.txt         # Includes huggingface_hub
└── .gitignore               # Excludes model.pt
```

## Troubleshooting

### "Repository not found" error

- Make sure you updated `REPO_ID` in both files
- Verify the repository exists at https://huggingface.co/your-username/repo-name
- Re-run `upload_model_to_hf.py`

### "Authentication required" error

- Set `HF_TOKEN` environment variable
- Or make the repository public on Hugging Face

### Download is slow

- First download may take several minutes (302MB file)
- Subsequent startups use cached version (much faster)

### Local model not being used

- Ensure `src/weights/model.pt` exists
- Check file permissions
- The loader prioritizes local files over downloads

## Advanced: Private Models

If you want to keep your model private:

1. In `upload_model_to_hf.py`, change line 76:
   ```python
   private=True  # Make repository private
   ```

2. Set `HF_TOKEN` environment variable on deployment:
   ```bash
   export HF_TOKEN=your_token_here
   ```

The bot will use the token to download from private repositories.

## Summary

- ✅ Model weights hosted on Hugging Face (free, fast, version-controlled)
- ✅ Bot automatically downloads on startup
- ✅ Git repository stays lightweight (<5MB)
- ✅ Local development still works with local files
- ✅ Deployment is simple - just set environment variables

For questions, see: https://huggingface.co/docs/huggingface_hub
