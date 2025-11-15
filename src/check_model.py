"""Check if the model file is valid and trained."""

import torch
import os

model_path = os.path.join(os.path.dirname(__file__), 'weights', 'model.pt')

print(f"Checking model at: {model_path}")
print(f"File exists: {os.path.exists(model_path)}")
print(f"File size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB\n")

# Load checkpoint
checkpoint = torch.load(model_path, map_location='cpu')

print("Checkpoint keys:")
for key in checkpoint.keys():
    print(f"  - {key}")

print(f"\nCheckpoint info:")
if 'epoch' in checkpoint:
    print(f"  Epoch: {checkpoint['epoch']}")
if 'val_accuracy' in checkpoint:
    print(f"  Validation accuracy: {checkpoint['val_accuracy']:.2%}")
if 'val_loss' in checkpoint:
    print(f"  Validation loss: {checkpoint['val_loss']:.4f}")

# Check model state dict
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    print(f"\nModel has {len(state_dict)} parameter tensors")

    # Check a few key parameters
    print("\nSample parameters:")
    for key in list(state_dict.keys())[:5]:
        param = state_dict[key]
        print(f"  {key}: shape={param.shape}, mean={param.mean():.6f}, std={param.std():.6f}")

    # Check if model is trained (parameters should not all be near zero)
    all_params = torch.cat([p.flatten() for p in state_dict.values()])
    print(f"\nAll parameters:")
    print(f"  Mean: {all_params.mean():.6f}")
    print(f"  Std: {all_params.std():.6f}")
    print(f"  Min: {all_params.min():.6f}")
    print(f"  Max: {all_params.max():.6f}")

    if all_params.std() < 0.01:
        print("\n⚠️  WARNING: Model appears to be untrained (very low std)")
    else:
        print("\n✅ Model appears to have trained weights")
else:
    print("\n❌ No model_state_dict found in checkpoint!")
