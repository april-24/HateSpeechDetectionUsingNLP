"""Recompute leakage-safe per-label OOF thresholds for already-trained models."""
from src.config import MODEL_FILES
from src.train_utils import retune_saved_bundle

for name, path in MODEL_FILES.items():
    print("\n" + "="*72)
    print("Retuning thresholds:", name)
    retune_saved_bundle(name, path)
