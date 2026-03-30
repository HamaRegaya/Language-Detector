"""
Character-level CNN model for language detection + model loading utilities.

This module contains:
  - CharCNN: PyTorch character-level convolutional neural network
  - Model loading functions for both NB pipeline and CharCNN
  - Prediction utilities
"""
import re
import pickle
import string
from pathlib import Path

import numpy as np

# ── Character vocabulary ──
# All printable ASCII + common Unicode ranges for European/Asian scripts
# We use a fixed char-to-index mapping
CHARS = string.printable  # 100 printable ASCII chars
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 = padding/unknown
VOCAB_SIZE = len(CHARS) + 1  # +1 for padding
MAX_LEN = 512  # max characters per sample

LABELS = [
    'Arabic', 'Danish', 'Dutch', 'English', 'French', 'German',
    'Greek', 'Hindi', 'Italian', 'Kannada', 'Malayalam', 'Portugeese',
    'Russian', 'Spanish', 'Sweedish', 'Tamil', 'Turkish'
]
NUM_CLASSES = len(LABELS)


def text_to_indices(text: str, max_len: int = MAX_LEN) -> list[int]:
    """Convert text to a list of character indices."""
    indices = []
    for c in text[:max_len]:
        indices.append(CHAR_TO_IDX.get(c, 0))  # 0 for unknown chars
    # Pad to max_len
    indices += [0] * (max_len - len(indices))
    return indices


def clean_text(text: str) -> str:
    """Clean text for the NB pipeline (must match notebook's clean_text)."""
    text = re.sub(r'[!@#$(),\\n"%^*?\\:;~`0-9]', ' ', text)
    text = re.sub(r'[\[\]]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class CharCNN(nn.Module):
        """
        Character-level CNN for language detection.
        
        Architecture:
          Char Embedding(64d) → Conv1D(128, k=3) → Conv1D(128, k=5) 
          → Conv1D(64, k=3) → GlobalMaxPool → FC(256) → Dropout(0.3) → FC(17)
        
        This captures character n-gram patterns (digrams through 11-grams
        via stacked convolutions) which are highly discriminative for 
        language identification.
        """
        def __init__(
            self,
            vocab_size: int = VOCAB_SIZE,
            embed_dim: int = 64,
            num_classes: int = NUM_CLASSES,
            dropout: float = 0.3,
        ):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            
            # Multi-scale convolutions capture different n-gram lengths
            self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
            self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
            
            self.bn1 = nn.BatchNorm1d(128)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(64)
            
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(64, 256)
            self.fc2 = nn.Linear(256, num_classes)

        def forward(self, x):
            # x: (batch, seq_len) of char indices
            x = self.embedding(x)           # (batch, seq_len, embed_dim)
            x = x.permute(0, 2, 1)          # (batch, embed_dim, seq_len) for Conv1d
            
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            
            # Global max pooling over the sequence dimension
            x = x.max(dim=2)[0]             # (batch, 64)
            
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.fc2(x)                 # (batch, num_classes)
            return x

except ImportError:
    # PyTorch not available — NB-only mode
    CharCNN = None


class ModelManager:
    """Load and manage both NB and CNN models for inference."""

    def __init__(self, nb_path: str = None, cnn_path: str = None):
        self.nb_model = None
        self.cnn_model = None
        self.device = "cpu"

        if nb_path and Path(nb_path).exists():
            self._load_nb(nb_path)
        if cnn_path and Path(cnn_path).exists():
            self._load_cnn(cnn_path)

    def _load_nb(self, path: str):
        """Load the scikit-learn NB pipeline."""
        import __main__
        __main__.clean_text = clean_text
        with open(path, "rb") as f:
            self.nb_model = pickle.load(f)

    def _load_cnn(self, path: str):
        """Load the PyTorch CharCNN model."""
        if CharCNN is None:
            return
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.cnn_model = CharCNN(
            vocab_size=checkpoint.get("vocab_size", VOCAB_SIZE),
            embed_dim=checkpoint.get("embed_dim", 64),
            num_classes=checkpoint.get("num_classes", NUM_CLASSES),
            dropout=checkpoint.get("dropout", 0.3),
        )
        self.cnn_model.load_state_dict(checkpoint["model_state_dict"])
        self.cnn_model.to(self.device)
        self.cnn_model.eval()

    def predict_nb(self, text: str) -> dict:
        """Predict with NB model."""
        if self.nb_model is None:
            return {"error": "NB model not loaded"}
        pred = self.nb_model.predict([text])[0]
        proba = self.nb_model.predict_proba([text])[0]
        return {
            "language": LABELS[pred],
            "confidence": float(proba.max()),
            "probabilities": {LABELS[i]: float(p) for i, p in enumerate(proba)},
        }

    def predict_cnn(self, text: str) -> dict:
        """Predict with CNN model."""
        if self.cnn_model is None:
            return {"error": "CNN model not loaded"}
        import torch
        indices = text_to_indices(text)
        x = torch.tensor([indices], dtype=torch.long).to(self.device)
        with torch.no_grad():
            logits = self.cnn_model(x)
            proba = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred = int(proba.argmax())
        return {
            "language": LABELS[pred],
            "confidence": float(proba.max()),
            "probabilities": {LABELS[i]: float(p) for i, p in enumerate(proba)},
        }

    def predict(self, text: str, model: str = "cnn") -> dict:
        """Predict with specified model."""
        if model == "nb":
            return self.predict_nb(text)
        return self.predict_cnn(text)
