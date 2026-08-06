import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.embeddings import EmbeddingLayer

# Example Input IDs
input_ids = torch.tensor([
    [101, 1045, 2293, 102]
])

embedding = EmbeddingLayer(
    vocab_size=30522,
    d_model=768
)

output = embedding(input_ids)

print("=" * 60)

print("Input Shape:")
print(input_ids.shape)

print("\nOutput Shape:")
print(output.shape)

print("=" * 60)