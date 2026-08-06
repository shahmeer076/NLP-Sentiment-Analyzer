import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import torch

from src.config import TransformerConfig
from models.classifier import TransformerClassifier

config = TransformerConfig()

model = TransformerClassifier(config)

input_ids = torch.randint(
    0,
    config.vocab_size,
    (2, 10)
)

output = model(input_ids)

print("=" * 60)
print("Input Shape :", input_ids.shape)
print("Output Shape:", output.shape)
print()
print("Logits:")
print(output)
print("=" * 60)