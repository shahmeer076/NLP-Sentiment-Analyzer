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
from models.transformer import TransformerEncoder

config = TransformerConfig()

input_ids = torch.randint(
    0,
    config.vocab_size,
    (2, 10)
)

model = TransformerEncoder(config)

output = model(input_ids)

print("=" * 60)
print("Input Shape:")
print(input_ids.shape)
print()
print("Output Shape:")
print(output.shape)
print("=" * 60)