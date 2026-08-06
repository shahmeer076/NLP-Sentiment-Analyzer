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

from models.attention import MultiHeadSelfAttention

x = torch.rand(
    1,
    10,
    768
)

model = MultiHeadSelfAttention()

output = model(x)

print("="*60)

print("Input Shape:")
print(x.shape)

print()

print("Output Shape:")
print(output.shape)

print("="*60)