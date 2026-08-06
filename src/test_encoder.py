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

from models.encoder import EncoderBlock

x = torch.rand(
    1,
    10,
    768
)

encoder = EncoderBlock()

output = encoder(x)

print("=" * 60)
print("Input Shape:")
print(x.shape)

print()

print("Output Shape:")
print(output.shape)
print("=" * 60)