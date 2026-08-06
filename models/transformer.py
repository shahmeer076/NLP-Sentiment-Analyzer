import torch.nn as nn

from src.config import TransformerConfig
from models.embeddings import EmbeddingLayer
from models.encoder import EncoderBlock


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder built from multiple Encoder Blocks.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.embedding = EmbeddingLayer(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_length=config.max_length,
            dropout=config.dropout
        )

        self.layers = nn.ModuleList([
            EncoderBlock(
                d_model=config.d_model,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
            for _ in range(config.num_layers)
        ])

    def forward(self, input_ids):

        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x)

        return x