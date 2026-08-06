import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    """
    Token Embedding + Positional Embedding
    """

    def __init__(
        self,
        vocab_size,
        d_model,
        max_length=512,
        dropout=0.1
    ):
        super().__init__()

        # Token Embedding
        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        # Positional Embedding
        self.position_embedding = nn.Embedding(
            max_length,
            d_model
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):

        batch_size, seq_length = input_ids.shape

        positions = torch.arange(
            seq_length,
            device=input_ids.device
        ).unsqueeze(0)

        token_embeddings = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(
            positions
        )

        embeddings = token_embeddings + position_embeddings

        return self.dropout(embeddings)