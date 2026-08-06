import torch
import torch.nn as nn

from models.attention import MultiHeadSelfAttention


class FeedForward(nn.Module):
    def __init__(self, d_model=768, d_ff=3072):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.network(x)


class EncoderBlock(nn.Module):
    def __init__(self, d_model=768, num_heads=12, dropout=0.1):
        super().__init__()

        self.attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        attention_output = self.attention(x)

        x = self.norm1(
            x + self.dropout(attention_output)
        )

        ffn_output = self.ffn(x)

        x = self.norm2(
            x + self.dropout(ffn_output)
        )

        return x