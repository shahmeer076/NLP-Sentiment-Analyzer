import math
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model=768, num_heads=12):

        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):

        batch_size, seq_length, _ = x.shape

        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        Q = Q.view(
            batch_size,
            seq_length,
            self.num_heads,
            self.head_dim
        ).transpose(1,2)

        K = K.view(
            batch_size,
            seq_length,
            self.num_heads,
            self.head_dim
        ).transpose(1,2)

        V = V.view(
            batch_size,
            seq_length,
            self.num_heads,
            self.head_dim
        ).transpose(1,2)

        scores = torch.matmul(
            Q,
            K.transpose(-2,-1)
        )

        scores = scores / math.sqrt(self.head_dim)

        attention = torch.softmax(
            scores,
            dim=-1
        )

        output = torch.matmul(
            attention,
            V
        )

        output = output.transpose(
            1,
            2
        ).contiguous()

        output = output.view(
            batch_size,
            seq_length,
            self.d_model
        )

        return self.out(output)