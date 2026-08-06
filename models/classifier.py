import torch
import torch.nn as nn

from src.config import TransformerConfig
from models.transformer import TransformerEncoder


class TransformerClassifier(nn.Module):
    """
    Transformer Encoder + Classification Head
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.transformer = TransformerEncoder(config)

        self.dropout = nn.Dropout(config.dropout)

        self.classifier = nn.Linear(
            config.d_model,
            config.num_classes
        )

    def forward(self, input_ids):

        encoder_output = self.transformer(input_ids)

        # CLS token representation
        cls_output = encoder_output[:, 0, :]

        cls_output = self.dropout(cls_output)

        logits = self.classifier(cls_output)

        return logits