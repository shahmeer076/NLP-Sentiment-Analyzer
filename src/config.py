from dataclasses import dataclass


@dataclass
class TransformerConfig:

    # Vocabulary
    vocab_size: int = 30522

    # Maximum sequence length
    max_length: int = 512

    # Transformer dimensions
    d_model: int = 768
    num_heads: int = 12
    num_layers: int = 6
    d_ff: int = 3072

    # Regularization
    dropout: float = 0.1

    # Classification
    num_classes: int = 2

    # Training
    batch_size: int = 16
    learning_rate: float = 2e-5
    epochs: int = 3