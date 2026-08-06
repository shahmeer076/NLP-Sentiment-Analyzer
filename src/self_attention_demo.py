import torch
import torch.nn.functional as F

# Example embeddings (3 tokens, embedding size = 4)
embeddings = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],   # I
    [0.0, 2.0, 0.0, 2.0],   # love
    [1.0, 1.0, 1.0, 1.0]    # AI
])

print("=" * 50)
print("Embeddings")
print(embeddings)

# Step 1: Attention Scores
scores = embeddings @ embeddings.T

print("\nAttention Scores")
print(scores)

# Step 2: Softmax
weights = F.softmax(scores, dim=1)

print("\nAttention Weights")
print(weights)

# Step 3: Final Output
output = weights @ embeddings

print("\nAttention Output")
print(output)

print("=" * 50)