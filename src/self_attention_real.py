import torch
import torch.nn.functional as F

torch.manual_seed(42)

# 3 tokens, embedding dimension = 4
x = torch.rand(3, 4)

print("=" * 60)
print("Input Embeddings")
print(x)

d_model = x.size(1)

# Weight matrices
W_Q = torch.rand(4, 4)
W_K = torch.rand(4, 4)
W_V = torch.rand(4, 4)

# Create Q, K, V
Q = x @ W_Q
K = x @ W_K
V = x @ W_V

print("\nQuery")
print(Q)

print("\nKey")
print(K)

print("\nValue")
print(V)

# Attention Scores
scores = Q @ K.T

print("\nScores")
print(scores)

# Scaling
scores = scores / torch.sqrt(torch.tensor(d_model, dtype=torch.float32))

# Softmax
weights = F.softmax(scores, dim=1)

print("\nAttention Weights")
print(weights)

# Final Output
output = weights @ V

print("\nFinal Output")
print(output)

print("=" * 60)