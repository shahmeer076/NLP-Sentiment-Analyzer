from transformers import AutoTokenizer, AutoModel
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

text = "I love Artificial Intelligence."

# Convert text into tensors
inputs = tokenizer(text, return_tensors="pt")

# Disable gradient calculation
with torch.no_grad():
    outputs = model(**inputs)

print("=" * 60)

print("Input IDs Shape:")
print(inputs["input_ids"].shape)

print("\nAttention Mask Shape:")
print(inputs["attention_mask"].shape)

print("\nLast Hidden State Shape:")
print(outputs.last_hidden_state.shape)

print("=" * 60)