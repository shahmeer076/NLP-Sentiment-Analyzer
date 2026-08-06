from transformers import AutoTokenizer

# Load BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "I love Artificial Intelligence."

# Tokenize text
encoded = tokenizer(text)

print("=" * 60)
print("Original Text:")
print(text)

print("\nTokens:")
print(tokenizer.tokenize(text))

print("\nInput IDs:")
print(encoded["input_ids"])

print("\nAttention Mask:")
print(encoded["attention_mask"])

print("\nDecoded Text:")
print(tokenizer.decode(encoded["input_ids"]))

print("=" * 60)