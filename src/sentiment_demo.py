from transformers import pipeline

# Load pre-trained sentiment analysis pipeline
classifier = pipeline("sentiment-analysis")

# Example text
text = "I really enjoyed this movie. It was amazing!"

# Predict sentiment
result = classifier(text)

# Display result
print("=" * 50)
print("Input Text:")
print(text)
print("=" * 50)

print("Prediction:")
print(result)

print("=" * 50)
print("Label      :", result[0]["label"])
print("Confidence :", round(result[0]["score"] * 100, 2), "%")
print("=" * 50)