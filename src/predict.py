import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "saved_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.to(device)

model.eval()

while True:

    print("\n" + "=" * 60)

    text = input("Enter Review (type exit to quit): ")

    if text.lower() == "exit":
        break

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():

        outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)

        prediction = torch.argmax(probs, dim=1).item()

        confidence = probs[0][prediction].item() * 100

    label = "Positive 😊" if prediction == 1 else "Negative 😞"

    print("\nPrediction :", label)
    print(f"Confidence : {confidence:.2f}%")

print("\nProgram Closed.")