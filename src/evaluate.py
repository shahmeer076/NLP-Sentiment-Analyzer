from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm.auto import tqdm
import torch

MODEL_PATH = "saved_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# -------------------------
# Load Tokenizer & Model
# -------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# -------------------------
# Load Test Dataset
# -------------------------

dataset = load_dataset("imdb")

# Sirf 200 samples for fast evaluation
test_dataset = dataset["test"].select(range(200))


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )


test_dataset = test_dataset.map(tokenize, batched=True)

test_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "label",
    ],
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
)

# -------------------------
# Evaluation
# -------------------------

predictions = []
true_labels = []

with torch.no_grad():

    for batch in tqdm(test_loader):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        preds = torch.argmax(outputs.logits, dim=1)

        predictions.extend(preds.cpu().numpy())
        true_labels.extend(batch["label"].numpy())

accuracy = accuracy_score(true_labels, predictions)

precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels,
    predictions,
    average="binary",
    zero_division=0,
)

print("=" * 60)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print("=" * 60)