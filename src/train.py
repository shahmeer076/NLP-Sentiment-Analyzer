from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
import os

# ==================================================
# Configuration
# ==================================================

MODEL_NAME = "bert-base-uncased"

BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 2e-5
MAX_LENGTH = 128

TRAIN_SIZE = 500
TEST_SIZE = 100

SAVE_PATH = "saved_model"

# ==================================================
# Device
# ==================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("Using Device :", device)
print("=" * 60)

# ==================================================
# Tokenizer
# ==================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ==================================================
# Dataset
# ==================================================

print("\nLoading IMDB Dataset...\n")

dataset = load_dataset("imdb")

# Remove unused split (Saves Time)
if "unsupervised" in dataset:
    del dataset["unsupervised"]

print("Using Development Dataset")

dataset["train"] = (
    dataset["train"]
    .shuffle(seed=42)
    .select(range(TRAIN_SIZE))
)

dataset["test"] = (
    dataset["test"]
    .shuffle(seed=42)
    .select(range(TEST_SIZE))
)

print("Training Samples :", len(dataset["train"]))
print("Testing Samples  :", len(dataset["test"]))

# ==================================================
# Tokenization (Train Only)
# ==================================================

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

print("\nTokenizing Training Dataset...\n")

train_dataset = dataset["train"].map(
    tokenize,
    batched=True,
)

train_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "label",
    ],
)

# ==================================================
# DataLoader
# ==================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

# ==================================================
# Model
# ==================================================

print("\nLoading BERT Model...\n")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)

model.to(device)

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)

print("Model Loaded Successfully")
print("=" * 60)

# ==================================================
# Training
# ==================================================

model.train()

for epoch in range(EPOCHS):

    print(f"\nStarting Epoch {epoch + 1}/{EPOCHS}\n")

    total_loss = 0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}",
    )

    for batch in progress_bar:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    avg_loss = total_loss / len(train_loader)

    print(f"\nAverage Loss : {avg_loss:.4f}")

print("\nTraining Completed Successfully")

# ==================================================
# Save Model
# ==================================================

os.makedirs(
    SAVE_PATH,
    exist_ok=True,
)

model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print("\nModel Saved Successfully")
print("Location :", SAVE_PATH)

print("=" * 60)
print("Training Finished")
print("=" * 60)