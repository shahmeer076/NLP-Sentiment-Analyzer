from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

MODEL_NAME = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

dataset = load_dataset("imdb")


def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
    )


tokenized_dataset = dataset.map(tokenize, batched=True)

tokenized_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"],
)

train_loader = DataLoader(
    tokenized_dataset["train"],
    batch_size=8,
    shuffle=True,
)

test_loader = DataLoader(
    tokenized_dataset["test"],
    batch_size=8,
)

batch = next(iter(train_loader))

print("=" * 60)
print("Input IDs Shape :", batch["input_ids"].shape)
print("Attention Shape :", batch["attention_mask"].shape)
print("Labels Shape    :", batch["label"].shape)
print("=" * 60)