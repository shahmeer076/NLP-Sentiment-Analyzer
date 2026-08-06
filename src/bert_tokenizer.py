from transformers import AutoTokenizer
from datasets import load_dataset


MODEL_NAME = "bert-base-uncased"


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


dataset = load_dataset("imdb")


def tokenize_function(example):

    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True
)


print("=" * 60)

print(tokenized_dataset)

print("=" * 60)

print(tokenized_dataset["train"][0].keys())

print("=" * 60)

print(tokenized_dataset["train"][0]["input_ids"][:20])

print("=" * 60)