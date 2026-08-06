import os
import time

# Increase Hugging Face download timeout (in seconds)
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

from datasets import load_dataset


def load_imdb_dataset(max_retries=3):

    for attempt in range(max_retries):
        try:
            print(f"\nLoading IMDB Dataset... (Attempt {attempt + 1}/{max_retries})")

            dataset = load_dataset(
                "imdb",
                download_mode="reuse_dataset_if_exists"
            )

            print("=" * 60)
            print(dataset)
            print("=" * 60)

            print("Training Samples :", len(dataset["train"]))
            print("Testing Samples  :", len(dataset["test"]))

            print("=" * 60)
            print("First Review:\n")
            print(dataset["train"][0]["text"])

            print("\nLabel :", dataset["train"][0]["label"])
            print("=" * 60)

            return dataset

        except Exception as e:
            print(f"\nError: {e}")

            if attempt < max_retries - 1:
                print("Retrying in 5 seconds...\n")
                time.sleep(5)
            else:
                print("\nFailed to download IMDB dataset after multiple attempts.")
                raise


if __name__ == "__main__":
    load_imdb_dataset()