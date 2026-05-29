import json
import os
from datasets import load_dataset, DownloadMode

RAW_DIR = "../../data/raw"
PROCESSED_DIR = "../../data/processed"

def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

def save_raw_dataset(dataset):
    raw_file = os.path.join(RAW_DIR, "train_raw.jsonl")
    with open(raw_file, "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Raw dataset saved to: {raw_file}")

def build_processed_dataset(dataset):
    output_file = os.path.join(PROCESSED_DIR, "persona_train.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for row in dataset:
            persona = " ".join(row["personality"])
            for utt in row["utterances"]:
                history = " ".join(utt["history"])
                answer = utt["candidates"][0]
                sample = {
                    "prompt": f"Persona: {persona}\nHistory: {history}\nResponse:",
                    "completion": f" {answer}"
                }
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Processed dataset saved to: {output_file}")

def load_and_process():
    ensure_dirs()
    dataset = load_dataset(
        "AlekseyKorshuk/persona-chat",
        split="train",
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS
    )
    save_raw_dataset(dataset)
    build_processed_dataset(dataset)

if __name__ == "__main__":
    load_and_process()
