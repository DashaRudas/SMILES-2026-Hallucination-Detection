"""
Hallucination Detection in Small Language Models
"""
import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from aggregation import aggregation_and_feature_extraction
from evaluate import print_summary, run_evaluation, save_predictions, save_results
from model import MAX_LENGTH, get_model_and_tokenizer
from probe import HallucinationProbe
from splitting import split_data

# ---------------------------------------------------------------------
DATA_FILE     = "./data/dataset.csv"
OUTPUT_FILE   = "results.json"
BATCH_SIZE    = 4
USE_GEOMETRIC = True
TEST_FILE     = "./data/test.csv"
PREDICTIONS_FILE = "predictions.csv"

assert OUTPUT_FILE == "results.json"
assert PREDICTIONS_FILE == "predictions.csv"
# ---------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device       : {device}")
    print(f"Geometric feats: {USE_GEOMETRIC}")

    df = pd.read_csv(DATA_FILE)
    all_texts  = [f"{row['prompt']}{row['response']}" for _, row in df.iterrows()]
    all_labels = np.array([int(float(h)) for h in df["label"]])
    print(f"Loaded {len(all_labels)} samples  ({all_labels.sum()} hallucinated)")

    model, tokenizer = get_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    prompt_lengths = [
        len(tokenizer(p, truncation=True, max_length=MAX_LENGTH)["input_ids"])
        for p in df["prompt"].tolist()
    ]

    all_features = []
    t0 = time.time()

    for start in tqdm(range(0, len(all_texts), BATCH_SIZE), desc="Extracting", unit="batch"):
        batch_texts = all_texts[start:start+BATCH_SIZE]
        encoding = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
        input_ids, attention_mask = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = torch.stack(outputs.hidden_states, dim=1).float()
        mask = attention_mask.cpu()

        for i in range(hidden.size(0)):
            feat = aggregation_and_feature_extraction(
                hidden[i], mask[i], 
                use_geometric=USE_GEOMETRIC, 
                prompt_len=prompt_lengths[start + i],
                prompt_text=df.iloc[start + i]["prompt"],      
                response_text=df.iloc[start + i]["response"],
                tokenizer = tokenizer
            )
            all_features.append(feat.cpu())

        del hidden, outputs, encoding, input_ids, attention_mask
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    extract_time = time.time() - t0
    print(f"Extraction done in {extract_time:.1f} s")

    X = np.vstack([f.numpy() for f in all_features])
    y = all_labels
    print(f"Feature matrix: {X.shape}")

    # Запуск CV через evaluate.py
    splits = split_data(y, df)
    fold_results = run_evaluation(splits, X, y, HallucinationProbe)
    print_summary(fold_results, X.shape[1], len(X), extract_time)
    save_results(fold_results, X.shape[1], len(X), extract_time, OUTPUT_FILE)

    # ── Test set ───────────────────────────────────────────────────────
    df_test = pd.read_csv(TEST_FILE)
    test_texts = [f"{row['prompt']}{row['response']}" for _, row in df_test.iterrows()]
    test_prompt_lens = [len(tokenizer(p, truncation=True, max_length=MAX_LENGTH)["input_ids"]) for p in df_test["prompt"].tolist()]
    test_ids = df_test.index
    print(f"Test set: {len(test_texts)} samples")

    test_features = []
    for start in tqdm(range(0, len(test_texts), BATCH_SIZE), desc="Test extraction"):
        batch_texts = test_texts[start:start+BATCH_SIZE]
        encoding = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
        input_ids, attention_mask = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = torch.stack(outputs.hidden_states, dim=1).float()
        mask = attention_mask.cpu()

        for i in range(hidden.size(0)):
            feat = aggregation_and_feature_extraction(
                hidden[i], mask[i], use_geometric=USE_GEOMETRIC,
                prompt_len=test_prompt_lens[start + i],
                prompt_text=df_test.iloc[start + i]["prompt"],     
                response_text=df_test.iloc[start + i]["response"], 
                tokenizer=tokenizer
            )
            test_features.append(feat.cpu())
        del hidden, outputs, encoding, input_ids, attention_mask

    X_test = np.vstack([f.numpy() for f in test_features])

    # Финальное обучение на всех не-тестовых данных
    idx_non_test = np.unique(np.concatenate([
        np.concatenate([tr, va]) for tr, va, _ in splits
    ]))
    final_probe = HallucinationProbe()
    final_probe.fit(X[idx_non_test], y[idx_non_test])
    save_predictions(final_probe, X_test, test_ids, PREDICTIONS_FILE)
