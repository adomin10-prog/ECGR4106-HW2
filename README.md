# ECGR 4106 Homework 2: Character Language Models
Below is the exicuted notebook organized as so
1. Requirements
2. Helper functions
3. Plot functions
4. Problem 1 - Basic RNN character language model
5. Problem 1 plots and results
6. Problem 2 - LSTM character language model
7. Problem 2 plots and results
8. Problem 3 - GRU character language model
9. Problem 3 plots and results
10. Analysis and conclusion

link to exicuted notebook file via Colab https://colab.research.google.com/github/adomin10-prog/ECGR4106-HW2/blob/main/ECGR4106_HW2_CLM.ipynb

## Notebook Cells

## Cell 1 

```python
# ============================================================
# 1. Requirements
# ===========================================================

import os
import math
import time
import random
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from IPython.display import display, Markdown

# ----------------------------
# Reproducibility
# ----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ----------------------------
# Device setup (REMEBER TO USE G4)
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ----------------------------
# Main experiment settings
# ----------------------------

FIXED_TEXT_SEQUENCE_LENGTHS = [10, 20, 30]
SHAKESPEARE_SEQUENCE_LENGTHS = [20, 30, 50]

FIXED_TEXT_EPOCHS = 60
SHAKESPEARE_EPOCHS = 10

FIXED_TEXT_BATCH_SIZE = 32
SHAKESPEARE_BATCH_SIZE = 256

LEARNING_RATE = 0.002
GRAD_CLIP = 1.0


SHAKESPEARE_MAX_CHARS = None

# Text generation settings
GENERATE_CHARS_FIXED_TEXT = 250
GENERATE_CHARS_SHAKESPEARE = 500
TEMPERATURE = 0.8
```

### Output

**stdout:**

```text
Using device: cuda
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
```

---

## Cell 2 — Code

````python
# ============================================================
# 2. Helper functions
# ============================================================

NEXT_CHARACTER_TEXT = """
Next character prediction is a fundamental task in the field of natural language processing (NLP) that involves predicting the next character in a sequence of text based on the characters that precede it. This task is essential for various applications, including text auto-completion, spell checking, and even in the development of sophisticated AI models capable of generating human-like text.

At its core, next character prediction relies on statistical models or deep learning algorithms to analyze a given sequence of text and predict which character is most likely to follow. These predictions are based on patterns and relationships learned from large datasets of text during the training phase of the model.

One of the most popular approaches to next character prediction involves the use of Recurrent Neural Networks (RNNs), and more specifically, a variant called Long Short-Term Memory (LSTM) networks. RNNs are particularly well-suited for sequential data like text, as they can maintain information in 'memory' about previous characters to inform the prediction of the next character. LSTM networks enhance this capability by being able to remember long-term dependencies, making them even more effective for next character prediction tasks.

Training a model for next character prediction involves feeding it large amounts of text data, allowing it to learn the probability of each character's appearance following a sequence of characters. During this training process, the model adjusts its parameters to minimize the difference between its predictions and the actual outcomes, thus improving its predictive accuracy over time.

Once trained, the model can be used to predict the next character in a given piece of text by considering the sequence of characters that precede it. This can enhance user experience in text editing software, improve efficiency in coding environments with auto-completion features, and enable more natural interactions with AI-based chatbots and virtual assistants.

In summary, next character prediction plays a crucial role in enhancing the capabilities of various NLP applications, making text-based interactions more efficient, accurate, and human-like. Through the use of advanced machine learning models like RNNs and LSTMs, next character prediction continues to evolve, opening new possibilities for the future of text-based technology.
""".strip()


def print_section(title):
    display(Markdown(f"### {title}"))


def build_char_vocab(text):
    """
    Build character-to-index and index-to-character dictionaries.
    """
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return chars, stoi, itos


def encode_text(text, stoi):
    """
    Convert a string into a tensor of integer character IDs.
    """
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def decode_ids(ids, itos):
    """
    Convert a list/tensor of integer IDs back into text.
    """
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    return "".join(itos[int(i)] for i in ids)


class CharSequenceDataset(Dataset):
    """
    Character-level next-token dataset.

    Input x:  sequence of length seq_len
    Target y: same sequence shifted one character forward

    Example:
        text = "machine"
        seq_len = 3
        x = "mac"
        y = "ach"
    """
    def __init__(self, encoded_text, seq_len):
        self.encoded_text = encoded_text
        self.seq_len = seq_len

        if len(self.encoded_text) <= self.seq_len + 1:
            raise ValueError("Text is too short for this sequence length.")

    def __len__(self):
        return len(self.encoded_text) - self.seq_len

    def __getitem__(self, idx):
        x = self.encoded_text[idx : idx + self.seq_len]
        y = self.encoded_text[idx + 1 : idx + self.seq_len + 1]
        return x, y


def make_train_val_loaders(text, seq_len, batch_size, train_fraction=0.9):
    """
    Split text into train/validation parts and create DataLoaders.
    """
    chars, stoi, itos = build_char_vocab(text)
    encoded = encode_text(text, stoi)

    split_idx = int(len(encoded) * train_fraction)

    train_encoded = encoded[:split_idx]

    val_start = max(0, split_idx - seq_len)
    val_encoded = encoded[val_start:]

    train_dataset = CharSequenceDataset(train_encoded, seq_len)
    val_dataset = CharSequenceDataset(val_encoded, seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    vocab_info = {
        "chars": chars,
        "stoi": stoi,
        "itos": itos,
        "vocab_size": len(chars),
        "text_length": len(text),
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
    }

    return train_loader, val_loader, vocab_info


class CharLanguageModel(nn.Module):
    """
    General character language model that can use RNN, LSTM, or GRU.

    Architecture:
        character IDs
        -> embedding layer
        -> recurrent layer, selected by model_type
        -> optional fully connected hidden layer
        -> output layer over vocabulary
    """
    def __init__(
        self,
        vocab_size,
        model_type="RNN",
        embed_size=64,
        hidden_size=128,
        num_layers=1,
        dropout=0.0,
        fc_hidden_size=0
    ):
        super().__init__()

        self.model_type = model_type.upper()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fc_hidden_size = fc_hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)

        recurrent_dropout = dropout if num_layers > 1 else 0.0

        if self.model_type == "RNN":
            self.recurrent = nn.RNN(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=recurrent_dropout,
                nonlinearity="tanh",
            )
        elif self.model_type == "LSTM":
            self.recurrent = nn.LSTM(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=recurrent_dropout,
            )
        elif self.model_type == "GRU":
            self.recurrent = nn.GRU(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=recurrent_dropout,
            )
        else:
            raise ValueError("model_type must be one of: RNN, LSTM, GRU")

        if fc_hidden_size and fc_hidden_size > 0:
            self.output_head = nn.Sequential(
                nn.Linear(hidden_size, fc_hidden_size),
                nn.ReLU(),
                nn.Linear(fc_hidden_size, vocab_size),
            )
        else:
            self.output_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        recurrent_output, hidden = self.recurrent(embedded, hidden)
        logits = self.output_head(recurrent_output)
        return logits, hidden


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model):
    """
    Approximate model size assuming 32-bit floating point parameters.
    """
    params = count_trainable_parameters(model)
    return params * 4 / (1024 ** 2)


def recurrent_gate_multiplier(model_type):
    """
    Basic RNN has 1 recurrent computation.
    GRU has 3 gate-style computations.
    LSTM has 4 gate-style computations.
    """
    model_type = model_type.upper()

    if model_type == "RNN":
        return 1
    if model_type == "GRU":
        return 3
    if model_type == "LSTM":
        return 4

    raise ValueError("model_type must be one of: RNN, LSTM, GRU")


def approximate_sequence_operations(
    model_type,
    seq_len,
    vocab_size,
    embed_size,
    hidden_size,
    num_layers,
    fc_hidden_size=0
):
    """
    Rough operation-count estimate for comparing models.

    This is not an exact hardware FLOP profiler. It is a model-complexity estimate.
    It counts the dominant recurrent matrix operations and output projection.
    """
    gates = recurrent_gate_multiplier(model_type)

    ops_per_time_step = 0

    for layer in range(num_layers):
        input_dim = embed_size if layer == 0 else hidden_size

        ops_per_time_step += gates * ((input_dim * hidden_size) + (hidden_size * hidden_size))

    if fc_hidden_size and fc_hidden_size > 0:
        ops_per_time_step += hidden_size * fc_hidden_size
        ops_per_time_step += fc_hidden_size * vocab_size
    else:
        ops_per_time_step += hidden_size * vocab_size

    return seq_len * ops_per_time_step


def evaluate_model(model, data_loader, criterion, device):
    """
    Compute validation loss, validation accuracy, and perplexity.
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            logits, _ = model(x)
            vocab_size = logits.shape[-1]

            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

            token_count = y.numel()
            total_loss += loss.item() * token_count
            total_tokens += token_count

            predictions = logits.argmax(dim=-1)
            correct_tokens += (predictions == y).sum().item()

    avg_loss = total_loss / max(1, total_tokens)
    accuracy = correct_tokens / max(1, total_tokens)


    perplexity = math.exp(min(avg_loss, 20))

    return avg_loss, accuracy, perplexity


def train_language_model(
    model,
    train_loader,
    val_loader,
    epochs,
    learning_rate,
    device,
    grad_clip=1.0,
):
    """
    Train a character language model and return history plus elapsed training time.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = []
    start_time = time.perf_counter()

    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()

        total_train_loss = 0.0
        total_train_tokens = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits, _ = model(x)
            vocab_size = logits.shape[-1]

            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            token_count = y.numel()
            total_train_loss += loss.item() * token_count
            total_train_tokens += token_count

        train_loss = total_train_loss / max(1, total_train_tokens)
        val_loss, val_accuracy, val_perplexity = evaluate_model(
            model, val_loader, criterion, device
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_perplexity": val_perplexity,
        })

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"val acc: {val_accuracy:.4f} | "
            f"val ppl: {val_perplexity:.2f}"
        )

    elapsed_time = time.perf_counter() - start_time
    history_df = pd.DataFrame(history)

    return history_df, elapsed_time


@torch.no_grad()
def generate_text(
    model,
    start_text,
    stoi,
    itos,
    max_new_chars=300,
    temperature=0.8,
    device=DEVICE
):
    """
    Generate text one character at a time.
    """
    model.eval()
    model.to(device)

    generated = start_text

    for _ in range(max_new_chars):

        x = torch.tensor(
            [[stoi.get(ch, 0) for ch in generated]],
            dtype=torch.long,
            device=device
        )

        logits, _ = model(x)
        next_logits = logits[:, -1, :] / max(temperature, 1e-6)
        probabilities = torch.softmax(next_logits, dim=-1)

        next_id = torch.multinomial(probabilities, num_samples=1).item()
        next_char = itos[next_id]
        generated += next_char

    return generated


def run_single_experiment(
    text,
    dataset_name,
    model_type,
    seq_len,
    epochs,
    batch_size,
    embed_size,
    hidden_size,
    num_layers,
    fc_hidden_size=0,
    dropout=0.0,
    learning_rate=LEARNING_RATE,
    grad_clip=GRAD_CLIP,
    generate_chars=300,
    start_text=None,
):

    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_type} | seq_len={seq_len} | embed={embed_size} | hidden={hidden_size} | layers={num_layers} | fc_hidden={fc_hidden_size}")
    print("=" * 80)

    train_loader, val_loader, vocab_info = make_train_val_loaders(
        text=text,
        seq_len=seq_len,
        batch_size=batch_size,
        train_fraction=0.9,
    )

    model = CharLanguageModel(
        vocab_size=vocab_info["vocab_size"],
        model_type=model_type,
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        fc_hidden_size=fc_hidden_size,
    )

    param_count = count_trainable_parameters(model)
    size_mb = model_size_mb(model)

    approx_ops = approximate_sequence_operations(
        model_type=model_type,
        seq_len=seq_len,
        vocab_size=vocab_info["vocab_size"],
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        fc_hidden_size=fc_hidden_size,
    )

    history_df, training_time_sec = train_language_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=DEVICE,
        grad_clip=grad_clip,
    )

    if start_text is None:
        start_text = text[:min(40, len(text))]

    inference_start = time.perf_counter()
    generated_sample = generate_text(
        model=model,
        start_text=start_text,
        stoi=vocab_info["stoi"],
        itos=vocab_info["itos"],
        max_new_chars=generate_chars,
        temperature=TEMPERATURE,
        device=DEVICE,
    )
    inference_time_sec = time.perf_counter() - inference_start

    final_row = history_df.iloc[-1].to_dict()

    result = {
        "dataset": dataset_name,
        "model_type": model_type.upper(),
        "sequence_length": seq_len,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "embedding_size": embed_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "fc_hidden_size": fc_hidden_size,
        "vocab_size": vocab_info["vocab_size"],
        "text_length_chars": vocab_info["text_length"],
        "train_examples": vocab_info["train_examples"],
        "val_examples": vocab_info["val_examples"],
        "final_train_loss": final_row["train_loss"],
        "final_val_loss": final_row["val_loss"],
        "final_val_accuracy": final_row["val_accuracy"],
        "final_val_perplexity": final_row["val_perplexity"],
        "training_time_sec": training_time_sec,
        "inference_time_sec": inference_time_sec,
        "trainable_parameters": param_count,
        "model_size_mb": size_mb,
        "approx_ops_per_sequence": approx_ops,
        "complexity_note": f"O(T * L * gates * (D*H + H^2) + T*H*V), gates={recurrent_gate_multiplier(model_type)}",
    }

    config_name = (
        f"{dataset_name}_{model_type}_"
        f"seq{seq_len}_h{hidden_size}_layers{num_layers}_fc{fc_hidden_size}"
    )
    config_name = config_name.replace(" ", "_")

    history_df["config_name"] = config_name
    history_df["dataset"] = dataset_name
    history_df["model_type"] = model_type.upper()
    history_df["sequence_length"] = seq_len
    history_df["hidden_size"] = hidden_size
    history_df["num_layers"] = num_layers
    history_df["fc_hidden_size"] = fc_hidden_size

    sample = {
        "config_name": config_name,
        "dataset": dataset_name,
        "model_type": model_type.upper(),
        "sequence_length": seq_len,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "fc_hidden_size": fc_hidden_size,
        "generated_text": generated_sample,
    }

    print("\nGenerated sample:")
    print(generated_sample[:1200])
    print()

    return model, pd.DataFrame([result]), history_df, sample


def load_tiny_shakespeare(max_chars=None):
    """
    Load Tiny Shakespeare.
    """
    local_path = Path("tinyshakespeare.txt")

    if not local_path.exists():
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print("Downloading Tiny Shakespeare dataset...")
        urllib.request.urlretrieve(url, local_path)

    text = local_path.read_text(encoding="utf-8")

    if max_chars is not None:
        text = text[:max_chars]

    return text


def save_results_csv(df, filename):

    return None
    return path


def show_samples(samples):
    """
    Display generated text samples in a readable format for troubleshooting.
    """
    for sample in samples:
        display(Markdown(
            f"#### {sample['config_name']}\n\n"
            f"```text\n{sample['generated_text'][:1500]}\n```"
        ))
````

_No saved output for this cell._

---

## Cell 3 — Code `In [14]`

```python
# ============================================================
# 3. Plot functions
# ============================================================

def plot_training_curves(histories, title, metric="train_loss", save_name=None):

    plt.figure(figsize=(10, 6))

    for history_df in histories:
        if len(history_df) == 0:
            continue

        label = history_df["config_name"].iloc[0]
        plt.plot(history_df["epoch"], history_df[metric], marker="o", label=label)

    plt.xlabel("Epoch")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()


    plt.show()


def plot_result_bar(
    results_df,
    x_col,
    y_col,
    title,
    group_col=None,
    save_name=None,
    rotation=45
):
    df = results_df.copy()

    if group_col is not None:
        df["plot_label"] = (
            df[group_col].astype(str)
            + " | seq "
            + df[x_col].astype(str)
        )
    else:
        df["plot_label"] = df[x_col].astype(str)

    plt.figure(figsize=(10, 6))
    plt.bar(df["plot_label"], df[y_col])
    plt.xlabel(x_col.replace("_", " ").title())
    plt.ylabel(y_col.replace("_", " ").title())
    plt.title(title)
    plt.xticks(rotation=rotation, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()


    plt.show()


def plot_metric_vs_sequence(
    results_df,
    model_type,
    dataset_name,
    metric,
    title,
    save_name=None
):
    """
    Plot a metric against sequence length for a selected model and dataset.
    """
    df = results_df[
        (results_df["model_type"] == model_type.upper())
        & (results_df["dataset"] == dataset_name)
    ].copy()

    df = df.sort_values("sequence_length")

    plt.figure(figsize=(8, 5))
    plt.plot(df["sequence_length"], df[metric], marker="o")
    plt.xlabel("Sequence Length")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


    plt.show()


def display_clean_results(df):

    columns = [
        "dataset",
        "model_type",
        "sequence_length",
        "embedding_size",
        "hidden_size",
        "num_layers",
        "fc_hidden_size",
        "final_train_loss",
        "final_val_loss",
        "final_val_accuracy",
        "final_val_perplexity",
        "training_time_sec",
        "inference_time_sec",
        "trainable_parameters",
        "model_size_mb",
        "approx_ops_per_sequence",
    ]

    display_cols = [c for c in columns if c in df.columns]
    display(df[display_cols].sort_values(["dataset", "model_type", "sequence_length", "hidden_size", "num_layers", "fc_hidden_size"]))
```

_No saved output for this cell._

---

## Cell 4 — Code

```python
# ============================================================
# 4. Problem 1 - Basic RNN character language model
# ============================================================

p1_rnn_results = []
p1_rnn_histories = []
p1_rnn_samples = []

for seq_len in FIXED_TEXT_SEQUENCE_LENGTHS:
    model, result_df, history_df, sample = run_single_experiment(
        text=NEXT_CHARACTER_TEXT,
        dataset_name="Provided Text",
        model_type="RNN",
        seq_len=seq_len,
        epochs=FIXED_TEXT_EPOCHS,
        batch_size=FIXED_TEXT_BATCH_SIZE,
        embed_size=64,
        hidden_size=128,
        num_layers=1,
        fc_hidden_size=0,
        dropout=0.0,
        learning_rate=LEARNING_RATE,
        grad_clip=GRAD_CLIP,
        generate_chars=GENERATE_CHARS_FIXED_TEXT,
        start_text="Next character prediction",
    )

    p1_rnn_results.append(result_df)
    p1_rnn_histories.append(history_df)
    p1_rnn_samples.append(sample)

p1_rnn_results_df = pd.concat(p1_rnn_results, ignore_index=True)
p1_rnn_history_df = pd.concat(p1_rnn_histories, ignore_index=True)

save_results_csv(p1_rnn_results_df, "problem1_basic_rnn_results.csv")
save_results_csv(p1_rnn_history_df, "problem1_basic_rnn_history.csv")

display_clean_results(p1_rnn_results_df)
```

### Output

**stdout:**

```text
================================================================================
Dataset: Provided Text
Model: RNN | seq_len=10 | embed=64 | hidden=128 | layers=1 | fc_hidden=0
================================================================================
Epoch 001/60 | train loss: 2.4913 | val loss: 2.2724 | val acc: 0.3896 | val ppl: 9.70
Epoch 002/60 | train loss: 1.8087 | val loss: 2.0490 | val acc: 0.4483 | val ppl: 7.76
Epoch 003/60 | train loss: 1.5086 | val loss: 1.9361 | val acc: 0.4904 | val ppl: 6.93
Epoch 004/60 | train loss: 1.2880 | val loss: 1.9295 | val acc: 0.4967 | val ppl: 6.89
Epoch 005/60 | train loss: 1.1219 | val loss: 1.9793 | val acc: 0.5067 | val ppl: 7.24
Epoch 006/60 | train loss: 0.9940 | val loss: 2.0037 | val acc: 0.5067 | val ppl: 7.42
Epoch 007/60 | train loss: 0.8954 | val loss: 2.0563 | val acc: 0.5079 | val ppl: 7.82
Epoch 008/60 | train loss: 0.8231 | val loss: 2.1192 | val acc: 0.5112 | val ppl: 8.32
Epoch 009/60 | train loss: 0.7643 | val loss: 2.1433 | val acc: 0.5146 | val ppl: 8.53
Epoch 010/60 | train loss: 0.7268 | val loss: 2.1910 | val acc: 0.5200 | val ppl: 8.94
Epoch 011/60 | train loss: 0.6955 | val loss: 2.2617 | val acc: 0.4988 | val ppl: 9.60
Epoch 012/60 | train loss: 0.6698 | val loss: 2.2684 | val acc: 0.5067 | val ppl: 9.66
Epoch 013/60 | train loss: 0.6524 | val loss: 2.3020 | val acc: 0.5133 | val ppl: 9.99
Epoch 014/60 | train loss: 0.6362 | val loss: 2.3271 | val acc: 0.5042 | val ppl: 10.25
Epoch 015/60 | train loss: 0.6225 | val loss: 2.3787 | val acc: 0.5108 | val ppl: 10.79
Epoch 016/60 | train loss: 0.6140 | val loss: 2.3613 | val acc: 0.5021 | val ppl: 10.60
Epoch 017/60 | train loss: 0.6051 | val loss: 2.4432 | val acc: 0.4938 | val ppl: 11.51
Epoch 018/60 | train loss: 0.5987 | val loss: 2.4454 | val acc: 0.4908 | val ppl: 11.54
Epoch 019/60 | train loss: 0.5885 | val loss: 2.4518 | val acc: 0.5079 | val ppl: 11.61
Epoch 020/60 | train loss: 0.5842 | val loss: 2.4484 | val acc: 0.5092 | val ppl: 11.57
Epoch 021/60 | train loss: 0.5776 | val loss: 2.4813 | val acc: 0.5100 | val ppl: 11.96
Epoch 022/60 | train loss: 0.5737 | val loss: 2.4775 | val acc: 0.5054 | val ppl: 11.91
Epoch 023/60 | train loss: 0.5707 | val loss: 2.5439 | val acc: 0.5050 | val ppl: 12.73
Epoch 024/60 | train loss: 0.5664 | val loss: 2.5303 | val acc: 0.5067 | val ppl: 12.56
Epoch 025/60 | train loss: 0.5645 | val loss: 2.5605 | val acc: 0.4958 | val ppl: 12.94
Epoch 026/60 | train loss: 0.5595 | val loss: 2.5556 | val acc: 0.5104 | val ppl: 12.88
Epoch 027/60 | train loss: 0.5585 | val loss: 2.6226 | val acc: 0.4942 | val ppl: 13.77
Epoch 028/60 | train loss: 0.5556 | val loss: 2.5693 | val acc: 0.4946 | val ppl: 13.06
Epoch 029/60 | train loss: 0.5545 | val loss: 2.5929 | val acc: 0.5000 | val ppl: 13.37
Epoch 030/60 | train loss: 0.5484 | val loss: 2.5953 | val acc: 0.5021 | val ppl: 13.40
Epoch 031/60 | train loss: 0.5465 | val loss: 2.6024 | val acc: 0.4963 | val ppl: 13.50
Epoch 032/60 | train loss: 0.5431 | val loss: 2.6575 | val acc: 0.4983 | val ppl: 14.26
Epoch 033/60 | train loss: 0.5452 | val loss: 2.6392 | val acc: 0.4954 | val ppl: 14.00
Epoch 034/60 | train loss: 0.5427 | val loss: 2.6656 | val acc: 0.4971 | val ppl: 14.38
Epoch 035/60 | train loss: 0.5411 | val loss: 2.6617 | val acc: 0.4921 | val ppl: 14.32
Epoch 036/60 | train loss: 0.5391 | val loss: 2.6451 | val acc: 0.4971 | val ppl: 14.08
Epoch 037/60 | train loss: 0.5370 | val loss: 2.6680 | val acc: 0.4963 | val ppl: 14.41
Epoch 038/60 | train loss: 0.5361 | val loss: 2.6928 | val acc: 0.5046 | val ppl: 14.77
Epoch 039/60 | train loss: 0.5351 | val loss: 2.6914 | val acc: 0.4942 | val ppl: 14.75
Epoch 040/60 | train loss: 0.5329 | val loss: 2.6690 | val acc: 0.5017 | val ppl: 14.43
Epoch 041/60 | train loss: 0.5309 | val loss: 2.6688 | val acc: 0.5104 | val ppl: 14.42
Epoch 042/60 | train loss: 0.5336 | val loss: 2.7116 | val acc: 0.4938 | val ppl: 15.05
Epoch 043/60 | train loss: 0.5291 | val loss: 2.7007 | val acc: 0.4942 | val ppl: 14.89
Epoch 044/60 | train loss: 0.5315 | val loss: 2.7209 | val acc: 0.4954 | val ppl: 15.19
Epoch 045/60 | train loss: 0.5287 | val loss: 2.7368 | val acc: 0.5008 | val ppl: 15.44
Epoch 046/60 | train loss: 0.5264 | val loss: 2.7497 | val acc: 0.5004 | val ppl: 15.64
Epoch 047/60 | train loss: 0.5265 | val loss: 2.7417 | val acc: 0.4975 | val ppl: 15.51
Epoch 048/60 | train loss: 0.5250 | val loss: 2.7058 | val acc: 0.5025 | val ppl: 14.97
Epoch 049/60 | train loss: 0.5227 | val loss: 2.7322 | val acc: 0.4942 | val ppl: 15.37
Epoch 050/60 | train loss: 0.5229 | val loss: 2.7131 | val acc: 0.5050 | val ppl: 15.08
Epoch 051/60 | train loss: 0.5242 | val loss: 2.7474 | val acc: 0.5121 | val ppl: 15.60
Epoch 052/60 | train loss: 0.5211 | val loss: 2.7764 | val acc: 0.5017 | val ppl: 16.06
Epoch 053/60 | train loss: 0.5206 | val loss: 2.7959 | val acc: 0.5017 | val ppl: 16.38
Epoch 054/60 | train loss: 0.5224 | val loss: 2.7651 | val acc: 0.5012 | val ppl: 15.88
Epoch 055/60 | train loss: 0.5209 | val loss: 2.7749 | val acc: 0.5079 | val ppl: 16.04
Epoch 056/60 | train loss: 0.5215 | val loss: 2.7714 | val acc: 0.5012 | val ppl: 15.98
Epoch 057/60 | train loss: 0.5195 | val loss: 2.7856 | val acc: 0.5079 | val ppl: 16.21
Epoch 058/60 | train loss: 0.5178 | val loss: 2.7984 | val acc: 0.4996 | val ppl: 16.42
Epoch 059/60 | train loss: 0.5161 | val loss: 2.7959 | val acc: 0.5033 | val ppl: 16.38
Epoch 060/60 | train loss: 0.5197 | val loss: 2.7554 | val acc: 0.5004 | val ppl: 15.73

Generated sample:
Next character prediction of the next character in a given sequence of text and predictions and the next character prediction involves feeding it large amounts of text auto-completion features, and even in the field of natural language process, the models or deep learning al

================================================================================
Dataset: Provided Text
Model: RNN | seq_len=20 | embed=64 | hidden=128 | layers=1 | fc_hidden=0
================================================================================
Epoch 001/60 | train loss: 2.4125 | val loss: 2.1248 | val acc: 0.4138 | val ppl: 8.37
Epoch 002/60 | train loss: 1.5680 | val loss: 1.8727 | val acc: 0.4971 | val ppl: 6.51
Epoch 003/60 | train loss: 1.1547 | val loss: 1.8633 | val acc: 0.5448 | val ppl: 6.45
Epoch 004/60 | train loss: 0.8615 | val loss: 1.9223 | val acc: 0.5504 | val ppl: 6.84
Epoch 005/60 | train loss: 0.6672 | val loss: 2.0018 | val acc: 0.5340 | val ppl: 7.40
Epoch 006/60 | train loss: 0.5496 | val loss: 2.0690 | val acc: 0.5487 | val ppl: 7.92
Epoch 007/60 | train loss: 0.4754 | val loss: 2.1701 | val acc: 0.5404 | val ppl: 8.76
Epoch 008/60 | train loss: 0.4304 | val loss: 2.2184 | val acc: 0.5417 | val ppl: 9.19
Epoch 009/60 | train loss: 0.4035 | val loss: 2.2460 | val acc: 0.5567 | val ppl: 9.45
Epoch 010/60 | train loss: 0.3814 | val loss: 2.2956 | val acc: 0.5531 | val ppl: 9.93
Epoch 011/60 | train loss: 0.3675 | val loss: 2.3311 | val acc: 0.5521 | val ppl: 10.29
Epoch 012/60 | train loss: 0.3570 | val loss: 2.4244 | val acc: 0.5440 | val ppl: 11.30
Epoch 013/60 | train loss: 0.3459 | val loss: 2.4371 | val acc: 0.5542 | val ppl: 11.44
Epoch 014/60 | train loss: 0.3392 | val loss: 2.4478 | val acc: 0.5556 | val ppl: 11.56
Epoch 015/60 | train loss: 0.3320 | val loss: 2.4870 | val acc: 0.5515 | val ppl: 12.03
Epoch 016/60 | train loss: 0.3261 | val loss: 2.5247 | val acc: 0.5421 | val ppl: 12.49
Epoch 017/60 | train loss: 0.3210 | val loss: 2.4700 | val acc: 0.5525 | val ppl: 11.82
Epoch 018/60 | train loss: 0.3167 | val loss: 2.5604 | val acc: 0.5460 | val ppl: 12.94
Epoch 019/60 | train loss: 0.3137 | val loss: 2.5143 | val acc: 0.5523 | val ppl: 12.36
Epoch 020/60 | train loss: 0.3118 | val loss: 2.5399 | val acc: 0.5467 | val ppl: 12.68
Epoch 021/60 | train loss: 0.3106 | val loss: 2.5875 | val acc: 0.5535 | val ppl: 13.30
Epoch 022/60 | train loss: 0.3066 | val loss: 2.5912 | val acc: 0.5581 | val ppl: 13.35
Epoch 023/60 | train loss: 0.3035 | val loss: 2.5987 | val acc: 0.5540 | val ppl: 13.45
Epoch 024/60 | train loss: 0.3013 | val loss: 2.6421 | val acc: 0.5571 | val ppl: 14.04
Epoch 025/60 | train loss: 0.3025 | val loss: 2.6201 | val acc: 0.5621 | val ppl: 13.74
Epoch 026/60 | train loss: 0.2982 | val loss: 2.6861 | val acc: 0.5587 | val ppl: 14.67
Epoch 027/60 | train loss: 0.2969 | val loss: 2.6354 | val acc: 0.5569 | val ppl: 13.95
Epoch 028/60 | train loss: 0.2957 | val loss: 2.6569 | val acc: 0.5600 | val ppl: 14.25
Epoch 029/60 | train loss: 0.2945 | val loss: 2.7229 | val acc: 0.5629 | val ppl: 15.22
Epoch 030/60 | train loss: 0.2931 | val loss: 2.6917 | val acc: 0.5583 | val ppl: 14.76
Epoch 031/60 | train loss: 0.2916 | val loss: 2.6868 | val acc: 0.5621 | val ppl: 14.68
Epoch 032/60 | train loss: 0.2914 | val loss: 2.7200 | val acc: 0.5558 | val ppl: 15.18
Epoch 033/60 | train loss: 0.2910 | val loss: 2.6722 | val acc: 0.5608 | val ppl: 14.47
Epoch 034/60 | train loss: 0.2882 | val loss: 2.7777 | val acc: 0.5571 | val ppl: 16.08
Epoch 035/60 | train loss: 0.2900 | val loss: 2.7402 | val acc: 0.5537 | val ppl: 15.49
Epoch 036/60 | train loss: 0.2867 | val loss: 2.7870 | val acc: 0.5652 | val ppl: 16.23
Epoch 037/60 | train loss: 0.2855 | val loss: 2.7720 | val acc: 0.5571 | val ppl: 15.99
Epoch 038/60 | train loss: 0.2863 | val loss: 2.8065 | val acc: 0.5560 | val ppl: 16.55
Epoch 039/60 | train loss: 0.2848 | val loss: 2.8158 | val acc: 0.5604 | val ppl: 16.71
Epoch 040/60 | train loss: 0.2845 | val loss: 2.7644 | val acc: 0.5567 | val ppl: 15.87
Epoch 041/60 | train loss: 0.2819 | val loss: 2.8145 | val acc: 0.5606 | val ppl: 16.69
Epoch 042/60 | train loss: 0.2826 | val loss: 2.7736 | val acc: 0.5650 | val ppl: 16.02
Epoch 043/60 | train loss: 0.2815 | val loss: 2.8465 | val acc: 0.5517 | val ppl: 17.23
Epoch 044/60 | train loss: 0.2810 | val loss: 2.8642 | val acc: 0.5644 | val ppl: 17.53
Epoch 045/60 | train loss: 0.2797 | val loss: 2.8129 | val acc: 0.5558 | val ppl: 16.66
Epoch 046/60 | train loss: 0.2797 | val loss: 2.8035 | val acc: 0.5658 | val ppl: 16.50
Epoch 047/60 | train loss: 0.2807 | val loss: 2.8712 | val acc: 0.5446 | val ppl: 17.66
Epoch 048/60 | train loss: 0.2802 | val loss: 2.8029 | val acc: 0.5577 | val ppl: 16.49
Epoch 049/60 | train loss: 0.2798 | val loss: 2.8522 | val acc: 0.5519 | val ppl: 17.33
Epoch 050/60 | train loss: 0.2782 | val loss: 2.8331 | val acc: 0.5681 | val ppl: 17.00
Epoch 051/60 | train loss: 0.2763 | val loss: 2.8720 | val acc: 0.5602 | val ppl: 17.67
Epoch 052/60 | train loss: 0.2767 | val loss: 2.8403 | val acc: 0.5583 | val ppl: 17.12
Epoch 053/60 | train loss: 0.2766 | val loss: 2.8529 | val acc: 0.5587 | val ppl: 17.34
Epoch 054/60 | train loss: 0.2770 | val loss: 2.8660 | val acc: 0.5548 | val ppl: 17.57
Epoch 055/60 | train loss: 0.2774 | val loss: 2.8303 | val acc: 0.5675 | val ppl: 16.95
Epoch 056/60 | train loss: 0.2750 | val loss: 2.8531 | val acc: 0.5675 | val ppl: 17.34
Epoch 057/60 | train loss: 0.2750 | val loss: 2.8604 | val acc: 0.5542 | val ppl: 17.47
Epoch 058/60 | train loss: 0.2748 | val loss: 2.8827 | val acc: 0.5523 | val ppl: 17.86
Epoch 059/60 | train loss: 0.2729 | val loss: 2.8779 | val acc: 0.5625 | val ppl: 17.78
Epoch 060/60 | train loss: 0.2746 | val loss: 2.9108 | val acc: 0.5563 | val ppl: 18.37

Generated sample:
Next character prediction involves the use of Recurrent Neural Networks (RNNs), and more specifically, a various applications, making text-based chatbots and virtual assistants.

In summary, next character prediction is a fundamental task in the field of natural language pro

================================================================================
Dataset: Provided Text
Model: RNN | seq_len=30 | embed=64 | hidden=128 | layers=1 | fc_hidden=0
================================================================================
Epoch 001/60 | train loss: 2.3572 | val loss: 2.0766 | val acc: 0.4200 | val ppl: 7.98
Epoch 002/60 | train loss: 1.4848 | val loss: 1.9039 | val acc: 0.5164 | val ppl: 6.71
Epoch 003/60 | train loss: 1.0070 | val loss: 1.9308 | val acc: 0.5617 | val ppl: 6.90
Epoch 004/60 | train loss: 0.6858 | val loss: 2.0145 | val acc: 0.5481 | val ppl: 7.50
Epoch 005/60 | train loss: 0.4965 | val loss: 2.1734 | val acc: 0.5308 | val ppl: 8.79
Epoch 006/60 | train loss: 0.3942 | val loss: 2.2500 | val acc: 0.5381 | val ppl: 9.49
Epoch 007/60 | train loss: 0.3396 | val loss: 2.3595 | val acc: 0.5303 | val ppl: 10.59
Epoch 008/60 | train loss: 0.3062 | val loss: 2.4156 | val acc: 0.5546 | val ppl: 11.20
Epoch 009/60 | train loss: 0.2835 | val loss: 2.4443 | val acc: 0.5463 | val ppl: 11.52
Epoch 010/60 | train loss: 0.2692 | val loss: 2.4980 | val acc: 0.5526 | val ppl: 12.16
Epoch 011/60 | train loss: 0.2587 | val loss: 2.5734 | val acc: 0.5492 | val ppl: 13.11
Epoch 012/60 | train loss: 0.2496 | val loss: 2.6253 | val acc: 0.5503 | val ppl: 13.81
Epoch 013/60 | train loss: 0.2413 | val loss: 2.6260 | val acc: 0.5479 | val ppl: 13.82
Epoch 014/60 | train loss: 0.2368 | val loss: 2.6636 | val acc: 0.5551 | val ppl: 14.35
Epoch 015/60 | train loss: 0.2317 | val loss: 2.7203 | val acc: 0.5550 | val ppl: 15.18
Epoch 016/60 | train loss: 0.2273 | val loss: 2.7397 | val acc: 0.5542 | val ppl: 15.48
Epoch 017/60 | train loss: 0.2244 | val loss: 2.7473 | val acc: 0.5549 | val ppl: 15.60
Epoch 018/60 | train loss: 0.2216 | val loss: 2.7841 | val acc: 0.5463 | val ppl: 16.19
Epoch 019/60 | train loss: 0.2179 | val loss: 2.7925 | val acc: 0.5522 | val ppl: 16.32
Epoch 020/60 | train loss: 0.2145 | val loss: 2.8113 | val acc: 0.5613 | val ppl: 16.63
Epoch 021/60 | train loss: 0.2149 | val loss: 2.8328 | val acc: 0.5478 | val ppl: 16.99
Epoch 022/60 | train loss: 0.2117 | val loss: 2.8709 | val acc: 0.5639 | val ppl: 17.65
Epoch 023/60 | train loss: 0.2105 | val loss: 2.9094 | val acc: 0.5490 | val ppl: 18.34
Epoch 024/60 | train loss: 0.2080 | val loss: 2.8609 | val acc: 0.5515 | val ppl: 17.48
Epoch 025/60 | train loss: 0.2070 | val loss: 2.9174 | val acc: 0.5494 | val ppl: 18.49
Epoch 026/60 | train loss: 0.2057 | val loss: 2.9165 | val acc: 0.5568 | val ppl: 18.48
Epoch 027/60 | train loss: 0.2055 | val loss: 2.9153 | val acc: 0.5468 | val ppl: 18.45
Epoch 028/60 | train loss: 0.2033 | val loss: 2.9401 | val acc: 0.5492 | val ppl: 18.92
Epoch 029/60 | train loss: 0.2024 | val loss: 2.9620 | val acc: 0.5572 | val ppl: 19.34
Epoch 030/60 | train loss: 0.2008 | val loss: 2.9811 | val acc: 0.5490 | val ppl: 19.71
Epoch 031/60 | train loss: 0.1993 | val loss: 2.9811 | val acc: 0.5460 | val ppl: 19.71
Epoch 032/60 | train loss: 0.1991 | val loss: 3.0100 | val acc: 0.5572 | val ppl: 20.29
Epoch 033/60 | train loss: 0.1989 | val loss: 2.9565 | val acc: 0.5433 | val ppl: 19.23
Epoch 034/60 | train loss: 0.1962 | val loss: 2.9924 | val acc: 0.5515 | val ppl: 19.93
Epoch 035/60 | train loss: 0.1963 | val loss: 3.0045 | val acc: 0.5478 | val ppl: 20.18
Epoch 036/60 | train loss: 0.1975 | val loss: 3.0499 | val acc: 0.5522 | val ppl: 21.11
Epoch 037/60 | train loss: 0.1962 | val loss: 3.0128 | val acc: 0.5487 | val ppl: 20.35
Epoch 038/60 | train loss: 0.1941 | val loss: 3.0338 | val acc: 0.5415 | val ppl: 20.78
Epoch 039/60 | train loss: 0.1945 | val loss: 3.0371 | val acc: 0.5485 | val ppl: 20.84
Epoch 040/60 | train loss: 0.1941 | val loss: 3.0290 | val acc: 0.5467 | val ppl: 20.68
Epoch 041/60 | train loss: 0.1943 | val loss: 3.0535 | val acc: 0.5494 | val ppl: 21.19
Epoch 042/60 | train loss: 0.1935 | val loss: 3.0697 | val acc: 0.5463 | val ppl: 21.54
Epoch 043/60 | train loss: 0.1931 | val loss: 3.1021 | val acc: 0.5489 | val ppl: 22.24
Epoch 044/60 | train loss: 0.1929 | val loss: 3.0975 | val acc: 0.5431 | val ppl: 22.14
Epoch 045/60 | train loss: 0.1931 | val loss: 3.1382 | val acc: 0.5492 | val ppl: 23.06
Epoch 046/60 | train loss: 0.1926 | val loss: 3.0758 | val acc: 0.5492 | val ppl: 21.67
Epoch 047/60 | train loss: 0.1911 | val loss: 3.0874 | val acc: 0.5413 | val ppl: 21.92
Epoch 048/60 | train loss: 0.1903 | val loss: 3.1599 | val acc: 0.5382 | val ppl: 23.57
Epoch 049/60 | train loss: 0.1895 | val loss: 3.2103 | val acc: 0.5419 | val ppl: 24.79
Epoch 050/60 | train loss: 0.1898 | val loss: 3.1525 | val acc: 0.5474 | val ppl: 23.39
Epoch 051/60 | train loss: 0.1894 | val loss: 3.1513 | val acc: 0.5496 | val ppl: 23.37
Epoch 052/60 | train loss: 0.1892 | val loss: 3.1938 | val acc: 0.5468 | val ppl: 24.38
Epoch 053/60 | train loss: 0.1890 | val loss: 3.1653 | val acc: 0.5492 | val ppl: 23.70
Epoch 054/60 | train loss: 0.1886 | val loss: 3.2222 | val acc: 0.5383 | val ppl: 25.08
Epoch 055/60 | train loss: 0.1887 | val loss: 3.1675 | val acc: 0.5517 | val ppl: 23.75
Epoch 056/60 | train loss: 0.1874 | val loss: 3.1898 | val acc: 0.5463 | val ppl: 24.28
Epoch 057/60 | train loss: 0.1889 | val loss: 3.1935 | val acc: 0.5504 | val ppl: 24.37
Epoch 058/60 | train loss: 0.1879 | val loss: 3.1902 | val acc: 0.5518 | val ppl: 24.29
Epoch 059/60 | train loss: 0.1875 | val loss: 3.2327 | val acc: 0.5481 | val ppl: 25.35
Epoch 060/60 | train loss: 0.1869 | val loss: 3.2467 | val acc: 0.5433 | val ppl: 25.70

Generated sample:
Next character prediction involves the use of Recurrent Neural Networks (RNNs), and more specifically, a variant called Long Short-Term Memory (LSTM) networks. RNNs are particularly well-suited for sequential data like text, as they can maintain information in 'memory' about
```

```text
         dataset model_type  sequence_length  embedding_size  hidden_size  \
0  Provided Text        RNN               10              64          128   
1  Provided Text        RNN               20              64          128   
2  Provided Text        RNN               30              64          128   

   num_layers  fc_hidden_size  final_train_loss  final_val_loss  \
0           1               0          0.519695        2.755376   
1           1               0          0.274615        2.910769   
2           1               0          0.186928        3.246676   

   final_val_accuracy  final_val_perplexity  training_time_sec  \
0            0.500417             15.726946           4.644834   
1            0.556250             18.370914           3.702814   
2            0.543333             25.704748           3.696915   

   inference_time_sec  trainable_parameters  model_size_mb  \
0            0.174232                 33517       0.127857   
1            0.054844                 33517       0.127857   
2            0.054806                 33517       0.127857   

   approx_ops_per_sequence  
0                   303360  
1                   606720  
2                   910080  
```

---

## Cell 5 — Code `In [16]`

```python
# ============================================================
# 5. Problem 1 plots and results
# ============================================================

print_section("Problem 1 - Basic RNN result table")
display_clean_results(p1_rnn_results_df)

print_section("Problem 1 - Basic RNN training loss curves")
plot_training_curves(
    p1_rnn_histories,
    title="Basic RNN training Loss - Provided Text",
    metric="train_loss",
    save_name="basic_rnn_training_loss.png",
)

print_section("Problem 1 - Basic RNN validation loss curves")
plot_training_curves(
    p1_rnn_histories,
    title="Basic RNN Validation Loss - Provided Text",
    metric="val_loss",
    save_name="basic_rnn_validation_loss.png",
)

print_section("Problem 1 - Basic RNN validation accuracy curves")
plot_training_curves(
    p1_rnn_histories,
    title="Basic RNN Validation accuracy - provided Text",
    metric="val_accuracy",
    save_name="basic_rnn_validation_accuracy.png",
)

print_section("Problem 1 - Basic RNN model size by sequence length")
plot_result_bar(
    p1_rnn_results_df,
    x_col="sequence_length",
    y_col="model_size_mb",
    group_col="model_type",
    title="Basic RNN Model Size - Provided Text",
    save_name="basic_rnn_model_size.png",
)

print_section("Problem 1 - Basic RNN generated text samples")
show_samples(p1_rnn_samples)
```

### Output

### Problem 1 - Basic RNN result table

```text
         dataset model_type  sequence_length  embedding_size  hidden_size  \
0  Provided Text        RNN               10              64          128   
1  Provided Text        RNN               20              64          128   
2  Provided Text        RNN               30              64          128   

   num_layers  fc_hidden_size  final_train_loss  final_val_loss  \
0           1               0          0.519695        2.755376   
1           1               0          0.274615        2.910769   
2           1               0          0.186928        3.246676   

   final_val_accuracy  final_val_perplexity  training_time_sec  \
0            0.500417             15.726946           4.644834   
1            0.556250             18.370914           3.702814   
2            0.543333             25.704748           3.696915   

   inference_time_sec  trainable_parameters  model_size_mb  \
0            0.174232                 33517       0.127857   
1            0.054844                 33517       0.127857   
2            0.054806                 33517       0.127857   

   approx_ops_per_sequence  
0                   303360  
1                   606720  
2                   910080  
```

### Problem 1 - Basic RNN training loss curves

![Cell 5 output 4](README_assets/cell_05_output_04_001.png)

### Problem 1 - Basic RNN validation loss curves

![Cell 5 output 6](README_assets/cell_05_output_06_002.png)

### Problem 1 - Basic RNN validation accuracy curves

![Cell 5 output 8](README_assets/cell_05_output_08_003.png)

### Problem 1 - Basic RNN model size by sequence length

![Cell 5 output 10](README_assets/cell_05_output_10_004.png)

### Problem 1 - Basic RNN generated text samples

#### Provided_Text_RNN_seq10_h128_layers1_fc0

```text
Next character prediction of the next character in a given sequence of text and predictions and the next character prediction involves feeding it large amounts of text auto-completion features, and even in the field of natural language process, the models or deep learning al
```

#### Provided_Text_RNN_seq20_h128_layers1_fc0

```text
Next character prediction involves the use of Recurrent Neural Networks (RNNs), and more specifically, a various applications, making text-based chatbots and virtual assistants.

In summary, next character prediction is a fundamental task in the field of natural language pro
```

#### Provided_Text_RNN_seq30_h128_layers1_fc0

```text
Next character prediction involves the use of Recurrent Neural Networks (RNNs), and more specifically, a variant called Long Short-Term Memory (LSTM) networks. RNNs are particularly well-suited for sequential data like text, as they can maintain information in 'memory' about
```

---

## Cell 6 — Code

```python
# ============================================================
# 6A. Notebook Problem 2 - LSTM on provided paragraph (Official HW Problem 1)
# ============================================================

p2_lstm_fixed_results = []
p2_lstm_fixed_histories = []
p2_lstm_fixed_samples = []

for seq_len in FIXED_TEXT_SEQUENCE_LENGTHS:
    model, result_df, history_df, sample = run_single_experiment(
        text=NEXT_CHARACTER_TEXT,
        dataset_name="Provided Text",
        model_type="LSTM",
        seq_len=seq_len,
        epochs=FIXED_TEXT_EPOCHS,
        batch_size=FIXED_TEXT_BATCH_SIZE,
        embed_size=64,
        hidden_size=128,
        num_layers=1,
        fc_hidden_size=0,
        dropout=0.0,
        learning_rate=LEARNING_RATE,
        grad_clip=GRAD_CLIP,
        generate_chars=GENERATE_CHARS_FIXED_TEXT,
        start_text="Next character prediction",
    )

    p2_lstm_fixed_results.append(result_df)
    p2_lstm_fixed_histories.append(history_df)
    p2_lstm_fixed_samples.append(sample)

p2_lstm_fixed_results_df = pd.concat(p2_lstm_fixed_results, ignore_index=True)
p2_lstm_fixed_history_df = pd.concat(p2_lstm_fixed_histories, ignore_index=True)

save_results_csv(p2_lstm_fixed_results_df, "problem2_lstm_provided_text_results.csv")
save_results_csv(p2_lstm_fixed_history_df, "problem2_lstm_provided_text_history.csv")

display_clean_results(p2_lstm_fixed_results_df)
```

### Output

**stdout:**

```text
================================================================================
Dataset: Provided Text
Model: LSTM | seq_len=10 | embed=64 | hidden=128 | layers=1 | fc_hidden=0
================================================================================
Epoch 001/60 | train loss: 2.7273 | val loss: 2.3732 | val acc: 0.3588 | val ppl: 10.73
Epoch 002/60 | train loss: 1.9675 | val loss: 2.1062 | val acc: 0.4288 | val ppl: 8.22
Epoch 003/60 | train loss: 1.5900 | val loss: 1.9892 | val acc: 0.4754 | val ppl: 7.31
Epoch 004/60 | train loss: 1.3235 | val loss: 1.9152 | val acc: 0.5025 | val ppl: 6.79
Epoch 005/60 | train loss: 1.1192 | val loss: 1.9170 | val acc: 0.5138 | val ppl: 6.80
Epoch 006/60 | train loss: 0.9728 | val loss: 1.9474 | val acc: 0.5225 | val ppl: 7.01
Epoch 007/60 | train loss: 0.8635 | val loss: 1.9807 | val acc: 0.5292 | val ppl: 7.25
Epoch 008/60 | train loss: 0.7853 | val loss: 2.0048 | val acc: 0.5267 | val ppl: 7.42
Epoch 009/60 | train loss: 0.7284 | val loss: 2.0819 | val acc: 0.5192 | val ppl: 8.02
Epoch 010/60 | train loss: 0.6860 | val loss: 2.1098 | val acc: 0.5262 | val ppl: 8.25
Epoch 011/60 | train loss: 0.6501 | val loss: 2.1977 | val acc: 0.5079 | val ppl: 9.00
Epoch 012/60 | train loss: 0.6279 | val loss: 2.2181 | val acc: 0.5100 | val ppl: 9.19
Epoch 013/60 | train loss: 0.6090 | val loss: 2.2647 | val acc: 0.5188 | val ppl: 9.63
Epoch 014/60 | train loss: 0.5927 | val loss: 2.3364 | val acc: 0.5117 | val ppl: 10.34
Epoch 015/60 | train loss: 0.5797 | val loss: 2.3040 | val acc: 0.5204 | val ppl: 10.01
Epoch 016/60 | train loss: 0.5732 | val loss: 2.3515 | val acc: 0.5108 | val ppl: 10.50
Epoch 017/60 | train loss: 0.5623 | val loss: 2.3520 | val acc: 0.5054 | val ppl: 10.51
Epoch 018/60 | train loss: 0.5561 | val loss: 2.4091 | val acc: 0.5121 | val ppl: 11.12
Epoch 019/60 | train loss: 0.5491 | val loss: 2.4068 | val acc: 0.5125 | val ppl: 11.10
Epoch 020/60 | train loss: 0.5423 | val loss: 2.4553 | val acc: 0.5092 | val ppl: 11.65
Epoch 021/60 | train loss: 0.5405 | val loss: 2.4708 | val acc: 0.5133 | val ppl: 11.83
Epoch 022/60 | train loss: 0.5366 | val loss: 2.5107 | val acc: 0.4988 | val ppl: 12.31
Epoch 023/60 | train loss: 0.5323 | val loss: 2.4922 | val acc: 0.5208 | val ppl: 12.09
Epoch 024/60 | train loss: 0.5290 | val loss: 2.5241 | val acc: 0.5029 | val ppl: 12.48
Epoch 025/60 | train loss: 0.5270 | val loss: 2.5348 | val acc: 0.5058 | val ppl: 12.61
Epoch 026/60 | train loss: 0.5229 | val loss: 2.5403 | val acc: 0.5017 | val ppl: 12.68
Epoch 027/60 | train loss: 0.5210 | val loss: 2.5404 | val acc: 0.5058 | val ppl: 12.68
Epoch 028/60 | train loss: 0.5188 | val loss: 2.5855 | val acc: 0.5058 | val ppl: 13.27
Epoch 029/60 | train loss: 0.5179 | val loss: 2.5854 | val acc: 0.5054 | val ppl: 13.27
Epoch 030/60 | train loss: 0.5143 | val loss: 2.6032 | val acc: 0.5083 | val ppl: 13.51
Epoch 031/60 | train loss: 0.5145 | val loss: 2.6069 | val acc: 0.5088 | val ppl: 13.56
Epoch 032/60 | train loss: 0.5119 | val loss: 2.5959 | val acc: 0.5138 | val ppl: 13.41
Epoch 033/60 | train loss: 0.5109 | val loss: 2.6093 | val acc: 0.5083 | val ppl: 13.59
Epoch 034/60 | train loss: 0.5121 | val loss: 2.6247 | val acc: 0.5183 | val ppl: 13.80
Epoch 035/60 | train loss: 0.5096 | val loss: 2.6166 | val acc: 0.5096 | val ppl: 13.69
Epoch 036/60 | train loss: 0.5065 | val loss: 2.6414 | val acc: 0.5062 | val ppl: 14.03
Epoch 037/60 | train loss: 0.5081 | val loss: 2.6746 | val acc: 0.5146 | val ppl: 14.51
Epoch 038/60 | train loss: 0.5066 | val loss: 2.6687 | val acc: 0.5062 | val ppl: 14.42
Epoch 039/60 | train loss: 0.5054 | val loss: 2.6410 | val acc: 0.5058 | val ppl: 14.03
Epoch 040/60 | train loss: 0.5050 | val loss: 2.6915 | val acc: 0.5096 | val ppl: 14.75
Epoch 041/60 | train loss: 0.5047 | val loss: 2.6863 | val acc: 0.5112 | val ppl: 14.68
Epoch 042/60 | train loss: 0.5022 | val loss: 2.7002 | val acc: 0.5067 | val ppl: 14.88
Epoch 043/60 | train loss: 0.5015 | val loss: 2.7282 | val acc: 0.5046 | val ppl: 15.30
Epoch 044/60 | train loss: 0.5019 | val loss: 2.6815 | val acc: 0.5083 | val ppl: 14.61
Epoch 045/60 | train loss: 0.4996 | val loss: 2.7343 | val acc: 0.5108 | val ppl: 15.40
Epoch 046/60 | train loss: 0.4992 | val loss: 2.6962 | val acc: 0.5192 | val ppl: 14.82
Epoch 047/60 | train loss: 0.4981 | val loss: 2.7362 | val acc: 0.5100 | val ppl: 15.43
Epoch 048/60 | train loss: 0.4979 | val loss: 2.7180 | val acc: 0.5125 | val ppl: 15.15
Epoch 049/60 | train loss: 0.4974 | val loss: 2.7372 | val acc: 0.5067 | val ppl: 15.44
Epoch 050/60 | train loss: 0.4973 | val loss: 2.7632 | val acc: 0.5088 | val ppl: 15.85
Epoch 051/60 | train loss: 0.4974 | val loss: 2.7458 | val acc: 0.5083 | val ppl: 15.58
Epoch 052/60 | train loss: 0.4960 | val loss: 2.7582 | val acc: 0.5121 | val ppl: 15.77
Epoch 053/60 | train loss: 0.4956 | val loss: 2.7694 | val acc: 0.5167 | val ppl: 15.95
Epoch 054/60 | train loss: 0.4965 | val loss: 2.7870 | val acc: 0.5154 | val ppl: 16.23
Epoch 055/60 | train loss: 0.4940 | val loss: 2.7766 | val acc: 0.5112 | val ppl: 16.06
Epoch 056/60 | train loss: 0.4945 | val loss: 2.7806 | val acc: 0.5142 | val ppl: 16.13
Epoch 057/60 | train loss: 0.4951 | val loss: 2.7732 | val acc: 0.5146 | val ppl: 16.01
Epoch 058/60 | train loss: 0.4930 | val loss: 2.7649 | val acc: 0.5217 | val ppl: 15.88
Epoch 059/60 | train loss: 0.4932 | val loss: 2.7910 | val acc: 0.5150 | val ppl: 16.30
Epoch 060/60 | train loss: 0.4925 | val loss: 2.8066 | val acc: 0.5125 | val ppl: 16.55

Generated sample:
Next character prediction involves prediction tasks.

Training a sequence of character prediction involves feeding it large amounts of text data, allowing a sequence of character prediction tasks.

Training a sequence of character is most likely to follow. These prediction i

================================================================================
Dataset: Provided Text
Model: LSTM | seq_len=20 | embed=64 | hidden=128 | layers=1 | fc_hidden=0
================================================================================
Epoch 001/60 | train loss: 2.6876 | val loss: 2.3192 | val acc: 0.3546 | val ppl: 10.17
Epoch 002/60 | train loss: 1.8163 | val loss: 2.0220 | val acc: 0.4481 | val ppl: 7.55
Epoch 003/60 | train loss: 1.3452 | val loss: 1.9198 | val acc: 0.5044 | val ppl: 6.82
Epoch 004/60 | train loss: 1.0117 | val loss: 1.9555 | val acc: 0.5092 | val ppl: 7.07
Epoch 005/60 | train loss: 0.7754 | val loss: 2.0225 | val acc: 0.5225 | val ppl: 7.56
Epoch 006/60 | train loss: 0.6219 | val loss: 2.1272 | val acc: 0.5127 | val ppl: 8.39
Epoch 007/60 | train loss: 0.5153 | val loss: 2.1917 | val acc: 0.5146 | val ppl: 8.95
Epoch 008/60 | train loss: 0.4520 | val loss: 2.2842 | val acc: 0.5133 | val ppl: 9.82
Epoch 009/60 | train loss: 0.4062 | val loss: 2.3325 | val acc: 0.5160 | val ppl: 10.30
Epoch 010/60 | train loss: 0.3770 | val loss: 2.3902 | val acc: 0.5079 | val ppl: 10.92
Epoch 011/60 | train loss: 0.3571 | val loss: 2.4397 | val acc: 0.5142 | val ppl: 11.47
Epoch 012/60 | train loss: 0.3426 | val loss: 2.4769 | val acc: 0.5115 | val ppl: 11.90
Epoch 013/60 | train loss: 0.3311 | val loss: 2.5316 | val acc: 0.5121 | val ppl: 12.57
Epoch 014/60 | train loss: 0.3221 | val loss: 2.5470 | val acc: 0.5012 | val ppl: 12.77
Epoch 015/60 | train loss: 0.3131 | val loss: 2.5858 | val acc: 0.5023 | val ppl: 13.27
Epoch 016/60 | train loss: 0.3063 | val loss: 2.5838 | val acc: 0.5106 | val ppl: 13.25
Epoch 017/60 | train loss: 0.3015 | val loss: 2.6444 | val acc: 0.5058 | val ppl: 14.08
Epoch 018/60 | train loss: 0.2973 | val loss: 2.6594 | val acc: 0.5135 | val ppl: 14.29
Epoch 019/60 | train loss: 0.2936 | val loss: 2.6538 | val acc: 0.4992 | val ppl: 14.21
Epoch 020/60 | train loss: 0.2907 | val loss: 2.7210 | val acc: 0.5115 | val ppl: 15.20
Epoch 021/60 | train loss: 0.2877 | val loss: 2.7399 | val acc: 0.5144 | val ppl: 15.49
Epoch 022/60 | train loss: 0.2843 | val loss: 2.7526 | val acc: 0.5056 | val ppl: 15.68
Epoch 023/60 | train loss: 0.2824 | val loss: 2.7639 | val acc: 0.5042 | val ppl: 15.86
Epoch 024/60 | train loss: 0.2809 | val loss: 2.7971 | val acc: 0.5069 | val ppl: 16.40
Epoch 025/60 | train loss: 0.2793 | val loss: 2.7708 | val acc: 0.5077 | val ppl: 15.97
Epoch 026/60 | train loss: 0.2774 | val loss: 2.7848 | val acc: 0.5088 | val ppl: 16.20
Epoch 027/60 | train loss: 0.2767 | val loss: 2.8555 | val acc: 0.4992 | val ppl: 17.38
Epoch 028/60 | train loss: 0.2740 | val loss: 2.8234 | val acc: 0.5083 | val ppl: 16.83
Epoch 029/60 | train loss: 0.2739 | val loss: 2.8187 | val acc: 0.5140 | val ppl: 16.76
Epoch 030/60 | train loss: 0.2732 | val loss: 2.8642 | val acc: 0.5104 | val ppl: 17.53
Epoch 031/60 | train loss: 0.2719 | val loss: 2.8890 | val acc: 0.5162 | val ppl: 17.98
Epoch 032/60 | train loss: 0.2707 | val loss: 2.8989 | val acc: 0.5058 | val ppl: 18.15
Epoch 033/60 | train loss: 0.2696 | val loss: 2.9092 | val acc: 0.5083 | val ppl: 18.34
Epoch 034/60 | train loss: 0.2690 | val loss: 2.9296 | val acc: 0.5065 | val ppl: 18.72
Epoch 035/60 | train loss: 0.2684 | val loss: 2.9215 | val acc: 0.5102 | val ppl: 18.57
Epoch 036/60 | train loss: 0.2675 | val loss: 2.9058 | val acc: 0.5208 | val ppl: 18.28
Epoch 037/60 | train loss: 0.2672 | val loss: 2.9569 | val acc: 0.5156 | val ppl: 19.24
Epoch 038/60 | train loss: 0.2651 | val loss: 2.9339 | val acc: 0.5152 | val ppl: 18.80
Epoch 039/60 | train loss: 0.2660 | val loss: 2.9597 | val acc: 0.5123 | val ppl: 19.29
Epoch 040/60 | train loss: 0.2649 | val loss: 3.0240 | val acc: 0.5008 | val ppl: 20.57
Epoch 041/60 | train loss: 0.2642 | val loss: 2.9689 | val acc: 0.5052 | val ppl: 19.47
Epoch 042/60 | train loss: 0.2638 | val loss: 3.0052 | val acc: 0.5146 | val ppl: 20.19
Epoch 043/60 | train loss: 0.2626 | val loss: 3.0308 | val acc: 0.5081 | val ppl: 20.71
Epoch 044/60 | train loss: 0.2631 | val loss: 3.0093 | val acc: 0.5079 | val ppl: 20.27
Epoch 045/60 | train loss: 0.2616 | val loss: 3.0587 | val acc: 0.5125 | val ppl: 21.30
Epoch 046/60 | train loss: 0.2622 | val loss: 3.0246 | val acc: 0.5056 | val ppl: 20.59
Epoch 047/60 | train loss: 0.2603 | val loss: 3.0508 | val acc: 0.5094 | val ppl: 21.13
Epoch 048/60 | train loss: 0.2605 | val loss: 3.0301 | val acc: 0.5125 | val ppl: 20.70
Epoch 049/60 | train loss: 0.2608 | val loss: 3.0219 | val acc: 0.5112 | val ppl: 20.53
Epoch 050/60 | train loss: 0.2608 | val loss: 3.0337 | val acc: 0.5035 | val ppl: 20.77
Epoch 051/60 | train loss: 0.2606 | val loss: 3.0621 | val acc: 0.5133 | val ppl: 21.37
Epoch 052/60 | train loss: 0.2590 | val loss: 3.0196 | val acc: 0.5135 | val ppl: 20.48
Epoch 053/60 | train loss: 0.2589 | val loss: 3.0842 | val acc: 0.5138 | val ppl: 21.85
Epoch 054/60 | train loss: 0.2593 | val loss: 3.0922 | val acc: 0.5021 | val ppl: 22.02
Epoch 055/60 | train loss: 0.2583 | val loss: 3.1089 | val acc: 0.5190 | val ppl: 22.40
Epoch 056/60 | train loss: 0.2588 | val loss: 3.0822 | val acc: 0.5215 | val ppl: 21.81
Epoch 057/60 | train loss: 0.2581 | val loss: 3.0877 | val acc: 0.5202 | val ppl: 21.93
Epoch 058/60 | train loss: 0.2578 | val loss: 3.0866 | val acc: 0.5117 | val ppl: 21.90
Epoch 059/60 | train loss: 0.2582 | val loss: 3.1556 | val acc: 0.5058 | val ppl: 23.47
Epoch 060/60 | train loss: 0.2577 | val loss: 3.0750 | val acc: 0.5173 | val ppl: 21.65

Generated sample:
Next character prediction plays a crucial role in enhancing the capabilities of various NLP applications, making text-based ith character prediction is a fundamental task in the field of natural language processing (NLP) that involves predicting the next character prediction

================================================================================
Dataset: Provided Text
Model: LSTM | seq_len=30 | embed=64 | hidden=128 | layers=1 | fc_hidden=0
================================================================================
Epoch 001/60 | train loss: 2.6416 | val loss: 2.2541 | val acc: 0.3931 | val ppl: 9.53
Epoch 002/60 | train loss: 1.7068 | val loss: 1.9687 | val acc: 0.4700 | val ppl: 7.16
Epoch 003/60 | train loss: 1.1749 | val loss: 1.8681 | val acc: 0.5154 | val ppl: 6.48
Epoch 004/60 | train loss: 0.8050 | val loss: 1.9650 | val acc: 0.5251 | val ppl: 7.13
Epoch 005/60 | train loss: 0.5759 | val loss: 2.0550 | val acc: 0.5369 | val ppl: 7.81
Epoch 006/60 | train loss: 0.4362 | val loss: 2.1417 | val acc: 0.5437 | val ppl: 8.51
Epoch 007/60 | train loss: 0.3588 | val loss: 2.2132 | val acc: 0.5385 | val ppl: 9.14
Epoch 008/60 | train loss: 0.3125 | val loss: 2.2810 | val acc: 0.5499 | val ppl: 9.79
Epoch 009/60 | train loss: 0.2830 | val loss: 2.3763 | val acc: 0.5399 | val ppl: 10.76
Epoch 010/60 | train loss: 0.2642 | val loss: 2.4143 | val acc: 0.5435 | val ppl: 11.18
Epoch 011/60 | train loss: 0.2492 | val loss: 2.4503 | val acc: 0.5461 | val ppl: 11.59
Epoch 012/60 | train loss: 0.2368 | val loss: 2.4672 | val acc: 0.5444 | val ppl: 11.79
Epoch 013/60 | train loss: 0.2304 | val loss: 2.4928 | val acc: 0.5432 | val ppl: 12.10
Epoch 014/60 | train loss: 0.2231 | val loss: 2.5094 | val acc: 0.5413 | val ppl: 12.30
Epoch 015/60 | train loss: 0.2171 | val loss: 2.5169 | val acc: 0.5586 | val ppl: 12.39
Epoch 016/60 | train loss: 0.2125 | val loss: 2.5603 | val acc: 0.5556 | val ppl: 12.94
Epoch 017/60 | train loss: 0.2091 | val loss: 2.5847 | val acc: 0.5406 | val ppl: 13.26
Epoch 018/60 | train loss: 0.2050 | val loss: 2.6176 | val acc: 0.5343 | val ppl: 13.70
Epoch 019/60 | train loss: 0.2023 | val loss: 2.6531 | val acc: 0.5436 | val ppl: 14.20
Epoch 020/60 | train loss: 0.1998 | val loss: 2.6758 | val acc: 0.5344 | val ppl: 14.52
Epoch 021/60 | train loss: 0.1978 | val loss: 2.7204 | val acc: 0.5365 | val ppl: 15.19
Epoch 022/60 | train loss: 0.1963 | val loss: 2.6772 | val acc: 0.5406 | val ppl: 14.54
Epoch 023/60 | train loss: 0.1928 | val loss: 2.7464 | val acc: 0.5340 | val ppl: 15.59
Epoch 024/60 | train loss: 0.1925 | val loss: 2.7948 | val acc: 0.5297 | val ppl: 16.36
Epoch 025/60 | train loss: 0.1912 | val loss: 2.7632 | val acc: 0.5308 | val ppl: 15.85
Epoch 026/60 | train loss: 0.1894 | val loss: 2.8320 | val acc: 0.5329 | val ppl: 16.98
Epoch 027/60 | train loss: 0.1890 | val loss: 2.8325 | val acc: 0.5342 | val ppl: 16.99
Epoch 028/60 | train loss: 0.1877 | val loss: 2.8369 | val acc: 0.5289 | val ppl: 17.06
Epoch 029/60 | train loss: 0.1867 | val loss: 2.8671 | val acc: 0.5292 | val ppl: 17.59
Epoch 030/60 | train loss: 0.1860 | val loss: 2.8445 | val acc: 0.5325 | val ppl: 17.19
Epoch 031/60 | train loss: 0.1846 | val loss: 2.8820 | val acc: 0.5324 | val ppl: 17.85
Epoch 032/60 | train loss: 0.1848 | val loss: 2.8822 | val acc: 0.5337 | val ppl: 17.85
Epoch 033/60 | train loss: 0.1849 | val loss: 2.8831 | val acc: 0.5324 | val ppl: 17.87
Epoch 034/60 | train loss: 0.1836 | val loss: 2.8753 | val acc: 0.5422 | val ppl: 17.73
Epoch 035/60 | train loss: 0.1827 | val loss: 2.8681 | val acc: 0.5350 | val ppl: 17.60
Epoch 036/60 | train loss: 0.1821 | val loss: 2.8905 | val acc: 0.5268 | val ppl: 18.00
Epoch 037/60 | train loss: 0.1814 | val loss: 2.9349 | val acc: 0.5431 | val ppl: 18.82
Epoch 038/60 | train loss: 0.1807 | val loss: 2.9331 | val acc: 0.5397 | val ppl: 18.79
Epoch 039/60 | train loss: 0.1805 | val loss: 2.9398 | val acc: 0.5369 | val ppl: 18.91
Epoch 040/60 | train loss: 0.1806 | val loss: 2.9698 | val acc: 0.5360 | val ppl: 19.49
Epoch 041/60 | train loss: 0.1795 | val loss: 2.9330 | val acc: 0.5382 | val ppl: 18.78
Epoch 042/60 | train loss: 0.1802 | val loss: 3.0835 | val acc: 0.5242 | val ppl: 21.83
Epoch 043/60 | train loss: 0.1794 | val loss: 3.0180 | val acc: 0.5296 | val ppl: 20.45
Epoch 044/60 | train loss: 0.1781 | val loss: 2.9836 | val acc: 0.5333 | val ppl: 19.76
Epoch 045/60 | train loss: 0.1789 | val loss: 3.0059 | val acc: 0.5349 | val ppl: 20.20
Epoch 046/60 | train loss: 0.1783 | val loss: 3.0625 | val acc: 0.5306 | val ppl: 21.38
Epoch 047/60 | train loss: 0.1783 | val loss: 3.0117 | val acc: 0.5383 | val ppl: 20.32
Epoch 048/60 | train loss: 0.1771 | val loss: 3.0017 | val acc: 0.5358 | val ppl: 20.12
Epoch 049/60 | train loss: 0.1775 | val loss: 3.1157 | val acc: 0.5371 | val ppl: 22.55
Epoch 050/60 | train loss: 0.1778 | val loss: 3.0333 | val acc: 0.5492 | val ppl: 20.77
Epoch 051/60 | train loss: 0.1773 | val loss: 3.0571 | val acc: 0.5335 | val ppl: 21.27
Epoch 052/60 | train loss: 0.1769 | val loss: 3.1005 | val acc: 0.5247 | val ppl: 22.21
Epoch 053/60 | train loss: 0.1767 | val loss: 3.1145 | val acc: 0.5303 | val ppl: 22.52
Epoch 054/60 | train loss: 0.1760 | val loss: 3.1628 | val acc: 0.5303 | val ppl: 23.64
Epoch 055/60 | train loss: 0.1768 | val loss: 3.0640 | val acc: 0.5400 | val ppl: 21.41
Epoch 056/60 | train loss: 0.1754 | val loss: 3.0995 | val acc: 0.5328 | val ppl: 22.19
Epoch 057/60 | train loss: 0.1757 | val loss: 3.1469 | val acc: 0.5356 | val ppl: 23.26
Epoch 058/60 | train loss: 0.1754 | val loss: 3.1153 | val acc: 0.5339 | val ppl: 22.54
Epoch 059/60 | train loss: 0.1750 | val loss: 3.1088 | val acc: 0.5281 | val ppl: 22.40
Epoch 060/60 | train loss: 0.1743 | val loss: 3.1443 | val acc: 0.5319 | val ppl: 23.20

Generated sample:
Next character prediction involves the use of Recurrent Neural Networks (RNNs), and more specifically, a variant called Long Short-Term Memory (LSTM) networks. RNNs are particularly well-suited for sequential data like text, as they can maintain information in 'memory' about
```

```text
         dataset model_type  sequence_length  embedding_size  hidden_size  \
0  Provided Text       LSTM               10              64          128   
1  Provided Text       LSTM               20              64          128   
2  Provided Text       LSTM               30              64          128   

   num_layers  fc_hidden_size  final_train_loss  final_val_loss  \
0           1               0          0.492500        2.806571   
1           1               0          0.257663        3.075028   
2           1               0          0.174311        3.144315   

   final_val_accuracy  final_val_perplexity  training_time_sec  \
0            0.512500             16.553061           4.047806   
1            0.517292             21.650498           4.064672   
2            0.531944             23.203768           4.100124   

   inference_time_sec  trainable_parameters  model_size_mb  \
0            0.126669                108013       0.412037   
1            0.125014                108013       0.412037   
2            0.125857                108013       0.412037   

   approx_ops_per_sequence  
0                  1040640  
1                  2081280  
2                  3121920  
```

---

## Cell 7 — Code

```python
# ============================================================
# 6B. Problem 2 - LSTM on Tiny Shakespeare
# ============================================================

tiny_shakespeare_text = load_tiny_shakespeare(max_chars=SHAKESPEARE_MAX_CHARS)

print("Tiny Shakespeare characters:", len(tiny_shakespeare_text))
print("Tiny Shakespeare vocabulary size:", len(set(tiny_shakespeare_text)))
print("First 300 characters:")
print(tiny_shakespeare_text[:300])


lstm_shakespeare_configs = [
    {
        "name": "LSTM_seq20_base",
        "seq_len": 20,
        "embed_size": 64,
        "hidden_size": 128,
        "num_layers": 1,
        "fc_hidden_size": 0,
        "dropout": 0.0,
    },
    {
        "name": "LSTM_seq30_base",
        "seq_len": 30,
        "embed_size": 64,
        "hidden_size": 128,
        "num_layers": 1,
        "fc_hidden_size": 0,
        "dropout": 0.0,
    },
    {
        "name": "LSTM_seq50_base",
        "seq_len": 50,
        "embed_size": 64,
        "hidden_size": 128,
        "num_layers": 1,
        "fc_hidden_size": 0,
        "dropout": 0.0,
    },
    {
        "name": "LSTM_seq30_smaller_hidden",
        "seq_len": 30,
        "embed_size": 64,
        "hidden_size": 64,
        "num_layers": 1,
        "fc_hidden_size": 0,
        "dropout": 0.0,
    },
    {
        "name": "LSTM_seq30_two_layers",
        "seq_len": 30,
        "embed_size": 64,
        "hidden_size": 128,
        "num_layers": 2,
        "fc_hidden_size": 0,
        "dropout": 0.2,
    },
    {
        "name": "LSTM_seq30_fc_head",
        "seq_len": 30,
        "embed_size": 64,
        "hidden_size": 128,
        "num_layers": 1,
        "fc_hidden_size": 128,
        "dropout": 0.0,
    },
]

p2_lstm_shakespeare_results = []
p2_lstm_shakespeare_histories = []
p2_lstm_shakespeare_samples = []

for cfg in lstm_shakespeare_configs:
    print_section(cfg["name"])

    model, result_df, history_df, sample = run_single_experiment(
        text=tiny_shakespeare_text,
        dataset_name="Tiny Shakespeare",
        model_type="LSTM",
        seq_len=cfg["seq_len"],
        epochs=SHAKESPEARE_EPOCHS,
        batch_size=SHAKESPEARE_BATCH_SIZE,
        embed_size=cfg["embed_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        fc_hidden_size=cfg["fc_hidden_size"],
        dropout=cfg["dropout"],
        learning_rate=LEARNING_RATE,
        grad_clip=GRAD_CLIP,
        generate_chars=GENERATE_CHARS_SHAKESPEARE,
        start_text="ROMEO:",
    )

    result_df["config_description"] = cfg["name"]
    history_df["config_description"] = cfg["name"]
    sample["config_description"] = cfg["name"]

    p2_lstm_shakespeare_results.append(result_df)
    p2_lstm_shakespeare_histories.append(history_df)
    p2_lstm_shakespeare_samples.append(sample)

p2_lstm_shakespeare_results_df = pd.concat(p2_lstm_shakespeare_results, ignore_index=True)
p2_lstm_shakespeare_history_df = pd.concat(p2_lstm_shakespeare_histories, ignore_index=True)

save_results_csv(p2_lstm_shakespeare_results_df, "problem2_lstm_tiny_shakespeare_results.csv")
save_results_csv(p2_lstm_shakespeare_history_df, "problem2_lstm_tiny_shakespeare_history.csv")

display_clean_results(p2_lstm_shakespeare_results_df)
```

### Output

**stdout:**

```text
Downloading Tiny Shakespeare dataset...
Tiny Shakespeare characters: 1115394
Tiny Shakespeare vocabulary size: 65
First 300 characters:
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us
```

### LSTM_seq20_base

**stdout:**

```text
================================================================================
Dataset: Tiny Shakespeare
Model: LSTM | seq_len=20 | embed=64 | hidden=128 | layers=1 | fc_hidden=0
================================================================================
Epoch 001/10 | train loss: 1.6617 | val loss: 1.6978 | val acc: 0.4969 | val ppl: 5.46
Epoch 002/10 | train loss: 1.4868 | val loss: 1.6688 | val acc: 0.5069 | val ppl: 5.31
Epoch 003/10 | train loss: 1.4564 | val loss: 1.6601 | val acc: 0.5092 | val ppl: 5.26
Epoch 004/10 | train loss: 1.4409 | val loss: 1.6612 | val acc: 0.5115 | val ppl: 5.27
Epoch 005/10 | train loss: 1.4311 | val loss: 1.6642 | val acc: 0.5121 | val ppl: 5.28
Epoch 006/10 | train loss: 1.4240 | val loss: 1.6689 | val acc: 0.5120 | val ppl: 5.31
Epoch 007/10 | train loss: 1.4189 | val loss: 1.6679 | val acc: 0.5115 | val ppl: 5.30
Epoch 008/10 | train loss: 1.4148 | val loss: 1.6682 | val acc: 0.5142 | val ppl: 5.30
Epoch 009/10 | train loss: 1.4114 | val loss: 1.6700 | val acc: 0.5142 | val ppl: 5.31
Epoch 010/10 | train loss: 1.4086 | val loss: 1.6737 | val acc: 0.5125 | val ppl: 5.33

Generated sample:
ROMEO:
That's some a child on the noble good sons:
And if I do make before the king, which should not thou hear
That may tell you, then I wish the truth.

QUEEN ELIZABETH:
God kill'd both the unto the pursuin the boy: I'll compare to lives to keep me,
Which thus to my death?
Or you in injurce to your brother's
must pity for the Sicils of this life
That may speak of usurper redoubled man?

Second Was above you that
Before the daffody perforelves from a name in heaven have to show us with that was in th
```

### LSTM_seq30_base

**stdout:**

```text
================================================================================
Dataset: Tiny Shakespeare
Model: LSTM | seq_len=30 | embed=64 | hidden=128 | layers=1 | fc_hidden=0
================================================================================
Epoch 001/10 | train loss: 1.6246 | val loss: 1.6608 | val acc: 0.5107 | val ppl: 5.26
Epoch 002/10 | train loss: 1.4335 | val loss: 1.6401 | val acc: 0.5170 | val ppl: 5.16
Epoch 003/10 | train loss: 1.4018 | val loss: 1.6410 | val acc: 0.5210 | val ppl: 5.16
Epoch 004/10 | train loss: 1.3860 | val loss: 1.6455 | val acc: 0.5213 | val ppl: 5.18
Epoch 005/10 | train loss: 1.3758 | val loss: 1.6496 | val acc: 0.5230 | val ppl: 5.21
Epoch 006/10 | train loss: 1.3687 | val loss: 1.6563 | val acc: 0.5233 | val ppl: 5.24
Epoch 007/10 | train loss: 1.3632 | val loss: 1.6619 | val acc: 0.5232 | val ppl: 5.27
Epoch 008/10 | train loss: 1.3592 | val loss: 1.6724 | val acc: 0.5232 | val ppl: 5.32
Epoch 009/10 | train loss: 1.3559 | val loss: 1.6704 | val acc: 0.5239 | val ppl: 5.31
Epoch 010/10 | train loss: 1.3531 | val loss: 1.6752 | val acc: 0.5230 | val ppl: 5.34

Generated sample:
ROMEO:
What husband, curse me to him?

MENENIUS:
I be not help it.

FRIAR LAURENCE:
A ningrating frowns,
Becomes court this credit in the ends what me at a while up, and atils for pity.

VIRGILIA:
Will you say and the care to hide thee the mind to the best.

KING RICHARD III:
How, see he shall come her to the cause,
To Pates, friend himself call him forth our satisfore him me be mine.

KING HENRY VI:
The sea go to keep those suspire to hear me to the world?
And in hians, for this intelligence to find 
```

### LSTM_seq50_base

**stdout:**

```text
================================================================================
Dataset: Tiny Shakespeare
Model: LSTM | seq_len=50 | embed=64 | hidden=128 | layers=1 | fc_hidden=0
================================================================================
Epoch 001/10 | train loss: 1.5617 | val loss: 1.6135 | val acc: 0.5237 | val ppl: 5.02
Epoch 002/10 | train loss: 1.3722 | val loss: 1.6172 | val acc: 0.5288 | val ppl: 5.04
Epoch 003/10 | train loss: 1.3410 | val loss: 1.6233 | val acc: 0.5301 | val ppl: 5.07
Epoch 004/10 | train loss: 1.3253 | val loss: 1.6268 | val acc: 0.5297 | val ppl: 5.09
Epoch 005/10 | train loss: 1.3154 | val loss: 1.6363 | val acc: 0.5296 | val ppl: 5.14
Epoch 006/10 | train loss: 1.3087 | val loss: 1.6429 | val acc: 0.5297 | val ppl: 5.17
Epoch 007/10 | train loss: 1.3037 | val loss: 1.6419 | val acc: 0.5301 | val ppl: 5.16
Epoch 008/10 | train loss: 1.2999 | val loss: 1.6443 | val acc: 0.5313 | val ppl: 5.18
Epoch 009/10 | train loss: 1.2968 | val loss: 1.6472 | val acc: 0.5308 | val ppl: 5.19
Epoch 010/10 | train loss: 1.2942 | val loss: 1.6584 | val acc: 0.5291 | val ppl: 5.25

Generated sample:
ROMEO:
O, then we born to be person bound thereof
To tell thee and sour children I'll find his wit took the services
and blessed the duck of thy noble as the fair and conqueiture pardon
That 'twere for't to me distrain to him!

MENENIUS:
Peace, we will bear her children flesh,
And though Clarence make it the green too, ever!

POMPEY:
My lord, yet what is a peace.

NORTHUMBERLAND:
Harp more that we of me to fetch thy news? how should
Show me alone: I am nothing can sighs, the says, all me.

First Lord:
```

### LSTM_seq30_smaller_hidden

**stdout:**

```text
================================================================================
Dataset: Tiny Shakespeare
Model: LSTM | seq_len=30 | embed=64 | hidden=64 | layers=1 | fc_hidden=0
================================================================================
Epoch 001/10 | train loss: 1.7683 | val loss: 1.7796 | val acc: 0.4762 | val ppl: 5.93
Epoch 002/10 | train loss: 1.5843 | val loss: 1.7535 | val acc: 0.4872 | val ppl: 5.77
Epoch 003/10 | train loss: 1.5547 | val loss: 1.7422 | val acc: 0.4906 | val ppl: 5.71
Epoch 004/10 | train loss: 1.5406 | val loss: 1.7390 | val acc: 0.4931 | val ppl: 5.69
Epoch 005/10 | train loss: 1.5320 | val loss: 1.7380 | val acc: 0.4934 | val ppl: 5.69
Epoch 006/10 | train loss: 1.5261 | val loss: 1.7362 | val acc: 0.4965 | val ppl: 5.68
Epoch 007/10 | train loss: 1.5217 | val loss: 1.7358 | val acc: 0.4989 | val ppl: 5.67
Epoch 008/10 | train loss: 1.5182 | val loss: 1.7357 | val acc: 0.4979 | val ppl: 5.67
Epoch 009/10 | train loss: 1.5152 | val loss: 1.7328 | val acc: 0.4972 | val ppl: 5.66
Epoch 010/10 | train loss: 1.5127 | val loss: 1.7337 | val acc: 0.4993 | val ppl: 5.66

Generated sample:
ROMEO:
They not, since to her the luit: all wonting, to this way the father, I have consent if we kneephard, my villans of the very
To company find, in Cronted of your more the pevisure and where I cannot dreams!
Why, I proud; but need, and this lord, they are we so, which that, if knowlight is to have have reveness the blood
'Tis their glove, I will not perget be late no man again: near more, we be power his breaks you made
With you, sir, my man of love live to a husband, the valon.

MERCUTIO:
Cousit
```

### LSTM_seq30_two_layers

**stdout:**

```text
================================================================================
Dataset: Tiny Shakespeare
Model: LSTM | seq_len=30 | embed=64 | hidden=128 | layers=2 | fc_hidden=0
================================================================================
Epoch 001/10 | train loss: 1.6327 | val loss: 1.6087 | val acc: 0.5208 | val ppl: 5.00
Epoch 002/10 | train loss: 1.4385 | val loss: 1.5946 | val acc: 0.5274 | val ppl: 4.93
Epoch 003/10 | train loss: 1.4089 | val loss: 1.5883 | val acc: 0.5293 | val ppl: 4.90
Epoch 004/10 | train loss: 1.3940 | val loss: 1.5896 | val acc: 0.5321 | val ppl: 4.90
Epoch 005/10 | train loss: 1.3845 | val loss: 1.5805 | val acc: 0.5322 | val ppl: 4.86
Epoch 006/10 | train loss: 1.3780 | val loss: 1.5874 | val acc: 0.5317 | val ppl: 4.89
Epoch 007/10 | train loss: 1.3731 | val loss: 1.5854 | val acc: 0.5319 | val ppl: 4.88
Epoch 008/10 | train loss: 1.3692 | val loss: 1.5835 | val acc: 0.5337 | val ppl: 4.87
Epoch 009/10 | train loss: 1.3659 | val loss: 1.5837 | val acc: 0.5317 | val ppl: 4.87
Epoch 010/10 | train loss: 1.3634 | val loss: 1.5824 | val acc: 0.5332 | val ppl: 4.87

Generated sample:
ROMEO:
'Tis thrown those instrument.

HENRY PERCY:
Marry, be play; he is
A pity, sirs, that come, whose threw o'er him.

BUCKINGHAM:
Bastings to my sir; one that thou wast's content
That party ow thy love to our head, which on your pale
part, is the mind and let's have lay thy both warm is the fairest privilege of death.
If but the trial her head and could follow the duke for thee.

WARWICK:
Pronounce our time and that I read itself that will live it, you could not have
so once with the nurse of York;
```

### LSTM_seq30_fc_head

**stdout:**

```text
================================================================================
Dataset: Tiny Shakespeare
Model: LSTM | seq_len=30 | embed=64 | hidden=128 | layers=1 | fc_hidden=128
================================================================================
Epoch 001/10 | train loss: 1.5584 | val loss: 1.6561 | val acc: 0.5178 | val ppl: 5.24
Epoch 002/10 | train loss: 1.3884 | val loss: 1.6551 | val acc: 0.5226 | val ppl: 5.23
Epoch 003/10 | train loss: 1.3571 | val loss: 1.6610 | val acc: 0.5238 | val ppl: 5.26
Epoch 004/10 | train loss: 1.3402 | val loss: 1.6766 | val acc: 0.5231 | val ppl: 5.35
Epoch 005/10 | train loss: 1.3294 | val loss: 1.6839 | val acc: 0.5230 | val ppl: 5.39
Epoch 006/10 | train loss: 1.3215 | val loss: 1.6901 | val acc: 0.5233 | val ppl: 5.42
Epoch 007/10 | train loss: 1.3158 | val loss: 1.6984 | val acc: 0.5219 | val ppl: 5.47
Epoch 008/10 | train loss: 1.3114 | val loss: 1.6920 | val acc: 0.5251 | val ppl: 5.43
Epoch 009/10 | train loss: 1.3076 | val loss: 1.7025 | val acc: 0.5236 | val ppl: 5.49
Epoch 010/10 | train loss: 1.3047 | val loss: 1.7006 | val acc: 0.5240 | val ppl: 5.48

Generated sample:
ROMEO:
His wife, know the yoke and that is a ballad and tear for the
base subjects to see her nurse, who hath the place will take we for my pains, and with the great blood,
Contidam sheep-sheartory of your grace's brow.
And'liety give me hurl me spelior with please. I am more
That Bolingbroke I met my son is hearing preserved.

KATHARINA:
I thank your crimes; so dancly himself beseech you, for his personace
That woman, give me suitors are has not spoke from me more,
Come, lords, they do not do remain 
```

```text
            dataset model_type  sequence_length  embedding_size  hidden_size  \
0  Tiny Shakespeare       LSTM               20              64          128   
3  Tiny Shakespeare       LSTM               30              64           64   
1  Tiny Shakespeare       LSTM               30              64          128   
5  Tiny Shakespeare       LSTM               30              64          128   
4  Tiny Shakespeare       LSTM               30              64          128   
2  Tiny Shakespeare       LSTM               50              64          128   

   num_layers  fc_hidden_size  final_train_loss  final_val_loss  \
0           1               0          1.408550        1.673675   
3           1               0          1.512735        1.733700   
1           1               0          1.353119        1.675177   
5           1             128          1.304747        1.700598   
4           2               0          1.363384        1.582417   
2           1               0          1.294236        1.658390   

   final_val_accuracy  final_val_perplexity  training_time_sec  \
0            0.512472              5.331725          71.172834   
3            0.499253              5.661564          70.851092   
1            0.522989              5.339740          72.508646   
5            0.524040              5.477220          75.469431   
4            0.533166              4.866706          88.641188   
2            0.529108              5.250850          80.596546   

   inference_time_sec  trainable_parameters  model_size_mb  \
0            0.432840                111873       0.426762   
3            0.151742                 41665       0.158939   
1            0.367741                111873       0.426762   
5            0.376018                128385       0.489750   
4            0.639143                243969       0.930668   
2            0.366469                111873       0.426762   

   approx_ops_per_sequence  
0                  2132480  
3                  1107840  
1                  3198720  
5                  3690240  
4                  7130880  
2                  5331200  
```

---

## Cell 8 — Code `In [17]`

```python
# ============================================================
# 7. Problem 2 plots and results
# ============================================================

print_section("LSTM provided text result table")
display_clean_results(p2_lstm_fixed_results_df)

print_section("LSTM provided text training loss")
plot_training_curves(
    p2_lstm_fixed_histories,
    title="LSTM Training Loss - Provided Text",
    metric="train_loss",
    save_name="lstm_provided_training_loss.png",
)

print_section("LSTM provided text validation accuracy")
plot_training_curves(
    p2_lstm_fixed_histories,
    title="LSTM Validation Accuracy - Provided Text",
    metric="val_accuracy",
    save_name="lstm_provided_validation_accuracy.png",
)

print_section("LSTM generated samples on provided text")
show_samples(p2_lstm_fixed_samples)

print_section("LSTM Tiny Shakespeare result table")
display_clean_results(p2_lstm_shakespeare_results_df)

print_section("LSTM Tiny Shakespeare training loss")
plot_training_curves(
    p2_lstm_shakespeare_histories,
    title="LSTM Training Loss - Tiny Shakespeare",
    metric="train_loss",
    save_name="lstm_shakespeare_training_loss.png",
)

print_section("LSTM Tiny Shakespeare validation loss")
plot_training_curves(
    p2_lstm_shakespeare_histories,
    title="LSTM Validation Loss - Tiny Shakespeare",
    metric="val_loss",
    save_name="lstm_shakespeare_validation_loss.png",
)

print_section("LSTM Tiny Shakespeare validation accuracy")
plot_training_curves(
    p2_lstm_shakespeare_histories,
    title="LSTM Validation Accuracy - Tiny Shakespeare",
    metric="val_accuracy",
    save_name="lstm_shakespeare_validation_accuracy.png",
)

print_section("LSTM Tiny Shakespeare perplexity")
plot_training_curves(
    p2_lstm_shakespeare_histories,
    title="LSTM Validation Perplexity - Tiny Shakespeare",
    metric="val_perplexity",
    save_name="lstm_shakespeare_perplexity.png",
)

print_section("LSTM Tiny Shakespeare model size")
plot_result_bar(
    p2_lstm_shakespeare_results_df,
    x_col="sequence_length",
    y_col="model_size_mb",
    group_col="config_description",
    title="LSTM Model Size - Tiny Shakespeare",
    save_name="lstm_shakespeare_model_size.png",
)

print_section("LSTM Tiny Shakespeare training time")
plot_result_bar(
    p2_lstm_shakespeare_results_df,
    x_col="sequence_length",
    y_col="training_time_sec",
    group_col="config_description",
    title="LSTM Training Time - Tiny Shakespeare",
    save_name="lstm_shakespeare_training_time.png",
)

print_section("LSTM generated Shakespeare samples")
show_samples(p2_lstm_shakespeare_samples)
```

### Output

### LSTM provided text result table

```text
         dataset model_type  sequence_length  embedding_size  hidden_size  \
0  Provided Text       LSTM               10              64          128   
1  Provided Text       LSTM               20              64          128   
2  Provided Text       LSTM               30              64          128   

   num_layers  fc_hidden_size  final_train_loss  final_val_loss  \
0           1               0          0.492500        2.806571   
1           1               0          0.257663        3.075028   
2           1               0          0.174311        3.144315   

   final_val_accuracy  final_val_perplexity  training_time_sec  \
0            0.512500             16.553061           4.047806   
1            0.517292             21.650498           4.064672   
2            0.531944             23.203768           4.100124   

   inference_time_sec  trainable_parameters  model_size_mb  \
0            0.126669                108013       0.412037   
1            0.125014                108013       0.412037   
2            0.125857                108013       0.412037   

   approx_ops_per_sequence  
0                  1040640  
1                  2081280  
2                  3121920  
```

### LSTM provided text training loss

![Cell 8 output 4](README_assets/cell_08_output_04_005.png)

### LSTM provided text validation accuracy

![Cell 8 output 6](README_assets/cell_08_output_06_006.png)

### LSTM generated samples on provided text

#### Provided_Text_LSTM_seq10_h128_layers1_fc0

```text
Next character prediction involves prediction tasks.

Training a sequence of character prediction involves feeding it large amounts of text data, allowing a sequence of character prediction tasks.

Training a sequence of character is most likely to follow. These prediction i
```

#### Provided_Text_LSTM_seq20_h128_layers1_fc0

```text
Next character prediction plays a crucial role in enhancing the capabilities of various NLP applications, making text-based ith character prediction is a fundamental task in the field of natural language processing (NLP) that involves predicting the next character prediction
```

#### Provided_Text_LSTM_seq30_h128_layers1_fc0

```text
Next character prediction involves the use of Recurrent Neural Networks (RNNs), and more specifically, a variant called Long Short-Term Memory (LSTM) networks. RNNs are particularly well-suited for sequential data like text, as they can maintain information in 'memory' about
```

### LSTM Tiny Shakespeare result table

```text
            dataset model_type  sequence_length  embedding_size  hidden_size  \
0  Tiny Shakespeare       LSTM               20              64          128   
3  Tiny Shakespeare       LSTM               30              64           64   
1  Tiny Shakespeare       LSTM               30              64          128   
5  Tiny Shakespeare       LSTM               30              64          128   
4  Tiny Shakespeare       LSTM               30              64          128   
2  Tiny Shakespeare       LSTM               50              64          128   

   num_layers  fc_hidden_size  final_train_loss  final_val_loss  \
0           1               0          1.408550        1.673675   
3           1               0          1.512735        1.733700   
1           1               0          1.353119        1.675177   
5           1             128          1.304747        1.700598   
4           2               0          1.363384        1.582417   
2           1               0          1.294236        1.658390   

   final_val_accuracy  final_val_perplexity  training_time_sec  \
0            0.512472              5.331725          71.172834   
3            0.499253              5.661564          70.851092   
1            0.522989              5.339740          72.508646   
5            0.524040              5.477220          75.469431   
4            0.533166              4.866706          88.641188   
2            0.529108              5.250850          80.596546   

   inference_time_sec  trainable_parameters  model_size_mb  \
0            0.432840                111873       0.426762   
3            0.151742                 41665       0.158939   
1            0.367741                111873       0.426762   
5            0.376018                128385       0.489750   
4            0.639143                243969       0.930668   
2            0.366469                111873       0.426762   

   approx_ops_per_sequence  
0                  2132480  
3                  1107840  
1                  3198720  
5                  3690240  
4                  7130880  
2                  5331200  
```

### LSTM Tiny Shakespeare training loss

![Cell 8 output 14](README_assets/cell_08_output_14_007.png)

### LSTM Tiny Shakespeare validation loss

![Cell 8 output 16](README_assets/cell_08_output_16_008.png)

### LSTM Tiny Shakespeare validation accuracy

![Cell 8 output 18](README_assets/cell_08_output_18_009.png)

### LSTM Tiny Shakespeare perplexity

![Cell 8 output 20](README_assets/cell_08_output_20_010.png)

### LSTM Tiny Shakespeare model size

![Cell 8 output 22](README_assets/cell_08_output_22_011.png)

### LSTM Tiny Shakespeare training time

![Cell 8 output 24](README_assets/cell_08_output_24_012.png)

### LSTM generated Shakespeare samples

#### Tiny_Shakespeare_LSTM_seq20_h128_layers1_fc0

```text
ROMEO:
That's some a child on the noble good sons:
And if I do make before the king, which should not thou hear
That may tell you, then I wish the truth.

QUEEN ELIZABETH:
God kill'd both the unto the pursuin the boy: I'll compare to lives to keep me,
Which thus to my death?
Or you in injurce to your brother's
must pity for the Sicils of this life
That may speak of usurper redoubled man?

Second Was above you that
Before the daffody perforelves from a name in heaven have to show us with that was in th
```

#### Tiny_Shakespeare_LSTM_seq30_h128_layers1_fc0

```text
ROMEO:
What husband, curse me to him?

MENENIUS:
I be not help it.

FRIAR LAURENCE:
A ningrating frowns,
Becomes court this credit in the ends what me at a while up, and atils for pity.

VIRGILIA:
Will you say and the care to hide thee the mind to the best.

KING RICHARD III:
How, see he shall come her to the cause,
To Pates, friend himself call him forth our satisfore him me be mine.

KING HENRY VI:
The sea go to keep those suspire to hear me to the world?
And in hians, for this intelligence to find 
```

#### Tiny_Shakespeare_LSTM_seq50_h128_layers1_fc0

```text
ROMEO:
O, then we born to be person bound thereof
To tell thee and sour children I'll find his wit took the services
and blessed the duck of thy noble as the fair and conqueiture pardon
That 'twere for't to me distrain to him!

MENENIUS:
Peace, we will bear her children flesh,
And though Clarence make it the green too, ever!

POMPEY:
My lord, yet what is a peace.

NORTHUMBERLAND:
Harp more that we of me to fetch thy news? how should
Show me alone: I am nothing can sighs, the says, all me.

First Lord:
```

#### Tiny_Shakespeare_LSTM_seq30_h64_layers1_fc0

```text
ROMEO:
They not, since to her the luit: all wonting, to this way the father, I have consent if we kneephard, my villans of the very
To company find, in Cronted of your more the pevisure and where I cannot dreams!
Why, I proud; but need, and this lord, they are we so, which that, if knowlight is to have have reveness the blood
'Tis their glove, I will not perget be late no man again: near more, we be power his breaks you made
With you, sir, my man of love live to a husband, the valon.

MERCUTIO:
Cousit
```

#### Tiny_Shakespeare_LSTM_seq30_h128_layers2_fc0

```text
ROMEO:
'Tis thrown those instrument.

HENRY PERCY:
Marry, be play; he is
A pity, sirs, that come, whose threw o'er him.

BUCKINGHAM:
Bastings to my sir; one that thou wast's content
That party ow thy love to our head, which on your pale
part, is the mind and let's have lay thy both warm is the fairest privilege of death.
If but the trial her head and could follow the duke for thee.

WARWICK:
Pronounce our time and that I read itself that will live it, you could not have
so once with the nurse of York;
```

#### Tiny_Shakespeare_LSTM_seq30_h128_layers1_fc128

```text
ROMEO:
His wife, know the yoke and that is a ballad and tear for the
base subjects to see her nurse, who hath the place will take we for my pains, and with the great blood,
Contidam sheep-sheartory of your grace's brow.
And'liety give me hurl me spelior with please. I am more
That Bolingbroke I met my son is hearing preserved.

KATHARINA:
I thank your crimes; so dancly himself beseech you, for his personace
That woman, give me suitors are has not spoke from me more,
Come, lords, they do not do remain 
```

---

## Cell 9 — Code

```python
# ============================================================
# 8A. Problem 3 - GRU on provided paragraph
# ============================================================

p3_gru_fixed_results = []
p3_gru_fixed_histories = []
p3_gru_fixed_samples = []

for seq_len in FIXED_TEXT_SEQUENCE_LENGTHS:
    model, result_df, history_df, sample = run_single_experiment(
        text=NEXT_CHARACTER_TEXT,
        dataset_name="Provided Text",
        model_type="GRU",
        seq_len=seq_len,
        epochs=FIXED_TEXT_EPOCHS,
        batch_size=FIXED_TEXT_BATCH_SIZE,
        embed_size=64,
        hidden_size=128,
        num_layers=1,
        fc_hidden_size=0,
        dropout=0.0,
        learning_rate=LEARNING_RATE,
        grad_clip=GRAD_CLIP,
        generate_chars=GENERATE_CHARS_FIXED_TEXT,
        start_text="Next character prediction",
    )

    p3_gru_fixed_results.append(result_df)
    p3_gru_fixed_histories.append(history_df)
    p3_gru_fixed_samples.append(sample)

p3_gru_fixed_results_df = pd.concat(p3_gru_fixed_results, ignore_index=True)
p3_gru_fixed_history_df = pd.concat(p3_gru_fixed_histories, ignore_index=True)

save_results_csv(p3_gru_fixed_results_df, "problem3_gru_provided_text_results.csv")
save_results_csv(p3_gru_fixed_history_df, "problem3_gru_provided_text_history.csv")

display_clean_results(p3_gru_fixed_results_df)
```

### Output

**stdout:**

```text
================================================================================
Dataset: Provided Text
Model: GRU | seq_len=10 | embed=64 | hidden=128 | layers=1 | fc_hidden=0
================================================================================
Epoch 001/60 | train loss: 2.6145 | val loss: 2.3124 | val acc: 0.3646 | val ppl: 10.10
Epoch 002/60 | train loss: 1.8240 | val loss: 2.0534 | val acc: 0.4575 | val ppl: 7.79
Epoch 003/60 | train loss: 1.4219 | val loss: 1.9479 | val acc: 0.4829 | val ppl: 7.01
Epoch 004/60 | train loss: 1.1387 | val loss: 1.9867 | val acc: 0.4954 | val ppl: 7.29
Epoch 005/60 | train loss: 0.9457 | val loss: 2.0332 | val acc: 0.5038 | val ppl: 7.64
Epoch 006/60 | train loss: 0.8185 | val loss: 2.1423 | val acc: 0.4979 | val ppl: 8.52
Epoch 007/60 | train loss: 0.7388 | val loss: 2.2316 | val acc: 0.4900 | val ppl: 9.31
Epoch 008/60 | train loss: 0.6825 | val loss: 2.2475 | val acc: 0.5042 | val ppl: 9.46
Epoch 009/60 | train loss: 0.6443 | val loss: 2.3504 | val acc: 0.4967 | val ppl: 10.49
Epoch 010/60 | train loss: 0.6174 | val loss: 2.3696 | val acc: 0.5038 | val ppl: 10.69
Epoch 011/60 | train loss: 0.5988 | val loss: 2.4321 | val acc: 0.5071 | val ppl: 11.38
Epoch 012/60 | train loss: 0.5834 | val loss: 2.4593 | val acc: 0.4979 | val ppl: 11.70
Epoch 013/60 | train loss: 0.5737 | val loss: 2.5050 | val acc: 0.5000 | val ppl: 12.24
Epoch 014/60 | train loss: 0.5629 | val loss: 2.5212 | val acc: 0.4917 | val ppl: 12.44
Epoch 015/60 | train loss: 0.5526 | val loss: 2.5881 | val acc: 0.4996 | val ppl: 13.30
Epoch 016/60 | train loss: 0.5467 | val loss: 2.6087 | val acc: 0.4967 | val ppl: 13.58
Epoch 017/60 | train loss: 0.5433 | val loss: 2.6107 | val acc: 0.4975 | val ppl: 13.61
Epoch 018/60 | train loss: 0.5350 | val loss: 2.6398 | val acc: 0.5042 | val ppl: 14.01
Epoch 019/60 | train loss: 0.5327 | val loss: 2.6512 | val acc: 0.5042 | val ppl: 14.17
Epoch 020/60 | train loss: 0.5313 | val loss: 2.6684 | val acc: 0.5067 | val ppl: 14.42
Epoch 021/60 | train loss: 0.5264 | val loss: 2.7232 | val acc: 0.5025 | val ppl: 15.23
Epoch 022/60 | train loss: 0.5248 | val loss: 2.7061 | val acc: 0.5121 | val ppl: 14.97
Epoch 023/60 | train loss: 0.5216 | val loss: 2.7346 | val acc: 0.5050 | val ppl: 15.40
Epoch 024/60 | train loss: 0.5185 | val loss: 2.7287 | val acc: 0.5125 | val ppl: 15.31
Epoch 025/60 | train loss: 0.5176 | val loss: 2.7711 | val acc: 0.5062 | val ppl: 15.98
Epoch 026/60 | train loss: 0.5148 | val loss: 2.7915 | val acc: 0.5025 | val ppl: 16.31
Epoch 027/60 | train loss: 0.5143 | val loss: 2.7893 | val acc: 0.5071 | val ppl: 16.27
Epoch 028/60 | train loss: 0.5120 | val loss: 2.7725 | val acc: 0.5117 | val ppl: 16.00
Epoch 029/60 | train loss: 0.5112 | val loss: 2.8025 | val acc: 0.4996 | val ppl: 16.49
Epoch 030/60 | train loss: 0.5096 | val loss: 2.8086 | val acc: 0.5042 | val ppl: 16.59
Epoch 031/60 | train loss: 0.5084 | val loss: 2.8525 | val acc: 0.4983 | val ppl: 17.33
Epoch 032/60 | train loss: 0.5070 | val loss: 2.8479 | val acc: 0.5000 | val ppl: 17.25
Epoch 033/60 | train loss: 0.5077 | val loss: 2.8343 | val acc: 0.5008 | val ppl: 17.02
Epoch 034/60 | train loss: 0.5045 | val loss: 2.8835 | val acc: 0.5038 | val ppl: 17.88
Epoch 035/60 | train loss: 0.5035 | val loss: 2.8775 | val acc: 0.5083 | val ppl: 17.77
Epoch 036/60 | train loss: 0.5040 | val loss: 2.8977 | val acc: 0.5088 | val ppl: 18.13
Epoch 037/60 | train loss: 0.5012 | val loss: 2.9214 | val acc: 0.5025 | val ppl: 18.57
Epoch 038/60 | train loss: 0.5018 | val loss: 2.9092 | val acc: 0.5021 | val ppl: 18.34
Epoch 039/60 | train loss: 0.5000 | val loss: 2.9133 | val acc: 0.5083 | val ppl: 18.42
Epoch 040/60 | train loss: 0.5011 | val loss: 2.8807 | val acc: 0.5088 | val ppl: 17.83
Epoch 041/60 | train loss: 0.4985 | val loss: 2.9207 | val acc: 0.5067 | val ppl: 18.55
Epoch 042/60 | train loss: 0.4992 | val loss: 2.9205 | val acc: 0.5096 | val ppl: 18.55
Epoch 043/60 | train loss: 0.4976 | val loss: 2.9435 | val acc: 0.5033 | val ppl: 18.98
Epoch 044/60 | train loss: 0.4980 | val loss: 2.9301 | val acc: 0.5067 | val ppl: 18.73
Epoch 045/60 | train loss: 0.4970 | val loss: 2.9317 | val acc: 0.5092 | val ppl: 18.76
Epoch 046/60 | train loss: 0.4975 | val loss: 2.9473 | val acc: 0.4979 | val ppl: 19.05
Epoch 047/60 | train loss: 0.4970 | val loss: 2.9370 | val acc: 0.5033 | val ppl: 18.86
Epoch 048/60 | train loss: 0.4940 | val loss: 2.9761 | val acc: 0.5075 | val ppl: 19.61
Epoch 049/60 | train loss: 0.4952 | val loss: 2.9846 | val acc: 0.5000 | val ppl: 19.78
Epoch 050/60 | train loss: 0.4939 | val loss: 2.9753 | val acc: 0.5008 | val ppl: 19.60
Epoch 051/60 | train loss: 0.4932 | val loss: 2.9623 | val acc: 0.5062 | val ppl: 19.34
Epoch 052/60 | train loss: 0.4926 | val loss: 3.0122 | val acc: 0.4983 | val ppl: 20.33
Epoch 053/60 | train loss: 0.4910 | val loss: 3.0007 | val acc: 0.4979 | val ppl: 20.10
Epoch 054/60 | train loss: 0.4922 | val loss: 2.9993 | val acc: 0.5158 | val ppl: 20.07
Epoch 055/60 | train loss: 0.4919 | val loss: 3.0233 | val acc: 0.5067 | val ppl: 20.56
Epoch 056/60 | train loss: 0.4900 | val loss: 3.0140 | val acc: 0.5083 | val ppl: 20.37
Epoch 057/60 | train loss: 0.4907 | val loss: 3.0215 | val acc: 0.5067 | val ppl: 20.52
Epoch 058/60 | train loss: 0.4911 | val loss: 3.0340 | val acc: 0.5125 | val ppl: 20.78
Epoch 059/60 | train loss: 0.4899 | val loss: 3.0067 | val acc: 0.5133 | val ppl: 20.22
Epoch 060/60 | train loss: 0.4902 | val loss: 3.0641 | val acc: 0.5046 | val ppl: 21.42

Generated sample:
Next character predictions and the actual outcomes, thus improving its prediction of the model.

One of the most popular approaches to next character prediction involves predictive accuracy over time.

Once trained, the model adjusts its parameters to minimize the difference

================================================================================
Dataset: Provided Text
Model: GRU | seq_len=20 | embed=64 | hidden=128 | layers=1 | fc_hidden=0
================================================================================
Epoch 001/60 | train loss: 2.4967 | val loss: 2.1657 | val acc: 0.4167 | val ppl: 8.72
Epoch 002/60 | train loss: 1.5875 | val loss: 1.8765 | val acc: 0.5154 | val ppl: 6.53
Epoch 003/60 | train loss: 1.0672 | val loss: 1.8305 | val acc: 0.5429 | val ppl: 6.24
Epoch 004/60 | train loss: 0.7286 | val loss: 1.9145 | val acc: 0.5563 | val ppl: 6.78
Epoch 005/60 | train loss: 0.5375 | val loss: 1.9913 | val acc: 0.5617 | val ppl: 7.32
Epoch 006/60 | train loss: 0.4388 | val loss: 2.1228 | val acc: 0.5640 | val ppl: 8.35
Epoch 007/60 | train loss: 0.3838 | val loss: 2.1829 | val acc: 0.5729 | val ppl: 8.87
Epoch 008/60 | train loss: 0.3531 | val loss: 2.2364 | val acc: 0.5696 | val ppl: 9.36
Epoch 009/60 | train loss: 0.3338 | val loss: 2.2835 | val acc: 0.5648 | val ppl: 9.81
Epoch 010/60 | train loss: 0.3205 | val loss: 2.3499 | val acc: 0.5669 | val ppl: 10.48
Epoch 011/60 | train loss: 0.3113 | val loss: 2.3974 | val acc: 0.5600 | val ppl: 10.99
Epoch 012/60 | train loss: 0.3045 | val loss: 2.4084 | val acc: 0.5710 | val ppl: 11.12
Epoch 013/60 | train loss: 0.2994 | val loss: 2.4685 | val acc: 0.5617 | val ppl: 11.81
Epoch 014/60 | train loss: 0.2943 | val loss: 2.4849 | val acc: 0.5723 | val ppl: 12.00
Epoch 015/60 | train loss: 0.2900 | val loss: 2.5382 | val acc: 0.5602 | val ppl: 12.66
Epoch 016/60 | train loss: 0.2869 | val loss: 2.5666 | val acc: 0.5531 | val ppl: 13.02
Epoch 017/60 | train loss: 0.2844 | val loss: 2.5462 | val acc: 0.5590 | val ppl: 12.76
Epoch 018/60 | train loss: 0.2820 | val loss: 2.5866 | val acc: 0.5615 | val ppl: 13.29
Epoch 019/60 | train loss: 0.2808 | val loss: 2.6071 | val acc: 0.5540 | val ppl: 13.56
Epoch 020/60 | train loss: 0.2779 | val loss: 2.6380 | val acc: 0.5579 | val ppl: 13.99
Epoch 021/60 | train loss: 0.2748 | val loss: 2.6808 | val acc: 0.5571 | val ppl: 14.60
Epoch 022/60 | train loss: 0.2742 | val loss: 2.6757 | val acc: 0.5635 | val ppl: 14.52
Epoch 023/60 | train loss: 0.2742 | val loss: 2.6805 | val acc: 0.5617 | val ppl: 14.59
Epoch 024/60 | train loss: 0.2737 | val loss: 2.7043 | val acc: 0.5671 | val ppl: 14.94
Epoch 025/60 | train loss: 0.2720 | val loss: 2.7391 | val acc: 0.5587 | val ppl: 15.47
Epoch 026/60 | train loss: 0.2708 | val loss: 2.7517 | val acc: 0.5698 | val ppl: 15.67
Epoch 027/60 | train loss: 0.2689 | val loss: 2.7509 | val acc: 0.5598 | val ppl: 15.66
Epoch 028/60 | train loss: 0.2686 | val loss: 2.7672 | val acc: 0.5652 | val ppl: 15.91
Epoch 029/60 | train loss: 0.2682 | val loss: 2.7431 | val acc: 0.5640 | val ppl: 15.54
Epoch 030/60 | train loss: 0.2673 | val loss: 2.7623 | val acc: 0.5613 | val ppl: 15.84
Epoch 031/60 | train loss: 0.2676 | val loss: 2.7635 | val acc: 0.5631 | val ppl: 15.86
Epoch 032/60 | train loss: 0.2663 | val loss: 2.8136 | val acc: 0.5644 | val ppl: 16.67
Epoch 033/60 | train loss: 0.2645 | val loss: 2.8194 | val acc: 0.5656 | val ppl: 16.77
Epoch 034/60 | train loss: 0.2637 | val loss: 2.8309 | val acc: 0.5608 | val ppl: 16.96
Epoch 035/60 | train loss: 0.2629 | val loss: 2.8602 | val acc: 0.5590 | val ppl: 17.47
Epoch 036/60 | train loss: 0.2630 | val loss: 2.8422 | val acc: 0.5758 | val ppl: 17.15
Epoch 037/60 | train loss: 0.2624 | val loss: 2.8495 | val acc: 0.5660 | val ppl: 17.28
Epoch 038/60 | train loss: 0.2614 | val loss: 2.8425 | val acc: 0.5627 | val ppl: 17.16
Epoch 039/60 | train loss: 0.2618 | val loss: 2.8601 | val acc: 0.5719 | val ppl: 17.46
Epoch 040/60 | train loss: 0.2613 | val loss: 2.8811 | val acc: 0.5581 | val ppl: 17.83
Epoch 041/60 | train loss: 0.2608 | val loss: 2.8817 | val acc: 0.5773 | val ppl: 17.85
Epoch 042/60 | train loss: 0.2603 | val loss: 2.9033 | val acc: 0.5696 | val ppl: 18.23
Epoch 043/60 | train loss: 0.2602 | val loss: 2.9219 | val acc: 0.5663 | val ppl: 18.58
Epoch 044/60 | train loss: 0.2593 | val loss: 2.9279 | val acc: 0.5698 | val ppl: 18.69
Epoch 045/60 | train loss: 0.2594 | val loss: 2.9394 | val acc: 0.5608 | val ppl: 18.90
Epoch 046/60 | train loss: 0.2588 | val loss: 2.9107 | val acc: 0.5671 | val ppl: 18.37
Epoch 047/60 | train loss: 0.2581 | val loss: 2.8975 | val acc: 0.5677 | val ppl: 18.13
Epoch 048/60 | train loss: 0.2589 | val loss: 2.9404 | val acc: 0.5629 | val ppl: 18.92
Epoch 049/60 | train loss: 0.2571 | val loss: 2.9459 | val acc: 0.5602 | val ppl: 19.03
Epoch 050/60 | train loss: 0.2577 | val loss: 2.9720 | val acc: 0.5648 | val ppl: 19.53
Epoch 051/60 | train loss: 0.2566 | val loss: 2.9781 | val acc: 0.5596 | val ppl: 19.65
Epoch 052/60 | train loss: 0.2566 | val loss: 2.9612 | val acc: 0.5610 | val ppl: 19.32
Epoch 053/60 | train loss: 0.2564 | val loss: 2.9750 | val acc: 0.5565 | val ppl: 19.59
Epoch 054/60 | train loss: 0.2558 | val loss: 2.9946 | val acc: 0.5590 | val ppl: 19.98
Epoch 055/60 | train loss: 0.2564 | val loss: 3.0227 | val acc: 0.5592 | val ppl: 20.55
Epoch 056/60 | train loss: 0.2564 | val loss: 3.0011 | val acc: 0.5565 | val ppl: 20.11
Epoch 057/60 | train loss: 0.2560 | val loss: 2.9951 | val acc: 0.5602 | val ppl: 19.99
Epoch 058/60 | train loss: 0.2550 | val loss: 2.9853 | val acc: 0.5692 | val ppl: 19.79
Epoch 059/60 | train loss: 0.2550 | val loss: 3.0177 | val acc: 0.5527 | val ppl: 20.44
Epoch 060/60 | train loss: 0.2554 | val loss: 3.0592 | val acc: 0.5563 | val ppl: 21.31

Generated sample:
Next character prediction involves feeding it large amounts of text data, allowing it to learn the prediction of the next character prediction involves feeding it large amounts of text data, allowing it to learn the probability of each character's appearance following a sequ

================================================================================
Dataset: Provided Text
Model: GRU | seq_len=30 | embed=64 | hidden=128 | layers=1 | fc_hidden=0
================================================================================
Epoch 001/60 | train loss: 2.4859 | val loss: 2.1159 | val acc: 0.4185 | val ppl: 8.30
Epoch 002/60 | train loss: 1.4768 | val loss: 1.8466 | val acc: 0.5175 | val ppl: 6.34
Epoch 003/60 | train loss: 0.8765 | val loss: 1.9609 | val acc: 0.5324 | val ppl: 7.11
Epoch 004/60 | train loss: 0.5201 | val loss: 2.1415 | val acc: 0.5312 | val ppl: 8.51
Epoch 005/60 | train loss: 0.3601 | val loss: 2.3439 | val acc: 0.5231 | val ppl: 10.42
Epoch 006/60 | train loss: 0.2916 | val loss: 2.4360 | val acc: 0.5285 | val ppl: 11.43
Epoch 007/60 | train loss: 0.2581 | val loss: 2.4989 | val acc: 0.5254 | val ppl: 12.17
Epoch 008/60 | train loss: 0.2394 | val loss: 2.5758 | val acc: 0.5282 | val ppl: 13.14
Epoch 009/60 | train loss: 0.2271 | val loss: 2.6111 | val acc: 0.5299 | val ppl: 13.61
Epoch 010/60 | train loss: 0.2184 | val loss: 2.6560 | val acc: 0.5317 | val ppl: 14.24
Epoch 011/60 | train loss: 0.2105 | val loss: 2.7004 | val acc: 0.5293 | val ppl: 14.89
Epoch 012/60 | train loss: 0.2069 | val loss: 2.7714 | val acc: 0.5299 | val ppl: 15.98
Epoch 013/60 | train loss: 0.2024 | val loss: 2.7914 | val acc: 0.5285 | val ppl: 16.30
Epoch 014/60 | train loss: 0.1994 | val loss: 2.7576 | val acc: 0.5290 | val ppl: 15.76
Epoch 015/60 | train loss: 0.1978 | val loss: 2.7632 | val acc: 0.5371 | val ppl: 15.85
Epoch 016/60 | train loss: 0.1951 | val loss: 2.7843 | val acc: 0.5408 | val ppl: 16.19
Epoch 017/60 | train loss: 0.1930 | val loss: 2.8562 | val acc: 0.5317 | val ppl: 17.39
Epoch 018/60 | train loss: 0.1916 | val loss: 2.8737 | val acc: 0.5288 | val ppl: 17.70
Epoch 019/60 | train loss: 0.1900 | val loss: 2.9079 | val acc: 0.5365 | val ppl: 18.32
Epoch 020/60 | train loss: 0.1881 | val loss: 2.9222 | val acc: 0.5343 | val ppl: 18.58
Epoch 021/60 | train loss: 0.1874 | val loss: 2.9328 | val acc: 0.5413 | val ppl: 18.78
Epoch 022/60 | train loss: 0.1890 | val loss: 2.9182 | val acc: 0.5328 | val ppl: 18.51
Epoch 023/60 | train loss: 0.1862 | val loss: 2.9567 | val acc: 0.5433 | val ppl: 19.23
Epoch 024/60 | train loss: 0.1852 | val loss: 2.9803 | val acc: 0.5400 | val ppl: 19.69
Epoch 025/60 | train loss: 0.1842 | val loss: 2.9869 | val acc: 0.5390 | val ppl: 19.82
Epoch 026/60 | train loss: 0.1840 | val loss: 2.9701 | val acc: 0.5368 | val ppl: 19.49
Epoch 027/60 | train loss: 0.1830 | val loss: 2.9827 | val acc: 0.5396 | val ppl: 19.74
Epoch 028/60 | train loss: 0.1821 | val loss: 3.0185 | val acc: 0.5275 | val ppl: 20.46
Epoch 029/60 | train loss: 0.1815 | val loss: 3.0559 | val acc: 0.5410 | val ppl: 21.24
Epoch 030/60 | train loss: 0.1809 | val loss: 3.0739 | val acc: 0.5310 | val ppl: 21.63
Epoch 031/60 | train loss: 0.1799 | val loss: 3.0379 | val acc: 0.5467 | val ppl: 20.86
Epoch 032/60 | train loss: 0.1806 | val loss: 3.0630 | val acc: 0.5344 | val ppl: 21.39
Epoch 033/60 | train loss: 0.1795 | val loss: 3.1069 | val acc: 0.5304 | val ppl: 22.35
Epoch 034/60 | train loss: 0.1796 | val loss: 3.1117 | val acc: 0.5350 | val ppl: 22.46
Epoch 035/60 | train loss: 0.1791 | val loss: 3.0996 | val acc: 0.5332 | val ppl: 22.19
Epoch 036/60 | train loss: 0.1780 | val loss: 3.1126 | val acc: 0.5410 | val ppl: 22.48
Epoch 037/60 | train loss: 0.1779 | val loss: 3.1022 | val acc: 0.5383 | val ppl: 22.25
Epoch 038/60 | train loss: 0.1783 | val loss: 3.0898 | val acc: 0.5383 | val ppl: 21.97
Epoch 039/60 | train loss: 0.1783 | val loss: 3.1550 | val acc: 0.5308 | val ppl: 23.45
Epoch 040/60 | train loss: 0.1774 | val loss: 3.1245 | val acc: 0.5321 | val ppl: 22.75
Epoch 041/60 | train loss: 0.1770 | val loss: 3.1638 | val acc: 0.5337 | val ppl: 23.66
Epoch 042/60 | train loss: 0.1767 | val loss: 3.1678 | val acc: 0.5368 | val ppl: 23.75
Epoch 043/60 | train loss: 0.1761 | val loss: 3.1901 | val acc: 0.5288 | val ppl: 24.29
Epoch 044/60 | train loss: 0.1756 | val loss: 3.1477 | val acc: 0.5279 | val ppl: 23.28
Epoch 045/60 | train loss: 0.1762 | val loss: 3.1721 | val acc: 0.5262 | val ppl: 23.86
Epoch 046/60 | train loss: 0.1754 | val loss: 3.1969 | val acc: 0.5365 | val ppl: 24.46
Epoch 047/60 | train loss: 0.1752 | val loss: 3.1987 | val acc: 0.5318 | val ppl: 24.50
Epoch 048/60 | train loss: 0.1753 | val loss: 3.2099 | val acc: 0.5303 | val ppl: 24.78
Epoch 049/60 | train loss: 0.1751 | val loss: 3.1564 | val acc: 0.5458 | val ppl: 23.49
Epoch 050/60 | train loss: 0.1751 | val loss: 3.2085 | val acc: 0.5371 | val ppl: 24.74
Epoch 051/60 | train loss: 0.1745 | val loss: 3.2120 | val acc: 0.5262 | val ppl: 24.83
Epoch 052/60 | train loss: 0.1741 | val loss: 3.2097 | val acc: 0.5321 | val ppl: 24.77
Epoch 053/60 | train loss: 0.1744 | val loss: 3.2146 | val acc: 0.5324 | val ppl: 24.89
Epoch 054/60 | train loss: 0.1746 | val loss: 3.2231 | val acc: 0.5300 | val ppl: 25.10
Epoch 055/60 | train loss: 0.1738 | val loss: 3.2207 | val acc: 0.5356 | val ppl: 25.04
Epoch 056/60 | train loss: 0.1731 | val loss: 3.2570 | val acc: 0.5396 | val ppl: 25.97
Epoch 057/60 | train loss: 0.1733 | val loss: 3.2484 | val acc: 0.5310 | val ppl: 25.75
Epoch 058/60 | train loss: 0.1731 | val loss: 3.2463 | val acc: 0.5290 | val ppl: 25.69
Epoch 059/60 | train loss: 0.1732 | val loss: 3.2318 | val acc: 0.5367 | val ppl: 25.32
Epoch 060/60 | train loss: 0.1730 | val loss: 3.2870 | val acc: 0.5293 | val ppl: 26.76

Generated sample:
Next character prediction involves the use of Recurrent Neural Networks (RNNs), and more specifically, a variant called Long Short-Term Memory (LSTM) networks. RNNs are particularly well-suited for sequential data like text, as they can maintain information in 'memory' about
```

```text
         dataset model_type  sequence_length  embedding_size  hidden_size  \
0  Provided Text        GRU               10              64          128   
1  Provided Text        GRU               20              64          128   
2  Provided Text        GRU               30              64          128   

   num_layers  fc_hidden_size  final_train_loss  final_val_loss  \
0           1               0          0.490221        3.064150   
1           1               0          0.255412        3.059160   
2           1               0          0.173011        3.287042   

   final_val_accuracy  final_val_perplexity  training_time_sec  \
0            0.504583             21.416240           3.942943   
1            0.556250             21.309644           3.921281   
2            0.529306             26.763581           3.926763   

   inference_time_sec  trainable_parameters  model_size_mb  \
0            0.066295                 83181        0.31731   
1            0.065132                 83181        0.31731   
2            0.065575                 83181        0.31731   

   approx_ops_per_sequence  
0                   794880  
1                  1589760  
2                  2384640  
```

---

## Cell 10 — Code

```python
# ============================================================
# 8B.Problem 3 - GRU on Tiny Shakespeare
# ============================================================

if "tiny_shakespeare_text" not in globals():
    tiny_shakespeare_text = load_tiny_shakespeare(max_chars=SHAKESPEARE_MAX_CHARS)



gru_shakespeare_configs = [
    {
        "name": "GRU_seq20_base",
        "seq_len": 20,
        "embed_size": 64,
        "hidden_size": 128,
        "num_layers": 1,
        "fc_hidden_size": 0,
        "dropout": 0.0,
    },
    {
        "name": "GRU_seq30_base",
        "seq_len": 30,
        "embed_size": 64,
        "hidden_size": 128,
        "num_layers": 1,
        "fc_hidden_size": 0,
        "dropout": 0.0,
    },
    {
        "name": "GRU_seq50_base",
        "seq_len": 50,
        "embed_size": 64,
        "hidden_size": 128,
        "num_layers": 1,
        "fc_hidden_size": 0,
        "dropout": 0.0,
    },
    {
        "name": "GRU_seq30_smaller_hidden",
        "seq_len": 30,
        "embed_size": 64,
        "hidden_size": 64,
        "num_layers": 1,
        "fc_hidden_size": 0,
        "dropout": 0.0,
    },
    {
        "name": "GRU_seq30_two_layers",
        "seq_len": 30,
        "embed_size": 64,
        "hidden_size": 128,
        "num_layers": 2,
        "fc_hidden_size": 0,
        "dropout": 0.2,
    },
    {
        "name": "GRU_seq30_fc_head",
        "seq_len": 30,
        "embed_size": 64,
        "hidden_size": 128,
        "num_layers": 1,
        "fc_hidden_size": 128,
        "dropout": 0.0,
    },
]

p3_gru_shakespeare_results = []
p3_gru_shakespeare_histories = []
p3_gru_shakespeare_samples = []

for cfg in gru_shakespeare_configs:
    print_section(cfg["name"])

    model, result_df, history_df, sample = run_single_experiment(
        text=tiny_shakespeare_text,
        dataset_name="Tiny Shakespeare",
        model_type="GRU",
        seq_len=cfg["seq_len"],
        epochs=SHAKESPEARE_EPOCHS,
        batch_size=SHAKESPEARE_BATCH_SIZE,
        embed_size=cfg["embed_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        fc_hidden_size=cfg["fc_hidden_size"],
        dropout=cfg["dropout"],
        learning_rate=LEARNING_RATE,
        grad_clip=GRAD_CLIP,
        generate_chars=GENERATE_CHARS_SHAKESPEARE,
        start_text="ROMEO:",
    )

    result_df["config_description"] = cfg["name"]
    history_df["config_description"] = cfg["name"]
    sample["config_description"] = cfg["name"]

    p3_gru_shakespeare_results.append(result_df)
    p3_gru_shakespeare_histories.append(history_df)
    p3_gru_shakespeare_samples.append(sample)

p3_gru_shakespeare_results_df = pd.concat(p3_gru_shakespeare_results, ignore_index=True)
p3_gru_shakespeare_history_df = pd.concat(p3_gru_shakespeare_histories, ignore_index=True)

save_results_csv(p3_gru_shakespeare_results_df, "problem3_gru_tiny_shakespeare_results.csv")
save_results_csv(p3_gru_shakespeare_history_df, "problem3_gru_tiny_shakespeare_history.csv")

display_clean_results(p3_gru_shakespeare_results_df)
```

### Output

### GRU_seq20_base

**stdout:**

```text
================================================================================
Dataset: Tiny Shakespeare
Model: GRU | seq_len=20 | embed=64 | hidden=128 | layers=1 | fc_hidden=0
================================================================================
Epoch 001/10 | train loss: 1.6355 | val loss: 1.6995 | val acc: 0.4967 | val ppl: 5.47
Epoch 002/10 | train loss: 1.4976 | val loss: 1.6841 | val acc: 0.5010 | val ppl: 5.39
Epoch 003/10 | train loss: 1.4772 | val loss: 1.6790 | val acc: 0.5049 | val ppl: 5.36
Epoch 004/10 | train loss: 1.4670 | val loss: 1.6743 | val acc: 0.5067 | val ppl: 5.33
Epoch 005/10 | train loss: 1.4607 | val loss: 1.6795 | val acc: 0.5043 | val ppl: 5.36
Epoch 006/10 | train loss: 1.4562 | val loss: 1.6743 | val acc: 0.5093 | val ppl: 5.33
Epoch 007/10 | train loss: 1.4527 | val loss: 1.6780 | val acc: 0.5071 | val ppl: 5.35
Epoch 008/10 | train loss: 1.4500 | val loss: 1.6752 | val acc: 0.5065 | val ppl: 5.34
Epoch 009/10 | train loss: 1.4476 | val loss: 1.6772 | val acc: 0.5071 | val ppl: 5.35
Epoch 010/10 | train loss: 1.4458 | val loss: 1.6779 | val acc: 0.5076 | val ppl: 5.35

Generated sample:
ROMEO:
Therefore idly then.

CAPULET:
A grant:
And therefore, nor so, sir, and he beseech you avove
And now, the gods to be so, you for you comes. Go hither with Caius will went become to colour with the strifest tale to save his purpose; now us.

FRIAR LAURENCE:
Go in breath, he would was the way,
Till I all the loath that the rage to lady!

ELBOW:
If your unroot, when the farewell: not of his brone of God!

Second Mortion:
Standon my life,
From the servant: reply how so the nack, and now them, for t
```

### GRU_seq30_base

**stdout:**

```text
================================================================================
Dataset: Tiny Shakespeare
Model: GRU | seq_len=30 | embed=64 | hidden=128 | layers=1 | fc_hidden=0
================================================================================
Epoch 001/10 | train loss: 1.5801 | val loss: 1.6645 | val acc: 0.5113 | val ppl: 5.28
Epoch 002/10 | train loss: 1.4423 | val loss: 1.6525 | val acc: 0.5143 | val ppl: 5.22
Epoch 003/10 | train loss: 1.4226 | val loss: 1.6494 | val acc: 0.5176 | val ppl: 5.20
Epoch 004/10 | train loss: 1.4126 | val loss: 1.6501 | val acc: 0.5159 | val ppl: 5.21
Epoch 005/10 | train loss: 1.4063 | val loss: 1.6487 | val acc: 0.5163 | val ppl: 5.20
Epoch 006/10 | train loss: 1.4017 | val loss: 1.6530 | val acc: 0.5165 | val ppl: 5.22
Epoch 007/10 | train loss: 1.3985 | val loss: 1.6536 | val acc: 0.5191 | val ppl: 5.23
Epoch 008/10 | train loss: 1.3959 | val loss: 1.6585 | val acc: 0.5173 | val ppl: 5.25
Epoch 009/10 | train loss: 1.3939 | val loss: 1.6580 | val acc: 0.5184 | val ppl: 5.25
Epoch 010/10 | train loss: 1.3922 | val loss: 1.6605 | val acc: 0.5173 | val ppl: 5.26

Generated sample:
ROMEO:
No more greathed with thy such a dispant this coronable the night me sent to question
Even any imity,
Why all thy wits for a lady pent to your prince!
Come, sir! he was here, my silk by the name me,
With the chamber,
There is treason as respect and to my brother is call my sister's way to so: but they that would you have partly past there, be contract you, sir; i' thy moners can see it did all the morning his homers that love of you show you
As I show.

LADY ANNE:
Ay, the king.

LADY ANNE:
Go b
```

### GRU_seq50_base

**stdout:**

```text
================================================================================
Dataset: Tiny Shakespeare
Model: GRU | seq_len=50 | embed=64 | hidden=128 | layers=1 | fc_hidden=0
================================================================================
Epoch 001/10 | train loss: 1.5233 | val loss: 1.6298 | val acc: 0.5225 | val ppl: 5.10
Epoch 002/10 | train loss: 1.3829 | val loss: 1.6213 | val acc: 0.5280 | val ppl: 5.06
Epoch 003/10 | train loss: 1.3646 | val loss: 1.6313 | val acc: 0.5241 | val ppl: 5.11
Epoch 004/10 | train loss: 1.3555 | val loss: 1.6281 | val acc: 0.5294 | val ppl: 5.09
Epoch 005/10 | train loss: 1.3497 | val loss: 1.6322 | val acc: 0.5294 | val ppl: 5.12
Epoch 006/10 | train loss: 1.3457 | val loss: 1.6347 | val acc: 0.5279 | val ppl: 5.13
Epoch 007/10 | train loss: 1.3426 | val loss: 1.6372 | val acc: 0.5303 | val ppl: 5.14
Epoch 008/10 | train loss: 1.3403 | val loss: 1.6424 | val acc: 0.5278 | val ppl: 5.17
Epoch 009/10 | train loss: 1.3384 | val loss: 1.6418 | val acc: 0.5293 | val ppl: 5.16
Epoch 010/10 | train loss: 1.3368 | val loss: 1.6504 | val acc: 0.5278 | val ppl: 5.21

Generated sample:
ROMEO:
Are the prison: if have a saint:
I know it must not come too lord, you have said;
And there is a lord, is Catesby:
I thank us an hour and lie again,
And she had no more fortune for Rome,
And had she bose prepared with strew herself:
You have not previct foul't:
An usants doft and the Christian tots having the manners
And hold the people, more, madam.
See for heaven death for the crown
With that batting, good myst in night should that we trone.
Thou art fair brother, and stir: take before me but
```

### GRU_seq30_smaller_hidden

**stdout:**

```text
================================================================================
Dataset: Tiny Shakespeare
Model: GRU | seq_len=30 | embed=64 | hidden=64 | layers=1 | fc_hidden=0
================================================================================
Epoch 001/10 | train loss: 1.7518 | val loss: 1.7968 | val acc: 0.4765 | val ppl: 6.03
Epoch 002/10 | train loss: 1.6086 | val loss: 1.7851 | val acc: 0.4826 | val ppl: 5.96
Epoch 003/10 | train loss: 1.5905 | val loss: 1.7839 | val acc: 0.4836 | val ppl: 5.95
Epoch 004/10 | train loss: 1.5812 | val loss: 1.7782 | val acc: 0.4863 | val ppl: 5.92
Epoch 005/10 | train loss: 1.5751 | val loss: 1.7789 | val acc: 0.4871 | val ppl: 5.92
Epoch 006/10 | train loss: 1.5706 | val loss: 1.7820 | val acc: 0.4858 | val ppl: 5.94
Epoch 007/10 | train loss: 1.5674 | val loss: 1.7820 | val acc: 0.4881 | val ppl: 5.94
Epoch 008/10 | train loss: 1.5646 | val loss: 1.7803 | val acc: 0.4884 | val ppl: 5.93
Epoch 009/10 | train loss: 1.5623 | val loss: 1.7844 | val acc: 0.4870 | val ppl: 5.96
Epoch 010/10 | train loss: 1.5603 | val loss: 1.7840 | val acc: 0.4862 | val ppl: 5.95

Generated sample:
ROMEO:
I metve, hear the
The honour be speak take it your gifery comfort staps,
And both.

QUEEN MARGARET:
All parts of thee of the send our out me to it, this whose down:
For the power,
Our slander heaven'd on wash hath chaicher same the pursess aip peace, but breakes with here of this gold, what on the eyous natry,
Thrust I would else the son and her, and the dour them stand beat ready the heaven gentleman:
So my twift, and sight you to in their last the dencuted might in brated most;
And comous Pea
```

### GRU_seq30_two_layers

**stdout:**

```text
================================================================================
Dataset: Tiny Shakespeare
Model: GRU | seq_len=30 | embed=64 | hidden=128 | layers=2 | fc_hidden=0
================================================================================
Epoch 001/10 | train loss: 1.5817 | val loss: 1.5978 | val acc: 0.5205 | val ppl: 4.94
Epoch 002/10 | train loss: 1.4523 | val loss: 1.5850 | val acc: 0.5269 | val ppl: 4.88
Epoch 003/10 | train loss: 1.4339 | val loss: 1.5830 | val acc: 0.5273 | val ppl: 4.87
Epoch 004/10 | train loss: 1.4244 | val loss: 1.5826 | val acc: 0.5283 | val ppl: 4.87
Epoch 005/10 | train loss: 1.4183 | val loss: 1.5822 | val acc: 0.5275 | val ppl: 4.87
Epoch 006/10 | train loss: 1.4139 | val loss: 1.5825 | val acc: 0.5283 | val ppl: 4.87
Epoch 007/10 | train loss: 1.4106 | val loss: 1.5800 | val acc: 0.5290 | val ppl: 4.86
Epoch 008/10 | train loss: 1.4079 | val loss: 1.5850 | val acc: 0.5281 | val ppl: 4.88
Epoch 009/10 | train loss: 1.4057 | val loss: 1.5807 | val acc: 0.5297 | val ppl: 4.86
Epoch 010/10 | train loss: 1.4038 | val loss: 1.5800 | val acc: 0.5285 | val ppl: 4.85

Generated sample:
ROMEO:
O magly, may a king, cut: she is he make ye to the charged of fair mother's dead,
Detrellex; or I prive for the Bohemis to death of this and shall go spread to proud.

KING RICHARD III:
Marry, my lord.

KING RICHARD III:
Will you hear thee? and how my love,
Musician that thy poor Bohemia.

MENENIUS:
I think, and therefore, let's hear our brother Edward, as he the wing.
Deck, what are the land, it were it in his grave,
In my friendly had been to the people. I
am an Romeo should bables the crown 
```

### GRU_seq30_fc_head

**stdout:**

```text
================================================================================
Dataset: Tiny Shakespeare
Model: GRU | seq_len=30 | embed=64 | hidden=128 | layers=1 | fc_hidden=128
================================================================================
Epoch 001/10 | train loss: 1.5357 | val loss: 1.6642 | val acc: 0.5170 | val ppl: 5.28
Epoch 002/10 | train loss: 1.3958 | val loss: 1.6740 | val acc: 0.5208 | val ppl: 5.33
Epoch 003/10 | train loss: 1.3733 | val loss: 1.6785 | val acc: 0.5222 | val ppl: 5.36
Epoch 004/10 | train loss: 1.3618 | val loss: 1.6834 | val acc: 0.5242 | val ppl: 5.38
Epoch 005/10 | train loss: 1.3545 | val loss: 1.6905 | val acc: 0.5256 | val ppl: 5.42
Epoch 006/10 | train loss: 1.3493 | val loss: 1.7000 | val acc: 0.5211 | val ppl: 5.47
Epoch 007/10 | train loss: 1.3455 | val loss: 1.7001 | val acc: 0.5249 | val ppl: 5.47
Epoch 008/10 | train loss: 1.3426 | val loss: 1.7009 | val acc: 0.5246 | val ppl: 5.48
Epoch 009/10 | train loss: 1.3403 | val loss: 1.6990 | val acc: 0.5256 | val ppl: 5.47
Epoch 010/10 | train loss: 1.3382 | val loss: 1.7070 | val acc: 0.5250 | val ppl: 5.51

Generated sample:
ROMEO:
Why, sir, my lord, as you speak and king of princess
To be come.

QUEEN MARGARET:
Thanks to-morrow by you sprave;
Or any ship; forbear will make thee no man? what had so farewell.

LUCIO:
Villain, to deaths. Additors were a father's like a time
That a world's grace me heaven for the fires their name:
To Romeo had he resign him.

JULIET:
So seem rhther from his life as it is fardel the nation comes would prove upon this bloody well-disprising?

ROMEO:
So haste hath been years that I will dispatc
```

```text
            dataset model_type  sequence_length  embedding_size  hidden_size  \
0  Tiny Shakespeare        GRU               20              64          128   
3  Tiny Shakespeare        GRU               30              64           64   
1  Tiny Shakespeare        GRU               30              64          128   
5  Tiny Shakespeare        GRU               30              64          128   
4  Tiny Shakespeare        GRU               30              64          128   
2  Tiny Shakespeare        GRU               50              64          128   

   num_layers  fc_hidden_size  final_train_loss  final_val_loss  \
0           1               0          1.445822        1.677889   
3           1               0          1.560297        1.783960   
1           1               0          1.392165        1.660514   
5           1             128          1.338169        1.707020   
4           2               0          1.403830        1.579962   
2           1               0          1.336810        1.650375   

   final_val_accuracy  final_val_perplexity  training_time_sec  \
0            0.507574              5.354240          69.933018   
3            0.486235              5.953384          69.581094   
1            0.517280              5.262015          69.732312   
5            0.525007              5.512509          73.388258   
4            0.528453              4.854773          76.103962   
2            0.527821              5.208931          70.854570   

   inference_time_sec  trainable_parameters  model_size_mb  \
0            0.164197                 87041       0.332035   
3            0.135900                 33345       0.127201   
1            0.162349                 87041       0.332035   
5            0.171707                103553       0.395023   
4            0.248079                186113       0.709965   
2            0.162233                 87041       0.332035   

   approx_ops_per_sequence  
0                  1640960  
3                   862080  
1                  2461440  
5                  2952960  
4                  5410560  
2                  4102400  
```

---

## Cell 11 — Code `In [20]`

```python
# ============================================================
# 9. Problem 3 plots and results
# ============================================================

print_section("GRU provided text result table")
display_clean_results(p3_gru_fixed_results_df)

print_section("GRU provided text training loss")
plot_training_curves(
    p3_gru_fixed_histories,
    title="GRU Training Loss - Provided Text",
    metric="train_loss",
    save_name="gru_provided_training_loss.png",
)

print_section("GRU provided text validation accuracy")
plot_training_curves(
    p3_gru_fixed_histories,
    title="GRU Validation Accuracy - Provided Text",
    metric="val_accuracy",
    save_name="gru_provided_validation_accuracy.png",
)

print_section("GRU generated samples on provided text")
show_samples(p3_gru_fixed_samples)

print_section("GRU Tiny Shakespeare result table")
display_clean_results(p3_gru_shakespeare_results_df)

print_section("GRU Tiny Shakespeare training loss")
plot_training_curves(
    p3_gru_shakespeare_histories,
    title="GRU Training Loss - Tiny Shakespeare",
    metric="train_loss",
    save_name="gru_shakespeare_training_loss.png",
)

print_section("GRU Tiny Shakespeare validation loss")
plot_training_curves(
    p3_gru_shakespeare_histories,
    title="GRU Validation Loss - Tiny Shakespeare",
    metric="val_loss",
    save_name="gru_shakespeare_validation_loss.png",
)

print_section("GRU Tiny Shakespeare validation accuracy")
plot_training_curves(
    p3_gru_shakespeare_histories,
    title="GRU Validation Accuracy - Tiny Shakespeare",
    metric="val_accuracy",
    save_name="gru_shakespeare_validation_accuracy.png",
)

print_section("GRU Tiny Shakespeare perplexity")
plot_training_curves(
    p3_gru_shakespeare_histories,
    title="GRU Validation Perplexity - Tiny Shakespeare",
    metric="val_perplexity",
    save_name="gru_shakespeare_perplexity.png",
)

print_section("GRU Tiny Shakespeare model size")
plot_result_bar(
    p3_gru_shakespeare_results_df,
    x_col="sequence_length",
    y_col="model_size_mb",
    group_col="config_description",
    title="GRU Model Size - Tiny Shakespeare",
    save_name="gru_shakespeare_model_size.png",
)

print_section("GRU Tiny Shakespeare training time")
plot_result_bar(
    p3_gru_shakespeare_results_df,
    x_col="sequence_length",
    y_col="training_time_sec",
    group_col="config_description",
    title="GRU Training Time - Tiny Shakespeare",
    save_name="gru_shakespeare_training_time.png",
)

print_section("GRU generated Shakespeare samples")
show_samples(p3_gru_shakespeare_samples)
```

### Output

### GRU provided text result table

```text
         dataset model_type  sequence_length  embedding_size  hidden_size  \
0  Provided Text        GRU               10              64          128   
1  Provided Text        GRU               20              64          128   
2  Provided Text        GRU               30              64          128   

   num_layers  fc_hidden_size  final_train_loss  final_val_loss  \
0           1               0          0.490221        3.064150   
1           1               0          0.255412        3.059160   
2           1               0          0.173011        3.287042   

   final_val_accuracy  final_val_perplexity  training_time_sec  \
0            0.504583             21.416240           3.942943   
1            0.556250             21.309644           3.921281   
2            0.529306             26.763581           3.926763   

   inference_time_sec  trainable_parameters  model_size_mb  \
0            0.066295                 83181        0.31731   
1            0.065132                 83181        0.31731   
2            0.065575                 83181        0.31731   

   approx_ops_per_sequence  
0                   794880  
1                  1589760  
2                  2384640  
```

### GRU provided text training loss

![Cell 11 output 4](README_assets/cell_11_output_04_013.png)

### GRU provided text validation accuracy

![Cell 11 output 6](README_assets/cell_11_output_06_014.png)

### GRU generated samples on provided text

#### Provided_Text_GRU_seq10_h128_layers1_fc0

```text
Next character predictions and the actual outcomes, thus improving its prediction of the model.

One of the most popular approaches to next character prediction involves predictive accuracy over time.

Once trained, the model adjusts its parameters to minimize the difference
```

#### Provided_Text_GRU_seq20_h128_layers1_fc0

```text
Next character prediction involves feeding it large amounts of text data, allowing it to learn the prediction of the next character prediction involves feeding it large amounts of text data, allowing it to learn the probability of each character's appearance following a sequ
```

#### Provided_Text_GRU_seq30_h128_layers1_fc0

```text
Next character prediction involves the use of Recurrent Neural Networks (RNNs), and more specifically, a variant called Long Short-Term Memory (LSTM) networks. RNNs are particularly well-suited for sequential data like text, as they can maintain information in 'memory' about
```

### GRU Tiny Shakespeare result table

```text
            dataset model_type  sequence_length  embedding_size  hidden_size  \
0  Tiny Shakespeare        GRU               20              64          128   
3  Tiny Shakespeare        GRU               30              64           64   
1  Tiny Shakespeare        GRU               30              64          128   
5  Tiny Shakespeare        GRU               30              64          128   
4  Tiny Shakespeare        GRU               30              64          128   
2  Tiny Shakespeare        GRU               50              64          128   

   num_layers  fc_hidden_size  final_train_loss  final_val_loss  \
0           1               0          1.445822        1.677889   
3           1               0          1.560297        1.783960   
1           1               0          1.392165        1.660514   
5           1             128          1.338169        1.707020   
4           2               0          1.403830        1.579962   
2           1               0          1.336810        1.650375   

   final_val_accuracy  final_val_perplexity  training_time_sec  \
0            0.507574              5.354240          69.933018   
3            0.486235              5.953384          69.581094   
1            0.517280              5.262015          69.732312   
5            0.525007              5.512509          73.388258   
4            0.528453              4.854773          76.103962   
2            0.527821              5.208931          70.854570   

   inference_time_sec  trainable_parameters  model_size_mb  \
0            0.164197                 87041       0.332035   
3            0.135900                 33345       0.127201   
1            0.162349                 87041       0.332035   
5            0.171707                103553       0.395023   
4            0.248079                186113       0.709965   
2            0.162233                 87041       0.332035   

   approx_ops_per_sequence  
0                  1640960  
3                   862080  
1                  2461440  
5                  2952960  
4                  5410560  
2                  4102400  
```

### GRU Tiny Shakespeare training loss

![Cell 11 output 14](README_assets/cell_11_output_14_015.png)

### GRU Tiny Shakespeare validation loss

![Cell 11 output 16](README_assets/cell_11_output_16_016.png)

### GRU Tiny Shakespeare validation accuracy

![Cell 11 output 18](README_assets/cell_11_output_18_017.png)

### GRU Tiny Shakespeare perplexity

![Cell 11 output 20](README_assets/cell_11_output_20_018.png)

### GRU Tiny Shakespeare model size

![Cell 11 output 22](README_assets/cell_11_output_22_019.png)

### GRU Tiny Shakespeare training time

![Cell 11 output 24](README_assets/cell_11_output_24_020.png)

### GRU generated Shakespeare samples

#### Tiny_Shakespeare_GRU_seq20_h128_layers1_fc0

```text
ROMEO:
Therefore idly then.

CAPULET:
A grant:
And therefore, nor so, sir, and he beseech you avove
And now, the gods to be so, you for you comes. Go hither with Caius will went become to colour with the strifest tale to save his purpose; now us.

FRIAR LAURENCE:
Go in breath, he would was the way,
Till I all the loath that the rage to lady!

ELBOW:
If your unroot, when the farewell: not of his brone of God!

Second Mortion:
Standon my life,
From the servant: reply how so the nack, and now them, for t
```

#### Tiny_Shakespeare_GRU_seq30_h128_layers1_fc0

```text
ROMEO:
No more greathed with thy such a dispant this coronable the night me sent to question
Even any imity,
Why all thy wits for a lady pent to your prince!
Come, sir! he was here, my silk by the name me,
With the chamber,
There is treason as respect and to my brother is call my sister's way to so: but they that would you have partly past there, be contract you, sir; i' thy moners can see it did all the morning his homers that love of you show you
As I show.

LADY ANNE:
Ay, the king.

LADY ANNE:
Go b
```

#### Tiny_Shakespeare_GRU_seq50_h128_layers1_fc0

```text
ROMEO:
Are the prison: if have a saint:
I know it must not come too lord, you have said;
And there is a lord, is Catesby:
I thank us an hour and lie again,
And she had no more fortune for Rome,
And had she bose prepared with strew herself:
You have not previct foul't:
An usants doft and the Christian tots having the manners
And hold the people, more, madam.
See for heaven death for the crown
With that batting, good myst in night should that we trone.
Thou art fair brother, and stir: take before me but
```

#### Tiny_Shakespeare_GRU_seq30_h64_layers1_fc0

```text
ROMEO:
I metve, hear the
The honour be speak take it your gifery comfort staps,
And both.

QUEEN MARGARET:
All parts of thee of the send our out me to it, this whose down:
For the power,
Our slander heaven'd on wash hath chaicher same the pursess aip peace, but breakes with here of this gold, what on the eyous natry,
Thrust I would else the son and her, and the dour them stand beat ready the heaven gentleman:
So my twift, and sight you to in their last the dencuted might in brated most;
And comous Pea
```

#### Tiny_Shakespeare_GRU_seq30_h128_layers2_fc0

```text
ROMEO:
O magly, may a king, cut: she is he make ye to the charged of fair mother's dead,
Detrellex; or I prive for the Bohemis to death of this and shall go spread to proud.

KING RICHARD III:
Marry, my lord.

KING RICHARD III:
Will you hear thee? and how my love,
Musician that thy poor Bohemia.

MENENIUS:
I think, and therefore, let's hear our brother Edward, as he the wing.
Deck, what are the land, it were it in his grave,
In my friendly had been to the people. I
am an Romeo should bables the crown 
```

#### Tiny_Shakespeare_GRU_seq30_h128_layers1_fc128

```text
ROMEO:
Why, sir, my lord, as you speak and king of princess
To be come.

QUEEN MARGARET:
Thanks to-morrow by you sprave;
Or any ship; forbear will make thee no man? what had so farewell.

LUCIO:
Villain, to deaths. Additors were a father's like a time
That a world's grace me heaven for the fires their name:
To Romeo had he resign him.

JULIET:
So seem rhther from his life as it is fardel the nation comes would prove upon this bloody well-disprising?

ROMEO:
So haste hath been years that I will dispatc
```

---

## Cell 12 — Code `In [19]`

```python
# ===========================================================
# 10. Final comparison and conclusion
# ============================================================

all_results_df = pd.concat(
    [
        p1_rnn_results_df,
        p2_lstm_fixed_results_df,
        p3_gru_fixed_results_df,
        p2_lstm_shakespeare_results_df,
        p3_gru_shakespeare_results_df,
    ],
    ignore_index=True
)

all_history_df = pd.concat(
    [
        p1_rnn_history_df,
        p2_lstm_fixed_history_df,
        p3_gru_fixed_history_df,
        p2_lstm_shakespeare_history_df,
        p3_gru_shakespeare_history_df,
    ],
    ignore_index=True
)

all_samples = (
    p1_rnn_samples
    + p2_lstm_fixed_samples
    + p3_gru_fixed_samples
    + p2_lstm_shakespeare_samples
    + p3_gru_shakespeare_samples
)

save_results_csv(all_results_df, "final_all_results.csv")
save_results_csv(all_history_df, "final_all_training_history.csv")

print_section("Final combined result table")
display_clean_results(all_results_df)

print_section("comparison: Provided Text, RNN vs LSTM vs GRU")
problem1_official_df = all_results_df[
    all_results_df["dataset"] == "Provided Text"
].copy()

display_clean_results(problem1_official_df)
save_results_csv(problem1_official_df, "_problem1_provided_text_results.csv")

print_section("comparison: Tiny Shakespeare, LSTM vs GRU")
problem2_official_df = all_results_df[
    all_results_df["dataset"] == "Tiny Shakespeare"
].copy()

display_clean_results(problem2_official_df)
save_results_csv(problem2_official_df, "official_problem2_tiny_shakespeare_results.csv")

print_section("validation accuracy comparison")
plot_result_bar(
    problem1_official_df,
    x_col="sequence_length",
    y_col="final_val_accuracy",
    group_col="model_type",
    title="Provided Text Validation Accuracy - RNN vs LSTM vs GRU",
    save_name="final_provided_text_validation_accuracy.png",
)

print_section("Final Problem 1 training loss comparison")
plot_result_bar(
    problem1_official_df,
    x_col="sequence_length",
    y_col="final_train_loss",
    group_col="model_type",
    title="Provided Text Final Training Loss - RNN vs LSTM vs GRU",
    save_name="final_provided_text_training_loss.png",
)

print_section("Final Problem 1 model size comparison")
plot_result_bar(
    problem1_official_df,
    x_col="sequence_length",
    y_col="model_size_mb",
    group_col="model_type",
    title="Provided Text Model Size - RNN vs LSTM vs GRU",
    save_name="final_provided_text_model_size.png",
)

print_section("Final Problem 1 training time comparison")
plot_result_bar(
    problem1_official_df,
    x_col="sequence_length",
    y_col="training_time_sec",
    group_col="model_type",
    title="Provided Text Training Time - RNN vs LSTM vs GRU",
    save_name="final_provided_text_training_time.png",
)

print_section("Final Tiny Shakespeare validation accuracy comparison")
plot_result_bar(
    problem2_official_df,
    x_col="sequence_length",
    y_col="final_val_accuracy",
    group_col="config_description",
    title="Tiny Shakespeare Validation Accuracy - LSTM vs GRU Configurations",
    save_name="final_shakespeare_validation_accuracy.png",
)

print_section("Final Tiny Shakespeare validation perplexity comparison")
plot_result_bar(
    problem2_official_df,
    x_col="sequence_length",
    y_col="final_val_perplexity",
    group_col="config_description",
    title="Tiny Shakespeare Validation Perplexity - LSTM vs GRU Configurations",
    save_name="final_shakespeare_perplexity.png",
)

print_section("Final Tiny Shakespeare model size comparison")
plot_result_bar(
    problem2_official_df,
    x_col="sequence_length",
    y_col="model_size_mb",
    group_col="config_description",
    title="Tiny Shakespeare Model Size - LSTM vs GRU Configurations",
    save_name="final_shakespeare_model_size.png",
)

print_section("Final Tiny Shakespeare training time comparison")
plot_result_bar(
    problem2_official_df,
    x_col="sequence_length",
    y_col="training_time_sec",
    group_col="config_description",
    title="Tiny Shakespeare Training Time - LSTM vs GRU Configurations",
    save_name="final_shakespeare_training_time.png",
)
```

### Output

### Final combined result table

```text
             dataset model_type  sequence_length  embedding_size  hidden_size  \
6      Provided Text        GRU               10              64          128   
7      Provided Text        GRU               20              64          128   
8      Provided Text        GRU               30              64          128   
3      Provided Text       LSTM               10              64          128   
4      Provided Text       LSTM               20              64          128   
5      Provided Text       LSTM               30              64          128   
0      Provided Text        RNN               10              64          128   
1      Provided Text        RNN               20              64          128   
2      Provided Text        RNN               30              64          128   
15  Tiny Shakespeare        GRU               20              64          128   
18  Tiny Shakespeare        GRU               30              64           64   
16  Tiny Shakespeare        GRU               30              64          128   
20  Tiny Shakespeare        GRU               30              64          128   
19  Tiny Shakespeare        GRU               30              64          128   
17  Tiny Shakespeare        GRU               50              64          128   
9   Tiny Shakespeare       LSTM               20              64          128   
12  Tiny Shakespeare       LSTM               30              64           64   
10  Tiny Shakespeare       LSTM               30              64          128   
14  Tiny Shakespeare       LSTM               30              64          128   
13  Tiny Shakespeare       LSTM               30              64          128   
11  Tiny Shakespeare       LSTM               50              64          128   

    num_layers  fc_hidden_size  final_train_loss  final_val_loss  \
6            1               0          0.490221        3.064150   
7            1               0          0.255412        3.059160   
8            1               0          0.173011        3.287042   
3            1               0          0.492500        2.806571   
4            1               0          0.257663        3.075028   
5            1               0          0.174311        3.144315   
0            1               0          0.519695        2.755376   
1            1               0          0.274615        2.910769   
2            1               0          0.186928        3.246676   
15           1               0          1.445822        1.677889   
18           1               0          1.560297        1.783960   
16           1               0          1.392165        1.660514   
20           1             128          1.338169        1.707020   
19           2               0          1.403830        1.579962   
17           1               0          1.336810        1.650375   
9            1               0          1.408550        1.673675   
12           1               0          1.512735        1.733700   
10           1               0          1.353119        1.675177   
14           1             128          1.304747        1.700598   
13           2               0          1.363384        1.582417   
11           1               0          1.294236        1.658390   

    final_val_accuracy  final_val_perplexity  training_time_sec  \
6             0.504583             21.416240           3.942943   
7             0.556250             21.309644           3.921281   
8             0.529306             26.763581           3.926763   
3             0.512500             16.553061           4.047806   
4             0.517292             21.650498           4.064672   
5             0.531944             23.203768           4.100124   
0             0.500417             15.726946           4.644834   
1             0.556250             18.370914           3.702814   
2             0.543333             25.704748           3.696915   
15            0.507574              5.354240          69.933018   
18            0.486235              5.953384          69.581094   
16            0.517280              5.262015          69.732312   
20            0.525007              5.512509          73.388258   
19            0.528453              4.854773          76.103962   
17            0.527821              5.208931          70.854570   
9             0.512472              5.331725          71.172834   
12            0.499253              5.661564          70.851092   
10            0.522989              5.339740          72.508646   
14            0.524040              5.477220          75.469431   
13            0.533166              4.866706          88.641188   
11            0.529108              5.250850          80.596546   

    inference_time_sec  trainable_parameters  model_size_mb  \
6             0.066295                 83181       0.317310   
7             0.065132                 83181       0.317310   
8             0.065575                 83181       0.317310   
3             0.126669                108013       0.412037   
4             0.125014                108013       0.412037   
5             0.125857                108013       0.412037   
0             0.174232                 33517       0.127857   
1             0.054844                 33517       0.127857   
2             0.054806                 33517       0.127857   
15            0.164197                 87041       0.332035   
18            0.135900                 33345       0.127201   
16            0.162349                 87041       0.332035   
20            0.171707                103553       0.395023   
19            0.248079                186113       0.709965   
17            0.162233                 87041       0.332035   
9             0.432840                111873       0.426762   
12            0.151742                 41665       0.158939   
10            0.367741                111873       0.426762   
14            0.376018                128385       0.489750   
13            0.639143                243969       0.930668   
11            0.366469                111873       0.426762   

    approx_ops_per_sequence  
6                    794880  
7                   1589760  
8                   2384640  
3                   1040640  
4                   2081280  
5                   3121920  
0                    303360  
1                    606720  
2                    910080  
15                  1640960  
18                   862080  
16                  2461440  
20                  2952960  
19                  5410560  
17                  4102400  
9                   2132480  
12                  1107840  
10                  3198720  
14                  3690240  
13                  7130880  
11                  5331200  
```

### comparison: Provided Text, RNN vs LSTM vs GRU

```text
         dataset model_type  sequence_length  embedding_size  hidden_size  \
6  Provided Text        GRU               10              64          128   
7  Provided Text        GRU               20              64          128   
8  Provided Text        GRU               30              64          128   
3  Provided Text       LSTM               10              64          128   
4  Provided Text       LSTM               20              64          128   
5  Provided Text       LSTM               30              64          128   
0  Provided Text        RNN               10              64          128   
1  Provided Text        RNN               20              64          128   
2  Provided Text        RNN               30              64          128   

   num_layers  fc_hidden_size  final_train_loss  final_val_loss  \
6           1               0          0.490221        3.064150   
7           1               0          0.255412        3.059160   
8           1               0          0.173011        3.287042   
3           1               0          0.492500        2.806571   
4           1               0          0.257663        3.075028   
5           1               0          0.174311        3.144315   
0           1               0          0.519695        2.755376   
1           1               0          0.274615        2.910769   
2           1               0          0.186928        3.246676   

   final_val_accuracy  final_val_perplexity  training_time_sec  \
6            0.504583             21.416240           3.942943   
7            0.556250             21.309644           3.921281   
8            0.529306             26.763581           3.926763   
3            0.512500             16.553061           4.047806   
4            0.517292             21.650498           4.064672   
5            0.531944             23.203768           4.100124   
0            0.500417             15.726946           4.644834   
1            0.556250             18.370914           3.702814   
2            0.543333             25.704748           3.696915   

   inference_time_sec  trainable_parameters  model_size_mb  \
6            0.066295                 83181       0.317310   
7            0.065132                 83181       0.317310   
8            0.065575                 83181       0.317310   
3            0.126669                108013       0.412037   
4            0.125014                108013       0.412037   
5            0.125857                108013       0.412037   
0            0.174232                 33517       0.127857   
1            0.054844                 33517       0.127857   
2            0.054806                 33517       0.127857   

   approx_ops_per_sequence  
6                   794880  
7                  1589760  
8                  2384640  
3                  1040640  
4                  2081280  
5                  3121920  
0                   303360  
1                   606720  
2                   910080  
```

### comparison: Tiny Shakespeare, LSTM vs GRU

```text
             dataset model_type  sequence_length  embedding_size  hidden_size  \
15  Tiny Shakespeare        GRU               20              64          128   
18  Tiny Shakespeare        GRU               30              64           64   
16  Tiny Shakespeare        GRU               30              64          128   
20  Tiny Shakespeare        GRU               30              64          128   
19  Tiny Shakespeare        GRU               30              64          128   
17  Tiny Shakespeare        GRU               50              64          128   
9   Tiny Shakespeare       LSTM               20              64          128   
12  Tiny Shakespeare       LSTM               30              64           64   
10  Tiny Shakespeare       LSTM               30              64          128   
14  Tiny Shakespeare       LSTM               30              64          128   
13  Tiny Shakespeare       LSTM               30              64          128   
11  Tiny Shakespeare       LSTM               50              64          128   

    num_layers  fc_hidden_size  final_train_loss  final_val_loss  \
15           1               0          1.445822        1.677889   
18           1               0          1.560297        1.783960   
16           1               0          1.392165        1.660514   
20           1             128          1.338169        1.707020   
19           2               0          1.403830        1.579962   
17           1               0          1.336810        1.650375   
9            1               0          1.408550        1.673675   
12           1               0          1.512735        1.733700   
10           1               0          1.353119        1.675177   
14           1             128          1.304747        1.700598   
13           2               0          1.363384        1.582417   
11           1               0          1.294236        1.658390   

    final_val_accuracy  final_val_perplexity  training_time_sec  \
15            0.507574              5.354240          69.933018   
18            0.486235              5.953384          69.581094   
16            0.517280              5.262015          69.732312   
20            0.525007              5.512509          73.388258   
19            0.528453              4.854773          76.103962   
17            0.527821              5.208931          70.854570   
9             0.512472              5.331725          71.172834   
12            0.499253              5.661564          70.851092   
10            0.522989              5.339740          72.508646   
14            0.524040              5.477220          75.469431   
13            0.533166              4.866706          88.641188   
11            0.529108              5.250850          80.596546   

    inference_time_sec  trainable_parameters  model_size_mb  \
15            0.164197                 87041       0.332035   
18            0.135900                 33345       0.127201   
16            0.162349                 87041       0.332035   
20            0.171707                103553       0.395023   
19            0.248079                186113       0.709965   
17            0.162233                 87041       0.332035   
9             0.432840                111873       0.426762   
12            0.151742                 41665       0.158939   
10            0.367741                111873       0.426762   
14            0.376018                128385       0.489750   
13            0.639143                243969       0.930668   
11            0.366469                111873       0.426762   

    approx_ops_per_sequence  
15                  1640960  
18                   862080  
16                  2461440  
20                  2952960  
19                  5410560  
17                  4102400  
9                   2132480  
12                  1107840  
10                  3198720  
14                  3690240  
13                  7130880  
11                  5331200  
```

### validation accuracy comparison

![Cell 12 output 8](README_assets/cell_12_output_08_021.png)

### Final Problem 1 training loss comparison

![Cell 12 output 10](README_assets/cell_12_output_10_022.png)

### Final Problem 1 model size comparison

![Cell 12 output 12](README_assets/cell_12_output_12_023.png)

### Final Problem 1 training time comparison

![Cell 12 output 14](README_assets/cell_12_output_14_024.png)

### Final Tiny Shakespeare validation accuracy comparison

![Cell 12 output 16](README_assets/cell_12_output_16_025.png)

### Final Tiny Shakespeare validation perplexity comparison

![Cell 12 output 18](README_assets/cell_12_output_18_026.png)

### Final Tiny Shakespeare model size comparison

![Cell 12 output 20](README_assets/cell_12_output_20_027.png)

### Final Tiny Shakespeare training time comparison

![Cell 12 output 22](README_assets/cell_12_output_22_028.png)

---

