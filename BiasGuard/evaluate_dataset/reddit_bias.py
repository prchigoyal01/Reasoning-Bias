# evaluate_toxicity.py

import numpy as np
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. MODEL WRAPPER
# -------------------------------------------------------------

class HFModelWrapper:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def predict(self, texts, threshold=0.5):
        enc = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        logits = self.model(**enc).logits
        probs = torch.sigmoid(logits).squeeze(-1)
        preds = (probs >= threshold).int().tolist()
        return preds, probs.tolist()


# -------------------------------------------------------------
# 2. DATASET LOADERS
# -------------------------------------------------------------

def load_redditbias(split="test"):
    ds = load_dataset("AriHoltzman/redditbias")[split]
    texts = ds["text"]
    labels = ds["label"]
    return texts, labels

def load_sbic(split="test"):
    ds = load_dataset("sbic", "default")[split]
    texts = ds["post"]
    labels = ds["offensiveYN"]  # binary toxicity
    return texts, labels

def load_implicit_toxicity():
    ds = load_dataset("ucberkeley-dlab/implicit-hate")["test"]
    texts = ds["text"]
    # convert toxicity score → binary label
    labels = [1 if x >= 0.5 else 0 for x in ds["toxicity"]]
    return texts, labels

def load_gab():
    ds = load_dataset("hate_speech18")["test"]
    texts = ds["text"]
    labels = ds["label"]
    return texts, labels


# -------------------------------------------------------------
# 3. METRICS: Accuracy, OF, UF, Confusion Matrix, AUROC
# -------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    of_score = fp / (fp + tn)  # false positive rate
    uf_score = fn / (fn + tp)  # false negative rate

    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = None

    return {
        "accuracy": acc,
        "OF": of_score,
        "UF": uf_score,
        "AUROC": auc,
        "confusion": (tn, fp, fn, tp),
    }


# -------------------------------------------------------------
# 4. PLOTTING (OPTIONAL)
# -------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.show()


# -------------------------------------------------------------
# 5. RUN EVALUATION
# -------------------------------------------------------------

def evaluate(model_name, dataset="redditbias"):

    loader_map = {
        "redditbias": load_redditbias,
        "sbic": load_sbic,
        "implicit": load_implicit_toxicity,
        "gab": load_gab,
    }

    print(f"\nLoading dataset: {dataset}")
    texts, labels = loader_map[dataset]()

    print(f"Loaded {len(texts)} samples.")

    model = HFModelWrapper(model_name)

    print("\nRunning predictions...")
    preds, probs = model.predict(texts)

    m = compute_metrics(labels, preds, probs)

    print("\n==== Results ====")
    print(f"Accuracy: {m['accuracy']:.4f}")
    print(f"OF (false positive rate): {m['OF']:.4f}")
    print(f"UF (false negative rate): {m['UF']:.4f}")

    if m["AUROC"] is not None:
        print(f"AUROC: {m['AUROC']:.4f}")

    tn, fp, fn, tp = m["confusion"]
    print(f"Confusion Matrix: tn={tn}, fp={fp}, fn={fn}, tp={tp}")

    return m
