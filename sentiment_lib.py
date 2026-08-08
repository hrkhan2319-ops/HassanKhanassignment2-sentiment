"""
sentiment_lib.py
Shared code for Assignment 2 sentiment classifier.

Model: an ensemble of 5 linear (logistic-regression-style) classifiers
implemented in PyTorch, trained on top of frozen pretrained features:

  * 300-d GloVe (glove-wiki-gigaword-300) mean-pooled word embeddings
  * 6 negation-aware sentiment-lexicon statistics (Hu & Liu opinion lexicon
    via NLTK)

Each ensemble member is trained with class-weighted cross-entropy (to handle
the 180/60 class imbalance), AdamW with weight decay, and early stopping on
its own stratified validation split. Predictions average the 5 softmax
outputs.
"""
import json
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn

SEEDS = [0, 1, 2, 3, 4]
EMBED_DIM = 300
N_LEX_FEATS = 6
TOKEN_RE = re.compile(r"[a-z']+")
NEGATORS = {"not", "n't", "no", "never", "nothing", "nobody",
            "neither", "nor", "hardly", "barely"}


# ----------------------------------------------------------------------------
# Pretrained resources (downloaded once, cached locally afterwards)
# ----------------------------------------------------------------------------
def load_glove():
    """GloVe 300-d vectors (~376 MB download on first run, then cached)."""
    import gensim.downloader as api
    return api.load("glove-wiki-gigaword-300")


def load_lexicon():
    """Hu & Liu opinion lexicon via NLTK (small download on first run)."""
    import nltk
    try:
        from nltk.corpus import opinion_lexicon
        opinion_lexicon.positive()
    except LookupError:
        nltk.download("opinion_lexicon", quiet=True)
        from nltk.corpus import opinion_lexicon
    return set(opinion_lexicon.positive()), set(opinion_lexicon.negative())


# ----------------------------------------------------------------------------
# Featurization
# ----------------------------------------------------------------------------
def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def glove_mean(text, kv):
    """Mean-pooled GloVe embedding of all in-vocabulary tokens."""
    vecs = [kv[t] for t in tokenize(text) if t in kv.key_to_index]
    if not vecs:
        return np.zeros(EMBED_DIM, dtype=np.float32)
    return np.mean(vecs, axis=0)


def lexicon_feats(text, pos_set, neg_set):
    """Six sentiment-lexicon statistics per 100 tokens.

    Raw positive/negative rates plus negation-aware versions in which a
    lexicon word directly preceded by a negator ("not good") flips polarity.
    """
    toks = tokenize(text)
    n = len(toks) + 1e-9
    p = sum(t in pos_set for t in toks)
    ng = sum(t in neg_set for t in toks)
    fp = fn = 0
    for i, t in enumerate(toks):
        prev = toks[i - 1] if i > 0 else ""
        if t in pos_set:
            fp, fn = (fp, fn + 1) if prev in NEGATORS else (fp + 1, fn)
        elif t in neg_set:
            fp, fn = (fp + 1, fn) if prev in NEGATORS else (fp, fn + 1)
    return [p / n * 100, ng / n * 100, (p - ng) / n * 100,
            fp / n * 100, fn / n * 100, (fp - fn) / n * 100]


def featurize(texts, kv, pos_set, neg_set):
    """texts -> (N, 306) feature matrix."""
    G = np.vstack([glove_mean(t, kv) for t in texts])
    L = np.array([lexicon_feats(t, pos_set, neg_set) for t in texts])
    return np.hstack([G, L]).astype(np.float32)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def make_model(n_features=EMBED_DIM + N_LEX_FEATS):
    return nn.Linear(n_features, 2)


def train_one(X, y, seed, lr=5e-3, weight_decay=3e-2,
              max_epochs=400, patience=60, val_frac=0.2, verbose=False):
    """Train one ensemble member with early stopping on a stratified split."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import balanced_accuracy_score

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    tr, va = train_test_split(np.arange(len(y)), test_size=val_frac,
                              stratify=y, random_state=seed)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xa = torch.tensor((X[tr] - mu) / sd, dtype=torch.float)
    Xv = torch.tensor((X[va] - mu) / sd, dtype=torch.float)
    ya = torch.tensor(y[tr])

    model = make_model(X.shape[1])
    counts = np.bincount(y[tr], minlength=2)
    class_w = torch.tensor(len(tr) / (2.0 * counts), dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optim = torch.optim.AdamW(model.parameters(), lr=lr,
                              weight_decay=weight_decay)

    best, best_state, since = -1.0, None, 0
    for epoch in range(max_epochs):
        model.train()
        optim.zero_grad()
        loss = criterion(model(Xa), ya)
        loss.backward()
        optim.step()

        model.eval()
        with torch.no_grad():
            ba = balanced_accuracy_score(y[va], model(Xv).argmax(1).numpy())
        if ba > best:
            best, since = ba, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        if since >= patience:
            break
        if verbose and epoch % 50 == 0:
            print(f"  seed {seed} epoch {epoch:3d} loss {loss.item():.4f} "
                  f"val_bal_acc {ba:.3f}")
    model.load_state_dict(best_state)
    return model, mu.astype(np.float32), sd.astype(np.float32), best


def train_ensemble(X, y, seeds=SEEDS, verbose=True):
    members = []
    for seed in seeds:
        model, mu, sd, val_ba = train_one(X, y, seed)
        members.append({"model": model, "mu": mu, "sd": sd})
        if verbose:
            print(f"seed {seed}: best val balanced accuracy = {val_ba:.3f}")
    return members


def predict_proba(members, X):
    probs = []
    for m in members:
        Xs = torch.tensor((X - m["mu"]) / m["sd"], dtype=torch.float)
        with torch.no_grad():
            probs.append(m["model"](Xs).softmax(1).numpy())
    return np.mean(probs, axis=0)


def predict(members, X):
    return predict_proba(members, X).argmax(1)


# ----------------------------------------------------------------------------
# Checkpointing
# ----------------------------------------------------------------------------
def save_checkpoint(members, path="model_checkpoint"):
    os.makedirs(path, exist_ok=True)
    config = {"model": "linear-ensemble", "n_members": len(members),
              "n_features": EMBED_DIM + N_LEX_FEATS,
              "embedding": "glove-wiki-gigaword-300 (mean pooled)",
              "lexicon": "nltk opinion_lexicon (Hu & Liu)",
              "seeds": SEEDS}
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    for i, m in enumerate(members):
        torch.save(m["model"].state_dict(),
                   os.path.join(path, f"model_seed{i}.pt"))
        np.savez(os.path.join(path, f"scaler_seed{i}.npz"),
                 mu=m["mu"], sd=m["sd"])


def load_checkpoint(path="model_checkpoint"):
    with open(os.path.join(path, "config.json")) as f:
        config = json.load(f)
    members = []
    for i in range(config["n_members"]):
        model = make_model(config["n_features"])
        model.load_state_dict(
            torch.load(os.path.join(path, f"model_seed{i}.pt"),
                       map_location="cpu"))
        model.eval()
        scal = np.load(os.path.join(path, f"scaler_seed{i}.npz"))
        members.append({"model": model, "mu": scal["mu"], "sd": scal["sd"]})
    return members, config
