"""Generate predictions from the saved Stage 1 checkpoint (no retraining).

Usage:
    python predict.py <input.csv> <output.csv> [checkpoint_dir]

The input CSV must contain `id` and `text` columns. The output CSV contains
exactly `id,predicted_label` with labels 0 (negative) or 1 (positive).
"""
import sys

import pandas as pd

import sentiment_lib as sl


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    in_path, out_path = sys.argv[1], sys.argv[2]
    ckpt = sys.argv[3] if len(sys.argv) > 3 else "model_checkpoint"

    df = pd.read_csv(in_path)
    members, config = sl.load_checkpoint(ckpt)
    print(f"loaded checkpoint: {config['model']} "
          f"({config['n_members']} members, {config['n_features']} features)")

    kv = sl.load_glove()
    pos_set, neg_set = sl.load_lexicon()
    X = sl.featurize(df["text"].tolist(), kv, pos_set, neg_set)
    pred = sl.predict(members, X)

    pd.DataFrame({"id": df["id"], "predicted_label": pred}).to_csv(
        out_path, index=False)
    print(f"wrote {len(pred)} predictions to {out_path}")


if __name__ == "__main__":
    main()
