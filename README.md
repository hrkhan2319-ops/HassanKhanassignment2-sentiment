# Assignment 2 — Sentiment Classification with Neural Language Models

Binary sentiment classification (0 = negative, 1 = positive) of movie reviews from the Pang & Lee polarity corpus, on a deliberately small (240 examples) and imbalanced (180 positive / 60 negative) training set.

## Model

A combination of 5 linear classifiers (PyTorch `nn.Linear(306, 2)`) over frozen pretrained features:

- 300-d GloVe embeddings (`glove-wiki-gigaword-300`), averaged over the review
- 6 negation-aware sentiment-lexicon features (Hu & Liu opinion lexicon via NLTK)

Each classifier is trained with class-balancing cross-entropy (inverse frequency of each class in the dataset as the balancing weight for the 3:1 imbalance), AdamW (lr 5e-3, wd 0.03), full batch gradients, and early stopping on its own 20% validation split. Final prediction is the average of softmax predictions of the 5 classifiers.

**Public test accuracy: 0.7575** (balanced accuracy 0.7575). See `stage1_notebook.ipynb` for the detailed report, confusion matrix, and comparison with other methods (TF-IDF n-grams, trainable fastText-style embeddings, MLP head – all underperformed the frozen-pretrained-features + linear-head approach at n=240).

## Repository layout

```
sentiment_lib.py             shared featurization / training / checkpoint code
stage1_notebook.ipynb        Stage 1: training, evaluation, writeup (executed)
predict.py                   standalone prediction script
model_checkpoint/            trained model (config, 5 state dicts, 5 scaler files)
public_test_predictions.csv  predictions for public_test.csv (id,predicted_label)
requirements.txt

# added in Stage 2 (after hidden_test.csv is released):
# stage2_notebook.ipynb      inference-only hidden-test evaluation
# hidden_test_predictions.csv
```

## Setup

Python 3.9+ on a CPU-only Mac or Windows laptop is sufficient.

```bash
pip install -r requirements.txt
```

The first run downloads two cached resources automatically:

- GloVe vectors via `gensim.downloader` (~376 MB, one-time)
- the NLTK opinion lexicon (small, one-time)

## Reproducing Stage 1

```bash
jupyter notebook stage1_notebook.ipynb   # run all cells
```

This trains the ensemble (seconds on CPU), saves `model_checkpoint/`, evaluates on the public test set, and produces `public_test_predictions.csv`. Everything is seeded to make it deterministic.

## Predictions without training

`predict.py` loads the saved checkpoint and generates predictions for any CSV that contains `id` and `text` columns:

```bash
python predict.py public_test.csv my_predictions.csv
```

## Stage 2 

Copy `hidden_test.csv` to the repository root and execute all the cells of `stage2_notebook.ipynb`. This script does inference only, i.e., it loads the frozen Stage 1 checkpoint, calculates hidden-test accuracy and produces a confusion matrix, and generates `hidden_test_predictions.csv`. There is no training or other changes to the model.

## Use of AI

Anthropic's Claude was utilized as the coding assistant to draft the featurization and training scripts, and conduct model comparison experiments documented in the notebook. Decisions regarding design of the algorithm were verified using 5-fold cross-validation on the training data only, and neither training nor test data samples were changed or manually annotated.

## References

- Pang & Lee (2004), movie review polarity corpus (course-provided / NLTK release)
- Pennington, Socher & Manning (2014), *GloVe: Global Vectors for Word Representation* — pre-trained GloVe vectors provided through `gensim.downloader`
- Hu & Liu (2004), *Mining and Summarizing Customer Reviews* — opinion lexicon provided through NLTK
