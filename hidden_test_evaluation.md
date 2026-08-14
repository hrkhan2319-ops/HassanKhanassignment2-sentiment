# Hidden Test Evaluation (Stage 2)

Inference-only evaluation of the frozen Stage 1 checkpoint on the released
`hidden_test.csv` (600 reviews, balanced 300 negative / 300 positive). No retraining
or fine-tuning was done. All predictions come from the exact `model_checkpoint/`
submitted in Stage 1.

## Results

| Test set | Size | Accuracy | Balanced accuracy |
|---|---|---|---|
| Public test (Stage 1) | 400 | 0.7575 | 0.7575 |
| Hidden test (Stage 2) | 600 | 0.7633 | 0.7633 |

### Hidden test confusion matrix

```
              predicted
              neg    pos
true neg  [  248     52 ]
true pos  [   90    210 ]
```

| | precision | recall | f1 | support |
|---|---|---|---|---|
| negative (0) | 0.7337 | 0.8267 | 0.7774 | 300 |
| positive (1) | 0.8015 | 0.7000 | 0.7473 | 300 |
| **accuracy** | | | **0.7633** | 600 |

## Comparison

The hidden test accuracy (0.7633) is very close to the public test accuracy (0.7575),
about 0.6 percentage points higher. Both test sets are balanced and come from the same
Pang & Lee corpus, and the public test was never used for training or model selection,
so getting similar numbers on both suggests the model generalizes to new reviews
instead of having memorized the public set.

From the confusion matrix, the model gets more negative reviews right (recall 0.83)
than positive ones (recall 0.70), so it leans slightly toward predicting negative.
This is probably left over from the 3:1 positive/negative imbalance in the 240-review
training set. The class weighting fixes most of that bias but not all of it.

## If we had more time or compute

- Fine-tune a small pretrained Transformer (e.g., DistilBERT) with a low learning
  rate, layer freezing, and strong regularization. Contextual embeddings should beat a
  frozen GloVe average even with only 240 labels.
- Better document encoding: SIF or weighted pooling, or sentence-level encoding with
  attention over sentences, since long reviews mix plot summary with opinion and plain
  averaging waters down the opinionated sentences.
- Semi-supervised learning: pseudo-labeling or consistency training on unlabeled movie
  reviews to make up for the small labeled set.
- Data augmentation allowed by the rules (embedding-space mixup between training
  examples) and tuning the decision threshold on cross-validated predictions.
