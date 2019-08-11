# NLI-LG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/abhishek9594/nli/blob/master/LICENSE)

Natural Language Inference- Language Generation: Given a *premise* and a label (*entailment, contradiction, or neutral*), generate an *hypothesis* for that label.

For example, an entailing *hypothesis* for the *premise* **The sky is clear**  can be  **It will not rain today**

We train 3 independent neural models to generate *hypothesis* for each label.

Before delving into the models and scripts let's take a look at the software versions used for this project:

- `Python 2.7.15+`
- `PyTorch 1.0.0`
- `NLTK 3.4.3`
- `docopt 0.6.2`
- `NumPy 1.15.4`
- `SciPy 1.2.0`

## Training:

### Dataset: [SNLI](refs/snli.bib)

To train the neural models on the dataset run the following command from `$src` directory:

```bash
python run.py train EVAL_MODEL --train-file=<file> --dev-file=<file> [options]
```

- where `EVAL_MODEL` is described in the next [section](#evaluating)
- for the format of data files please see [utils](src/utils.py)

The model's architecture is described in [NeuralModel](src/neural_model.py), whereas the hyperparameters are set from [run.py](src/run.py).

Our model is similar in implementation as the Neural Machine Translation(NMT) models, with encoder and decoder.  We use BiLSTM encoder and LSTM decoder with attention mechanism.  We use cross-entropy loss function as our training objective while predicting the hypothesis words over the training corpus.

## Evaluating:

Coming with evaluation metrics is often challenging for language generation.  Specially for this task, we could have multiple possible *hypotheses* for a given *premise* and a label.  However the *hypotheses* provided in the dataset should have some degree of semantic simmilarity with the generated *hypotheses*.  We therefore train 3 seperate models for Semantic Textual Similarity (STS) task on the [STS Benchmark: Main English dataset](refs/sts.bib), [STS Wiki](http://ixa2.si.ehu.eus/stswiki).

The STS task is to compute similarity score for a given pair of sentences on a continuous scale of 0.0 to 5.0.  Two very similar sentences should be given a high score, whereas dissimilar sentences should get a low score.

An example of sentences with a score of 5.0, from the corpus, is: **A band is performing on a stage.**  and **A band is playing onstage.**

In order to evaluate semantic similarity we come up with a feature vector using fixed size sentence embeddings and pass the feature vector to a scoring function which yields the similarity score.

We train and compare performances of the following 2 models and choose the one which performs the best on the STS test set:

1. [Word Averaging Cosine Similarity](src/sts_avg.py)
  - This is our baseline model, where the sentence embedding is computed by averaging word embeddings and the feature vector is obtained by concatenating the embeddings.  We use cosine similarity as the scoring function and scale the score on 5.0 scale.
  For training details please see [STS train Avg Sim](src/sts_train_avg.py).
  To run and obtain the STS model please run the following command from `$src` directory: 
  ```bash
  python sts_train_avg train --train-file=<file> --dev-file=<file> [options]
  ```
 2. [BiLSTM Encoder](src/sts_bilstm.py)
  - We run BiLSTM for each sentence and the sentence embedding is computed by concatenating all the hidden outputs from the   last time-steps for both forward and backward directions, from each layers.  The final feature vector is obtained by \[*s_a; s_b; abs(s_a \- s_b); s_a \* s_b*\], where *s_a* and *s_b* are the sentence embeddings for the sentence pair *a* and *b* and *';'* denotes concatenation between those vectors.
  For training details please see [STS train BiLSTM Sim](src/sts_train_bilstm.py).
  To run and obtain the STS model please run the following command from `$src` directory: 
  ```bash
  python sts_train_bilstm train --train-file=<file> --dev-file=<file> [options]
  ```
  
We use mean square error (MSE) as our training objective.  The squared error is calculated between the predicted vs the actual similarity score.  We save the model which performs the best on the STS dev set.

To finally evaluate the performance we compare the Pearson correlation coefficient (Pearson's r) between the 2 models:

| Model | Dev | Test  |
|-|-|-|
| [Word Averaging Cosine Similarity](src/sts_avg.py) | 61% | 50% |
| [BiLSTM Encoder](src/sts_bilstm.py) | 74% | 74.5% |

Having high Pearson's r means having high correlation with the actula results and hence we pick the [BiLSTM Encoder](src/sts_bilstm.py) as our `EVAL_MODEL`.
