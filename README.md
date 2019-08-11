# NLI-LG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/abhishek9594/nli/blob/master/LICENSE)

Natural Language Inference- Language Generation: Given a *premise* and a label (*entailment, contradiction, or neutral*), generate an *hypothesis* for that label.

For example, for the *premise* **The sky is clear**  an entailing *hypothesis* can be  **It will not rain today**

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

Our model is similar in implementation as the Neural Machine Translation(NMT) models (seq2seq), with encoder and decoder.  We use BiLSTM encoder and LSTM decoder with attention mechanism.  We use cross-entropy loss function as our training objective while predicting the hypothesis words over the training corpus.  The size of the SNLI training corpus for all the 3 labels is approximately **549k** and is roughly balanced for each label.

## Evaluating:

### Semantic Similarity

Coming with evaluation metrics is often challenging for language generation.  Specially for this task, we could have multiple possible *hypotheses* for a given *premise* and a label.  However the *hypotheses* provided in the dataset should have some degree of semantic similarity with the generated *hypotheses*.  We therefore train 2 separate models for Semantic Textual Similarity (STS) task on the [STS Benchmark: Main English dataset](refs/sts.bib), [STS Wiki](http://ixa2.si.ehu.eus/stswiki).

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
  - We run BiLSTM for each sentence and the sentence embedding is computed by concatenating all the hidden outputs from the   last time-steps for both forward and backward directions, from each layers.  The final feature vector is obtained by \[*s_a; s_b;* \|*s_a* \- *s_b*\|; *s_a* \* *s_b*\], where *s_a* and *s_b* are the sentence embeddings for the sentence pair *a* and *b* and *';'* denotes concatenation between those vectors.  We then pass this feature vector to a Feed-forward layer followed by ReLu and dropout.  The output of the network is again passed to another Feed-forward layer followed by sigmoid.  Finally we scale the sigmoid output on 5.0 scale to predict the similarity score.
  For training details please see [STS train BiLSTM Sim](src/sts_train_bilstm.py).
  To run and obtain the STS trained model please run the following command from `$src` directory: 
  ```bash
  python sts_train_bilstm train --train-file=<file> --dev-file=<file> [options]
  ```
  
We use mean square error (MSE) as our training objective.  The squared error is calculated between the predicted vs the actual similarity score.  We save the model which performs the best on the STS dev set.  The size of the STS training corpus is **5.7k**

To run evalutions on a trained model (for example on the BiLSTM model) please run the following command from `$src` directory:
```bash
python sts_train_bilstm.py test MODEL_PATH --test-file=<file> [options]
 ```

To finally evaluate the performance we compare the Pearson correlation coefficient (Pearson's r) between the 2 models:

| Model | Dev (Size = **1.5k**) | Test (Size = **1.3k**)  |
|-|-|-|
| [Word Averaging Cosine Similarity](src/sts_avg.py) | 61% | 50% |
| [BiLSTM Encoder](src/sts_bilstm.py) | 74% | 74.5% |

Having high Pearson's r means having high correlation with the actual results and hence we pick the [BiLSTM Encoder](src/sts_bilstm.py) as our `EVAL_MODEL`.

### [BLEU](https://www.aclweb.org/anthology/P02-1040)

We would also like to evaluate the word overlap between the generated *hypotheses* against the dataset *hypotheses*.  We choose BLEU to evaluate the word overlap as it is commonly used for the NMT task.  However, the generated *hypotheses* could have very different words than the ones in the dataset and yet still be correct.  Therefore, to avoid penalizing such cases we only weigh unigram and bigram matching precision, with equal weights of 0.5.

## Results:

The quantitative results on the SNLI dev and test sets are the comparisons between pairs of generated *hypotheses* vs dataset *hypotheses*.  We use Similarity and BLEU scores as described in the previous [section](#evaluating).  Higher the scores better the match between the *hypotheses* pairs.

Given that we have trained [NeuralModels](src/neural_model.py) for each label and a trained Similarity model as our `EVAL_MODEL`, to obtain the results please run the following command from `$src` directory:
```bash
python run.py test ENTAIL_MODEL NEUTRAL_MODEL CONTRADICT_MODEL EVAL_MODEL --test-file=<file> [options]
```
We use the trained eocoder and decoder with attention mechanism with beam-width = 5, to generate the *hypotheses* and choose the final *hypothesis* having the maximum log-probability.

The quantitative performance of the Neural models on each label is shown here: (the numbers in the first set of columns is for the Dev set and in the second set of columns is for Test set)

| label | Similarity (0.0 - 5.0) | BLEU (0.0 - 1.0) | Dev Size | Similarity (0.0 - 5.0) | BLEU (0.0 - 1.0) | Test Size |
|-|-|-|-|-|-|-
| *entailment* | 2.34 | 0.29 | 3.3k | 2.33 | 0.29 | 3.3k
| *neutral* | 1.98 | 0.21 | 3.2k | 1.97 | 0.21 | 3.2k
| *contradiction* | 1.40 | 0.19 | 3.2k | 1.40 | 0.19 | 3.2k

Whereas the qualitative performance will be how "good" the generated hypotheses are.  We randomly pick 5 *premises* and show the *hypotheses* generated by our model and provided in the datset, for each label:


## Discussions:

The Similarity and BLEU scores are just heuristics to evaluate the quality of our language generation.  However, we can relatively compare the results across each label. It is interesting that the relative order of the labels in terms of their scores (from high to low) are:

  1. *entailment*
  2. *neutral*
  3. *contradiction*
  
This observation is also reflected in the qualitative results.  Notice for the *premise*

It would be interesting to find the reasons for these results.

The dataset *hypotheses* are composed by humans and in some cases humans can disagree on annotating labels for a *premise*, *hypothesis* pair.  For example the following pair could have multiple labels:

| *premise* | *hypothesis* | label |
|-|-|-|
| I am going to Europe. | I will be traveling in Rome. | *entailment*, or *neutral*, or *contradiction* ?|

Therefore it could either arise from the nature of the training corpus or the limitation of the Neural models in generating *neutral* and *contradictory* *hypotheses*.

## Miscellaneous:
