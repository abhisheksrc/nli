# Can Machines Infer? Natural Language Inference Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/abhishek9594/nli/blob/master/LICENSE)

Natural Language Inference- Language Generation: Given a *premise* and a label (*entailment, contradiction, or neutral*), generate an *hypothesis* for that label.

For example, for the *premise* **The sky is clear**  an entailing *hypothesis* can be  **It might not rain today**

We train 3 independent neural models to generate *hypothesis* for each label.

Before delving into the models and scripts let's take a look at the software versions we use for this project:

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
- for the format of data files please see [read_line](src/utils.py) in utils.

The model's architecture is described in [NeuralModel](src/neural_model.py), whereas the hyperparameters are set from [run.py](src/run.py).

Our model is similar in implementation as the Neural Machine Translation(NMT) models (seq2seq), with encoder and decoder.  We use BiLSTM encoder and LSTM decoder with attention mechanism.  We use cross-entropy loss function as our training objective while predicting the hypothesis words over the training corpus.  The size of the SNLI training corpus for all the 3 labels is approximately **549k** and is roughly balanced for each label.

## Evaluating:

### Semantic Similarity

Coming with evaluation metrics is often challenging for language generation.  Specially for this task, we could have multiple possible *hypotheses* for a given *premise* and a label.  However the *hypotheses* provided in the dataset should have some degree of semantic similarity with the generated *hypotheses*.  We therefore train 2 separate models for Semantic Textual Similarity (STS) task on the [STS Benchmark: Main English dataset](refs/sts.bib), [STS Wiki](http://ixa2.si.ehu.eus/stswiki).

The STS task is to compute meaning similarity score for a given pair of sentences on a continuous scale of `0.0 to 5.0`.  Two very similar meaning sentences should be given a high score, whereas dissimilar meaning sentences should get a low score.

An example of sentences with a score of `5.0`, from the corpus, is: **A band is performing on a stage.**  and **A band is playing onstage.**

In order to evaluate semantic similarity we come up with a feature vector using fixed size sentence embeddings and pass the feature vector(s) to a scoring function which yields the similarity score.

We train and compare performances of the following 2 models and choose the one which performs the best on the STS test set:

1. [Word Averaging Cosine Similarity](src/sts_avg.py)
  - This is our baseline model, where the sentence embedding is computed by averaging word embeddings.  We use cosine similarity as the scoring function on the input of sentence pair embeddings and scale the score on `5.0` scale.
  For training details please see [STS train Avg Sim](src/sts_train_avg.py).
  To run and obtain the STS model please run the following command from `$src` directory: 
  ```bash
  python sts_train_avg train --train-file=<file> --dev-file=<file> [options]
  ```
 2. [BiLSTM Encoder](src/sts_bilstm.py)
  - We run BiLSTM for each sentence and the sentence embedding is computed by concatenating all the hidden outputs from the   last time-steps for both forward and backward directions, from each layers.  The final feature vector is obtained by \[*s_a; s_b;* \|*s_a* \- *s_b*\|; *s_a* \* *s_b*\], where *s_a* and *s_b* are the sentence embeddings for the sentence pair *a* and *b*, *';'* denotes concatenation between those vectors, and *'\*'* denotes Hadamard product.  We then pass this feature vector to a Feed-forward layer followed by ReLu and dropout.  The output of the network is again passed to another Feed-forward layer followed by sigmoid.  Finally we scale the sigmoid output on `5.0` scale to predict the similarity score.
  For training details please see [STS train BiLSTM Sim](src/sts_train_bilstm.py).
  To run and obtain the STS trained model please run the following command from `$src` directory: 
  ```bash
  python sts_train_bilstm train --train-file=<file> --dev-file=<file> [options]
  ```
  
We use mean square error (MSE) as our training objective.  The squared error is calculated between the predicted vs the actual similarity score.  We save the model which performs the best on the STS dev set.  The size of the STS training corpus is **5.7k**

To run evaluations on a trained model (for example on the BiLSTM model) please run the following command from `$src` directory:
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

We would also like to evaluate the word overlap between the generated *hypotheses* against the dataset *hypotheses*.  We choose BLEU to evaluate the word overlap as it is commonly used for the NMT task.  However, the generated *hypotheses* could have very different words than the ones in the dataset and yet still be correct.  Therefore, to avoid penalizing such cases we only weigh `unigram` and `bigram` matching precision, with equal weights of `0.5`.

## Results:

The quantitative results on the SNLI dev and test sets are the comparisons between pairs of generated *hypotheses* vs dataset *hypotheses*.  We use Similarity (range between `0.0 to 5.0`) and BLEU (range between `0.0 to 1.0`) scores as described in the previous [section](#evaluating).  Higher the scores more the overlap between the *hypotheses* pairs.

Given that we have trained [NeuralModels](src/neural_model.py) for each label and a trained Similarity model as our `EVAL_MODEL`, to obtain the results please run the following command from `$src` directory:
```bash
python run.py test ENTAIL_MODEL NEUTRAL_MODEL CONTRADICT_MODEL EVAL_MODEL --test-file=<file> [options]
```
We use the trained encoder and decoder with attention mechanism with `beam-width = 5`, to generate the *hypotheses* and choose the final *hypothesis* having the maximum log-probability.

The quantitative performance of the Neural models on each label is shown here: (the numbers in the first set of columns is for the Dev set and in the second set of columns is for the Test set)

| label | Similarity | BLEU | Dev Size | Similarity| BLEU| Test Size |
|-|-|-|-|-|-|-
| *entailment* | 2.34 | 0.29 | 3.3k | 2.33 | 0.29 | 3.3k
| *neutral* | 1.98 | 0.21 | 3.2k | 1.97 | 0.21 | 3.2k
| *contradiction* | 1.40 | 0.19 | 3.2k | 1.40 | 0.19 | 3.2k

For the qualitative analysis we randomly pick 5 *premises* and show the generated *hypotheses* along with the dataset *hypotheses*, for each label:

  1. *premise*: The two young girls are dressed as fairies, and are playing in the leaves outdoors.
  
  |label| generated *hypothesis* |dataset *hypothesis*|
  |-|-|-|
  |*entailment*| Two young girls are playing outside. |Girls are playing outdoors.|
  |*neutral*| The girls are playing in the leaves. |The two girls play in the Autumn.|
  |*contradiction*| The girls are inside. |Two girls play dress up indoors.|
  
  2. *premise*: A young family enjoys feeling ocean waves lap at their feet.
  
  |label| generated *hypothesis* |dataset *hypothesis*|
  |-|-|-|
  |*entailment*| A family is enjoying the water. |A family is at the beach.|
  |*neutral*| A young family enjoys the ocean waves. |A young man and woman take their child to the beach for the first time.|
  |*contradiction*| The family is at the beach. |A family is out at a restaurant.|
  
  3. *premise*: Several men at a bar watch a sports game on the television.
  
  |label| generated *hypothesis* |dataset *hypothesis*|
  |-|-|-|
  |*entailment*| Men are watching a sports game. |Some men are watching TV at a bar.|
  |*neutral*| The men are watching a sports game. |There are men drinking beer at a sports bar.|
  |*contradiction*| The men are watching a movie. |The men are at a baseball game.|
  
  4. *premise*: Mom and little boy having fun & eating by the lake.
  
  |label| generated *hypothesis* |dataset *hypothesis*|
  |-|-|-|
  |*entailment*| The family is eating. |Two humans enjoying the outside.|
  |*neutral*| Mother and son at the lake. |A mom and a son spending time together on their weekend supervised visit.|
  |*contradiction*| A father and son are swimming. |A mom is pouring grape juice on the ground while her son is crying.|
  
  5. *premise*: A woman in a teal apron prepares a meal at a restaurant.
  
  |label| generated *hypothesis* |dataset *hypothesis*|
  |-|-|-|
  |*entailment*| A woman is cooking. |A woman in restaurant|
  |*neutral*| A woman is cooking dinner. |A woman prepare a lunch in restaurant|
  |*contradiction*| A woman is eating at a restaurant. |A woman is walking in park|


## Discussions:

The Similarity and BLEU scores are just heuristics to evaluate the quality of our language generation.  However, we can relatively compare the results across each label. It is interesting that the relative order of the labels in terms of their scores (from high to low) is:

  1. *entailment*
  2. *neutral*
  3. *contradiction*
  
For the qualitative analysis, we only have 5 examples, but notice that the model performs quite well in generating *entailment* *hypotheses*, whereas poor in generating *neutral* *hypotheses* by getting the first 4 examples wrong, and average in the *contradiction* case by only making a mistake in the 2nd example.  

It would be interesting to find the reasons for these results.

The dataset *hypotheses* are composed by humans and in some cases humans can disagree on annotating labels for a *premise*, *hypothesis* pair.  For example the following pair could have multiple labels:

| *premise* | *hypothesis* | label |
|-|-|-|
| I am going to Europe. | I will be traveling in Rome. | *entailment*, or *neutral*, or *contradiction* ?|

Therefore it could either arise from the nature of the training corpus or the limitation of the Neural models in generating *neutral* and *contradiction* *hypotheses*.  We also feel that it could be sometimes challenging to differentiate between *neutral* and *contradiction* examples, which could effect the models.

## Miscellaneous:

### Vocabulary:

For the STS task we construct the vocabulary by using the SNLI and STS training corpus, whereas for language generation we only use the SNLI training corpus.  The reason for using both the corpus for STS is to train our Similarity models on the words used in the SNLI corpus, so they can better evaluate the generated results.  Whereas the reason for using only the SNLI corpus for language generation is that we want to limit outside knowledge to the Neural models.

To generate the `pickle` dump for SNLI + STS vocabulary please run the following command from `$src` directory:
```bash
python vocab.py --snli-corpus=<file> --sts-corpus=<file> [options]
```

The number of tokens in the SNLI vocab is `42,634` whereas in the SNLI + STS vocab is `48,802`.

We use [NLTK](https://www.nltk.org/) to extract these tokens from the training corpus.

### Embeddings:

We assign embeddings to all of the words present in our vocabulary.  We use the [GloVe](refs/glove.bib) embeddings for the vocabulary words found in the pre-trained word embeddings [file](https://nlp.stanford.edu/projects/glove/) (trained on 840B tokens from Common Crawl), and random embeddings for the words not found in this pre-trained file.

We also fine-tune these embeddings while training the models on the specific tasks.

To generate the `pickle` dump for the embeddings, please see [load_embeddings](src/utils.py) in the utils.

### Training details:

We use [Adam](refs/adam.bib) optimizer, with an initial learning rate of `1e-3` and decay of `0.5` after every `2` epochs.  We run a maximum of `15` epochs with the early stopping criteria of no improvement on the dev set after `5` epochs.  As a measure of improvement, for language generation task we use similarity score on the dev set and for the STS task we use MSE loss on the dev set.

We also use [gradient clipping](refs/grad_clip.bib) for our language generation model.

The total number of hyperparameters:
  1. `NeuralModel = 25.35M`
  2. `BiLSTMSim = 19.00M`
  3. `AvgSim = 14.64M`
  
We use the performance measure on the dev set to decide the training parameters and the models' hyperparameters.

To run all our experiments we utilize GPU and it takes roughly 3.5 hours to train the NeuralModel, 0.03 hour for the BiLSTMSim model, and 0.02 hour for the AvgSim model.
