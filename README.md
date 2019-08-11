# NLI-LG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/abhishek9594/nli/blob/master/LICENSE)

Natural Language Inference- Language Generation: Given a *premise* and a label (*entailment, contradiction, or neutral*), generate an *hypothesis* for that label.

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

- where `EVAL_MODEL` is descibed in the next section
