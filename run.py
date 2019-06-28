#!/usr/bin/python
"""
run.py: run script for NLI Model
Abhishek Sharma <sharm271@cs.purdue.edu>

Usage:
    run.py train [--train-file=<file> --dev-file=<file>] [--word_embeddings=<file>] [options]

Options:
    -h --help                           show this screen.
    --train_file=<file>                 train_corpus [default: ../data/snli_train.txt]
    --dev_file=<file>                   dev_corpus [default: ../data/snli_dev.txt]
    --word_embeddings=<file>            word_vecs [default: ../data/wiki-news-300d-1M.txt]
    --max-epoch=<int>                   max epoch [default: 10]
    --batch-size=<int>                  batch size [default: 16]
    --embed-dim=<int>                   word embed_dim [default: 300]
    --hidden-dim=<int>                  hidden dim [default: 512]
    --clip-grad=<float>                 grad clip [default: 5.0]
    --lr=<float>                        learning rate [default: 0.05]
    --dropout=<float>                   dropout rate [default: 0.5]
    --save-model=<file>                 save trained model [default: model.pt]
"""

import torch

from docopt import docopt

from model_embeddings import ModelEmbeddings
from utils import readCorpus
from utils import loadEmbeddings
from vocab import Vocab

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    """
    train neural models
    @param args (Dict): the command line args
    """
    train_sents = readCorpus(args['--train_file'])
    vocab = Vocab.build(train_sents)
    embeddings = loadEmbeddings(vocab, args['--word_embeddings'], device)

if __name__ == "__main__":
    args = docopt(__doc__)
    if args['train']:
        train(args)
