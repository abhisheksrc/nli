#!/usr/bin/python
"""
nli_train.py: Script for NLI Model
Abhishek Sharma <sharm271@cs.purdue.edu>

Usage:
    nli_train.py train [--train-file=<file> --dev-file=<file>] [--word-embeddings=<file>] [options]

Options:
    -h --help                           show this screen.
    --train-file=<file>                 train_corpus [default: ../data/snli_train.txt]
    --dev-file=<file>                   dev_corpus [default: ../data/snli_dev.txt]
    --word-embeddings=<file>            word_vecs [default: ../data/wiki-news-300d-1M.txt]
    --max-epoch=<int>                   max epoch [default: 15]
    --batch-size=<int>                  batch size [default: 32]
    --log-every=<int>                   log every [default: 10]
    --embed-size=<int>                  word embed_dim [default: 300]
    --hidden-size=<int>                 hidden dim [default: 256]
    --num-layers=<int>                  number of layers [default: 1]
    --lr=<float>                        learning rate [default: 0.0002]
    --dropout=<float>                   dropout rate [default: 0.1]
    --save-model-to=<file>              save trained model [default: nli_model.pt]
"""
from __future__ import division

import time
import math

import torch

from docopt import docopt

from utils import readCorpus
from utils import loadEmbeddings
from utils import extractPairCorpus
from utils import batch_iter
from utils import save
from vocab import Vocab

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    """
    train NLI model
    @param args (Dict): command line args
    """
    train_sents = readCorpus(args['--train-file'])
    vocab = Vocab.build(train_sents)
    embeddings = loadEmbeddings(vocab, args['--word-embeddings'], device)

    #train NLI prediction model
    (prems, hyps, labels) = extractSentLabel(args['--train-file'])
    train_data = (prems, hyps, labels)

    train_batch_size = int(args['--batch-size'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-model-to']

    model = NeuralModel(vocab, int(args['--embed-size']), embeddings,
                        hidden_size=int(args['--hidden-size']),
                        num_layers=int(args['--num-layers']),
                        dropout_rate=float(args['--dropout']), device=device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))

    train_time = begin_time = time.time()
    train_iter = report_loss = 0
    patience = cum_loss = cum_examples = report_examples = 0

    for epoch in range(int(args['--max-epoch'])):
        for prems, hyps, labels in batch_iter(train_data, batch_size=train_batch_size, shuffle=True, label=True):
            train_iter += 1

            optimizer.zero_grad()
            
            batch_size = len(prems)
            batch_loss = -model(prems, hyps).sum()
            loss = batch_loss / batch_size

            loss.backward()

            optimizer.step()
            
            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch= %d, iter= %d, avg. loss= %.2f, cum. examples= %d, time elapsed= %.2f sec'
                    %(epoch, train_iter, report_loss / report_examples, cum_examples, time.time() - begin_time))

                train_time = time.time()
                report_loss = report_examples = 0

    save(model.state_dict(), model_save_path)

if __name__ == "__main__":
    args = docopt(__doc__)
    if args['train']:
        train(args)
