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
    --vocab-file=<file>                 vocab json [default: vocab.json]
    --word-embeddings=<file>            word_vecs [default: ../data/wiki-news-300d-1M.txt]
    --max-epoch=<int>                   max epoch [default: 15]
    --batch-size=<int>                  batch size [default: 32]
    --log-every=<int>                   log every [default: 10]
    --embed-size=<int>                  word embed_dim [default: 300]
    --hidden-size=<int>                 hidden dim [default: 300]
    --num-layers=<int>                  number of layers [default: 2]
    --clip-grad=<float>                 grad clip [default: 5.0]
    --lr=<float>                        learning rate [default: 0.001]
    --dropout=<float>                   dropout rate [default: 0.1]
    --save-model-to=<file>              save trained model [default: nli_model.pt]
"""
from __future__ import division

import time
import math

import torch
import torch.nn.functional as F

from docopt import docopt

from utils import readCorpus
from utils import loadEmbeddings
from utils import extractPairCorpus
from utils import batch_iter
from utils import save
from utils import labels_to_indices
from utils import extractSentLabel
from vocab import Vocab
from nli_model import NLIModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    """
    train NLI model
    @param args (Dict): command line args
    """
    train_sents = readCorpus(args['--train-file'])
    clip_grad = float(args['--clip-grad'])
    vocab = Vocab.load(args['--vocab-file'])
    embeddings = loadEmbeddings(vocab, args['--word-embeddings'], device)

    #train NLI prediction model
    train_data = extractSentLabel(args['--train-file'])

    train_batch_size = int(args['--batch-size'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-model-to']

    model = NLIModel(vocab, int(args['--embed-size']), embeddings,
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
            labels_pred = model(prems, hyps)

            P = F.log_softmax(labels_pred, dim=-1)
            labels_indices = labels_to_indices(labels)
            labels_indices = labels_indices.to(device) 
            cross_entropy_loss = torch.gather(P, dim=-1,
                index=labels_indices.unsqueeze(-1)).squeeze(-1)

            batch_loss = -cross_entropy_loss.sum()
            loss = batch_loss / batch_size

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            if epoch == 0 and train_iter < 5:
                print('initial loss= %.2f' %(batch_loss.item()))
            
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