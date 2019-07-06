#!/usr/bin/python
"""
run.py: run script for NLI Model
Abhishek Sharma <sharm271@cs.purdue.edu>

Usage:
    run.py train [--train-file=<file> --dev-file=<file>] [--word-embeddings=<file>] [options]

Options:
    -h --help                           show this screen.
    --train-file=<file>                 train_corpus [default: ../data/snli_train.txt]
    --dev-file=<file>                   dev_corpus [default: ../data/snli_dev.txt]
    --word-embeddings=<file>            word_vecs [default: ../data/wiki-news-300d-1M.txt]
    --max-epoch=<int>                   max epoch [default: 15]
    --batch-size=<int>                  batch size [default: 16]
    --embed-size=<int>                  word embed_dim [default: 256]
    --hidden-size=<int>                 hidden dim [default: 256]
    --num-layers=<int>                  number of layers [default: 2]
    --clip-grad=<float>                 grad clip [default: 5.0]
    --lr=<float>                        learning rate [default: 0.001]
    --dropout=<float>                   dropout rate [default: 0.3]
    --save-model-to=<file>              save trained model [default: model.pt]
"""
from __future__ import division

import torch

from docopt import docopt

from utils import readCorpus
from utils import loadEmbeddings
from utils import extractPairCorpus
from utils import batch_iter
from utils import save
from vocab import Vocab

from neural_model import NeuralModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(args, vocab, embeddings, train_data, label):
    """
    train LG model on the specific label
    @param train_data (List[tuple]): list of sent pairs containing premise and hypothesis
    @param args (Dict): command line args
    @param label (str): hyp label    
    """
    batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    model_save_path = args['--save-model-to']

    model = NeuralModel(vocab, int(args['--embed-size']), embeddings,
                        hidden_size=int(args['--hidden-size']),
                        num_layers=int(args['--num-layers']),
                        dropout_rate=float(args['--dropout']), device=device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))

    for epoch in range(int(args['--max-epoch'])):
        epoch_loss = 0.0
        batch_losses_val = 0.0
        for prems, hyps in batch_iter(train_data, batch_size, shuffle=True):
            optimizer.zero_grad()
            
            batch_loss = model(prems, hyps).sum()
            loss = -batch_loss / len(prems)

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            
            batch_losses_val += batch_loss.item()

        epoch_loss = batch_losses_val / len(train_data)
        print('progress: loss= %.2f' % (epoch_loss))

    save(model.state_dict(), model_save_path)

def train(args):
    """
    train neural models
    @param args (Dict): command line args
    """
    train_sents = readCorpus(args['--train-file'])
    vocab = Vocab.build(train_sents)
    #embeddings = loadEmbeddings(vocab, args['--word-embeddings'], device)

    #construct set of train sent pairs for each hyp class
    entail_pairs, neutral_pais, contradict_pairs = extractPairCorpus(args['--train-file'])
    
    #train LG model for each hyp class
    train_model(args, vocab, embeddings=None, train_data=entail_pairs, label='entailment')

if __name__ == "__main__":
    args = docopt(__doc__)
    if args['train']:
        train(args)
