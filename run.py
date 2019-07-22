#!/usr/bin/python
"""
run.py: run script for NLI LG Model
Abhishek Sharma <sharm271@cs.purdue.edu>

Usage:
    run.py train --train-file=<file> --dev-file=<file> [options]

Options:
    -h --help                           show this screen.
    --train-file=<file>                 train_corpus
    --dev-file=<file>                   dev_corpus
    --vocab-file=<file>                 vocab json [default: vocab.json]
    --word-embeddings=<file>            word_vecs [default: ../data/wiki-news-300d-1M.txt]
    --max-epoch=<int>                   max epoch [default: 15]
    --patience=<int>                    wait for how many epochs to exit training [default: 2]
    --batch-size=<int>                  batch size [default: 16]
    --embed-size=<int>                  word embed_dim [default: 300]
    --hidden-size=<int>                 hidden dim [default: 256]
    --num-layers=<int>                  number of layers [default: 1]
    --clip-grad=<float>                 grad clip [default: 5.0]
    --lr=<float>                        learning rate [default: 0.001]
    --dropout=<float>                   dropout rate [default: 0.3]
    --beam-size=<int>                   beam size [default: 2]
    --max-decoding-time-step=<int>      max decoding time steps [default: 70]
    --save-model-to=<file>              save trained model [default: _model.pt]
"""
from __future__ import division

import time
import math

import torch

from docopt import docopt

from utils import loadEmbeddings
from utils import extractPairCorpus
from utils import batch_iter
from vocab import Vocab

from neural_model import NeuralModel
from nli_model import NLIModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(args, model, prems, label):
    """ 
    Evaluate the model on the data
    @param args (dict): command line args
    @param model (NeuralModel): Neural Model
    @param prems (list[list[str]]): list of prems
    @param label (str): hyp label to generate
    @return TODO
    """
    was_training = model.training
    model.eval()
    
    hyps = []
    with torch.no_grad():
         for prem in prems:
            possible_hyps = model.beam_search(prem, 
                beam_size=int(args['--beam-size']),
                max_decoding_time_step=int(args['max-decoding-time-step']))
            hyps.append(possible_hyps)

    if was_training:
        model.train()

    #return TODO

def train_lg_model(args, vocab, embeddings, train_data, label):
    """
    train LG model on the specific label
    @param args (dict): command line args
    @param vocab (Vocab): Vocab class obj
    @param embeddings (torch.tensor(len(vocab), embed_dim)): pretrained word embeddings
    @param train_data (list[tuple]): list of sent pairs containing premise and hypothesis
    @param label (str): hyp label    
    """
    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    model_save_path = label + args['--save-model-to']

    model = NeuralModel(vocab, int(args['--embed-size']), embeddings,
                        hidden_size=int(args['--hidden-size']),
                        num_layers=int(args['--num-layers']),
                        dropout_rate=float(args['--dropout']), device=device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))

    total_loss = .0
    total_hyp_words = 0

    begin_time = time.time()
    for epoch in range(int(args['--max-epoch'])):
        for prems, hyps in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            num_hyp_words_to_predict = sum(len(hyp[1:]) for hyp in hyps)

            optimizer.zero_grad()
            
            batch_size = len(prems)
            batch_loss = -model(prems, hyps).sum()
            loss = batch_loss / num_hyp_words_to_predict

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            
            batch_losses_val = batch_loss.item()
            total_loss += batch_losses_val

            total_hyp_words += num_hyp_words_to_predict
        
        print('epoch = %d, loss = %.2f, perplexity = %.2f, time_elapsed = %.2f sec'
            % (epoch, total_loss / total_hyp_words, 2**(total_loss / total_hyp_words), time.time() - begin_time))
        #reset epoch progress vars
        total_loss = .0
        total_hyp_words = 0

        #perform validation

def train(args):
    """
    train neural models
    @param args (dict): command line args
    """
    vocab = Vocab.load(args['--vocab-file'])
    embeddings = loadEmbeddings(vocab, args['--word-embeddings'], device)

    #construct set of train sent pairs for each hyp class
    entail_pairs, neutral_pais, contradict_pairs = extractPairCorpus(args['--train-file'])
    
    #train LG model for each hyp class
    train_lg_model(args, vocab, embeddings, train_data=entail_pairs, label='entailment')

if __name__ == "__main__":
    args = docopt(__doc__)
    if args['train']:
        train(args)
