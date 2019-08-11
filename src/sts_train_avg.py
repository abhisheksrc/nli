#!/usr/bin/python
"""
nli_train.py: Script for NLI Model
Abhishek Sharma <sharm271@cs.purdue.edu>

Usage:
    sts_train_avg.py train --train-file=<file> --dev-file=<file> [options]
    sts_train_avg.py test MODEL_PATH --test-file=<file> [options]

Options:
    -h --help                           show this screen.
    --train-file=<file>                 train_corpus
    --dev-file=<file>                   dev_corpus
    --test-file=<file>                  test_corpus
    --vocab-file=<file>                 vocab dump [default: vocab.pickle]
    --pretrained-embeddings=<file>      word embeddings [default: glove-embeddings.pickle]
    --max-epoch=<int>                   max epoch [default: 15]
    --patience=<int>                    wait for how many epochs to exit training [default: 5]
    --batch-size=<int>                  batch size [default: 32]
    --embed-size=<int>                  embedding size [default: 300]
    --lr=<float>                        learning rate [default: 1e-3]
    --save-model-to=<file>              save trained model [default: avg_sim.pt]
"""
from __future__ import division

import time
import math
import pickle
from scipy.stats import pearsonr

import torch
import torch.nn.functional as F

from docopt import docopt

from utils import batch_iter
from utils import extract_sents_score
from vocab import Vocab
from sts_avg import AvgSim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, data, batch_size):
    """
    Evaluate the model on the data
    @param model (AvgSim): AvgSim Model
    @param data (list[tuple(sent1, sent2, score)]): list of sent_pairs, sim_score
    @param batch_size (int): batch size
    @return mean_loss (float): MSE loss on the scores_pred vs scores
    @return corr (float): correlation b/w scores_pred vs scores
    """
    was_training = model.training
    model.eval()

    total_loss = .0
    cum_scores = []
    cum_scores_pred = []
    with torch.no_grad():
        for sents1, sents2, scores in batch_iter(data, batch_size, shuffle=False, result=True):
            scores = torch.tensor(scores, dtype=torch.float, device=device)
            scores_pred = model(sents1, sents2)
            loss = F.mse_loss(scores_pred, scores, reduction='sum')
            total_loss += loss.item()

            cum_scores.extend(scores.tolist())
            cum_scores_pred.extend(scores_pred.tolist())

    mean_loss = total_loss / len(data)
    corr, p_val = pearsonr(cum_scores_pred, cum_scores)
    
    if was_training:
        model.train()

    return mean_loss, corr

def train(args):
    """
    train AvgSim model
    @param args (dict): command line args
    """
    vocab = Vocab.load(args['--vocab-file'])
    embeddings = pickle.load(open(args['--pretrained-embeddings'], 'rb'))
    embeddings = torch.tensor(embeddings, dtype=torch.float, device=device)

    train_data = extract_sents_score(args['--train-file'])
    dev_data = extract_sents_score(args['--dev-file'])

    train_batch_size = int(args['--batch-size'])
    dev_batch_size = int(args['--batch-size'])
    model_save_path = args['--save-model-to']

    model = AvgSim(vocab, int(args['--embed-size']), embeddings)

    model = model.train()
    model = model.to(device)

    init_lr = float(args['--lr'])
    optimizer = torch.optim.Adam(model.parameters(), init_lr)

    total_loss = .0
    hist_dev_corrs = []
    patience = 0

    begin_time = time.time()
    for epoch in range(int(args['--max-epoch'])):
        for sents1, sents2, scores in batch_iter(train_data, batch_size=train_batch_size, shuffle=True, result=True):
            optimizer.zero_grad()
   
            scores = torch.tensor(scores, dtype=torch.float, device=device) 
            scores_pred = model(sents1, sents2)

            batch_size = len(sents1)
            batch_loss = F.mse_loss(scores_pred, scores, reduction='sum')
            loss = batch_loss / batch_size
            loss.backward()

            optimizer.step()
            
            total_loss += batch_loss.item()

        #print train loss at the end of each epoch
        train_loss = total_loss / len(train_data)
        print('epoch = %d: mean loss = %.2f, time elapsed = %.2f sec'
            % (epoch, train_loss, time.time() - begin_time))
        total_loss = .0

        #perform validation
        dev_loss, dev_corr = evaluate(model, dev_data, dev_batch_size)
        is_better = epoch == 0 or dev_corr > max(hist_dev_corrs)
        hist_dev_corrs.append(dev_corr)

        if is_better:
            #reset patience
            patience = 0
            #save model
            model.save(model_save_path)

        else:
            patience += 1
            if patience == int(args['--patience']):
                print('finishing training: dev loss = %.2f, dev corr. = %.2f'
                    % (dev_loss, dev_corr))
                exit(0)

        print('validation: dev loss = %.2f, dev corr. = %.2f'
            % (dev_loss, dev_corr))
        
        #update lr after every 2 epochs
        lr = init_lr / 2 ** (epoch // 2)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def test(args):
    """
    test Avg Sim
    @param args (dict): command line args
    """
    test_data = extract_sents_score(args['--test-file'])
    model = AvgSim.load(args['MODEL_PATH'])
    model = model.to(device)
    test_loss, test_corr = evaluate(model, test_data, batch_size=int(args['--batch-size']))
    print('final test correlation= %.4f' %(test_corr))

if __name__ == "__main__":
    args = docopt(__doc__)
    if args['train']:
        train(args)
    elif args['test']:
        test(args)
