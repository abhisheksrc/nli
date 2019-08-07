#!/usr/bin/python
"""
nli_train.py: Script for NLI Model
Abhishek Sharma <sharm271@cs.purdue.edu>

Usage:
    sts_train.py train --train-file=<file> --dev-file=<file> [options]
    sts_train.py test MODEL_PATH --test-file=<file> [options]

Options:
    -h --help                           show this screen.
    --train-file=<file>                 train_corpus
    --dev-file=<file>                   dev_corpus
    --test-file=<file>                  test_corpus
    --vocab-file=<file>                 vocab json [default: vocab.json]
    --word-embeddings=<file>            word_vecs [default: ../data/wiki-news-300d-1M.txt]
    --max-epoch=<int>                   max epoch [default: 15]
    --patience=<int>                    wait for how many epochs to exit training [default: 5]
    --batch-size=<int>                  batch size [default: 32]
    --embed-size=<int>                  word embed_dim [default: 300]
    --hidden-size=<int>                 hidden dim [default: 512]
    --mlp-hidden-size=<int>             mlp hidden dim [default: 1600]
    --num-layers=<int>                  number of layers [default: 3]
    --lr=<float>                        learning rate [default: 2e-4]
    --dropout=<float>                   dropout rate [default: 0.1]
    --save-model-to=<file>              save trained model [default: ../models/neural_sim.pt]
"""
from __future__ import division

import time
import math

import torch
import torch.nn.functional as F

from docopt import docopt

from utils import load_embeddings
from utils import extract_sents_result
from utils import batch_iter
from vocab import Vocab
from sts_model import NeuralSim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, data, batch_size):
    """
    Evaluate the model on the data
    @param model (NLIModel): NLI Model
    @param data (tuple(prem, hyp, label)): data returned by extract_sents_result
    @param batch_size (int): batch size
    @return avg_loss (float): avg. cross entropy loss on the data
    @return avg_acc (float): avg. classification accuracy on the data
    """
    was_training = model.training
    model.eval()

    total_loss = .0
    total_correct_preds = 0
    with torch.no_grad():
        for prems, hyps, labels in batch_iter(data, batch_size, shuffle=True, label=True):
            labels_pred = model(prems, hyps)

            P = F.log_softmax(labels_pred, dim=-1)
            labels_indices = labels_to_indices(labels)
            labels_indices = labels_indices.to(device) 
            cross_entropy_loss = torch.gather(P, dim=-1,
                index=labels_indices.unsqueeze(-1)).squeeze(-1)

            batch_loss = -cross_entropy_loss.sum()
            num_correct_preds = compareLabels(labels_pred, labels)

            batch_losses_val = batch_loss.item()
            total_loss += batch_losses_val
            total_correct_preds += num_correct_preds

    avg_loss = total_loss / len(data)
    avg_acc = total_correct_preds / len(data)
    
    if was_training:
        model.train()

    return avg_loss, avg_acc

def train(args):
    """
    train NLI model
    @param args (dict): command line args
    """
    vocab = Vocab.load(args['--vocab-file'])
    embeddings = load_embeddings(vocab, args['--word-embeddings'])

    #train Sim model
    train_data = extract_sents_result(args['--train-file'])
    dev_data = extract_sents_result(args['--dev-file'])

    train_batch_size = int(args['--batch-size'])
    dev_batch_size = int(args['--batch-size'])
    model_save_path = args['--save-model-to']

    model = NeuralSim(vocab, int(args['--embed-size']), embeddings,
                    hidden_size=int(args['--hidden-size']),
                    mlp_size=int(args['--mlp-hidden-size']),
                    dropout_rate=float(args['--dropout']))

    model = model.train()
    model = model.to(device)

    init_lr = float(args['--lr'])
    optimizer = torch.optim.Adam(model.parameters(), init_lr)

    total_loss = .0
    hist_dev_scores = []
    patience = 0

    begin_time = time.time()
    for epoch in range(int(args['--max-epoch'])):
        for sents1, sents2, results in batch_iter(train_data, batch_size=train_batch_size, shuffle=True, result=True):
            optimizer.zero_grad()
   
            results = torch.tensor(results, device=device)         
            results_pred = model(prems, hyps)

            loss = F.mse_loss(results_pred, results)
            loss.backward()

            optimizer.step()
            
            total_loss += loss.item()

        #print train loss at the end of each epoch
        train_loss = total_loss / len(train_data)
        print('epoch = %d: avg. loss = %.2f, time elapsed = %.2f sec'
            % (epoch, train_loss, time.time() - begin_time))
        total_loss = .0

        #perform validation
        dev_loss, dev_acc = evaluate(model, dev_data, dev_batch_size)
        is_better = epoch == 0 or dev_acc > max(hist_dev_scores)
        hist_dev_scores.append(dev_acc)

        if is_better:
            #reset patience
            patience = 0
            #save model
            model.save(model_save_path)

        else:
            patience += 1
            if patience == int(args['--patience']):
                print('finishing training: dev loss = %.2f, dev acc. = %.2f'
                    % (dev_loss, dev_acc))
                exit(0)

        print('validation: dev loss = %.2f, dev acc. = %.2f'
            % (dev_loss, dev_acc))
        
        #update lr after every 2 epochs
        lr = init_lr / 2 ** (epoch // 2)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def test(args):
    """
    test NLI model
    @param args (dict): command line args
    """
    test_data = extract_sents_result(args['--test-file'])
    model = SimModel.load(args['MODEL_PATH'])
    model = model.to(device)
    test_loss, test_acc = evaluate(model, test_data, batch_size=int(args['--batch-size']))
    print('final test accuracy= %.4f' %(test_acc))

if __name__ == "__main__":
    args = docopt(__doc__)
    if args['train']:
        train(args)
    elif args['test']:
        test(args)
