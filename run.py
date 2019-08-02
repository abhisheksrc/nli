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
    --save-generated-hyp-to=<file>      save generated hyp [default: _lg_result_att_longest.txt]              
    --vocab-file=<file>                 vocab json [default: vocab.json]
    --word-embeddings=<file>            word_vecs [default: ../data/wiki-news-300d-1M.txt]
    --max-epoch=<int>                   max epoch [default: 15]
    --patience=<int>                    wait for how many epochs to exit training [default: 5]
    --batch-size=<int>                  batch size [default: 32]
    --embed-size=<int>                  word embed_dim [default: 300]
    --hidden-size=<int>                 hidden dim [default: 256]
    --clip-grad=<float>                 grad clip [default: 5.0]
    --lr=<float>                        learning rate [default: 1e-3]
    --dropout=<float>                   dropout rate [default: 0.3]
    --beam-size=<int>                   beam size [default: 5]
    --max-decoding-time-step=<int>      max decoding time steps [default: 70]
    --save-model-to=<file>              save trained model [default: _model_att_b5_longest.pt]
"""
from __future__ import division

import time
import math

import torch
import torch.nn.functional as F

from docopt import docopt

from utils import loadEmbeddings
from utils import extractPairCorpus
from utils import batch_iter
from utils import save_generated_hyps
from vocab import Vocab

from neural_model import NeuralModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(args, data, model, vocab, embeddings):
    """ 
    Evaluate the model on the data
    @param args (dict): command line args
    @param data (list(tuple)): list of (prem, hyp) pairs
    @param model (NeuralModel): Neural Model
    @param vocab (Vocab): Vocab class obj
    @param embeddings (torch.tensor(len(vocab), embed_dim)): pretrained word embeddings
    @return gen_hyps (list[list[str]]): list of gen hyps
    @return eval_loss (float): eval loss on the data
    """
    was_training = model.training
    model.eval()
    
    gen_hyps = []
    data_hyps = []
    with torch.no_grad():
         for prem, hyp in data:
            gen_hyp = model.beam_search(prem, 
                beam_size=int(args['--beam-size']),
                max_decoding_time_step=int(args['--max-decoding-time-step']))
            gen_hyps.append(gen_hyp)
            data_hyps.append(hyp)

    if was_training:
        model.train()

    eval_data = []
    for gen_hyp, data_hyp in zip(gen_hyps, data_hyps):
        if len(gen_hyp) > 0 and len(data_hyp) > 0:
            eval_data.append((gen_hyp, data_hyp))

    num_empty_gen_hyp = len(gen_hyps) - len(eval_data)
    if num_empty_gen_hyp > 0:
        print('generated %d empty hypothesis' %(num_empty_gen_hyp))

    eval_loss = eval_avg_sim(eval_data, vocab, embeddings)
    return hyps, eval_loss

def eval_avg_sim(data, vocab, embeddings):
    """
    Eval cosine sim for sent pairs by averaging sent word embedding
    @param data (list(tuple)): list of (sent1, sent2) pairs
    @param vocab (Vocab): Vocab class obj
    @param embeddings (torch.tensor(len(vocab), embed_dim)): pretrained word embeddings
    @return loss (float): loss on sent similarity
    """
    total_loss = .0
    for sent1, sent2 in data:
        sent1_ids = vocab.words2indices(sent1)
        sent1_ids = torch.tensor(sent1_ids, dtype=torch.long, device=device)
        sent2_ids = vocab.words2indices(sent2)
        sent2_ids = torch.tensor(sent2_ids, dtype=torch.long, device=device)
        sent1_embed = embeddings[sent1_ids]
        sent1_embed = torch.mean(sent1_embed, dim=-1)
        sent2_embed = embeddings[sent2_ids]
        sent2_embed = torch.mean(sent2_embed, dim=-1)
        sim = F.cosine_similarity(sent1_embed, sent2_embed, dim=-1)
        loss = 1 - sim.item()
        total_loss += loss
    return total_loss / len(data)
        
def train_lg_model(args, vocab, embeddings, train_data, dev_data, label):
    """
    train LG model on the specific label
    @param args (dict): command line args
    @param vocab (Vocab): Vocab class obj
    @param embeddings (torch.tensor(len(vocab), embed_dim)): pretrained word embeddings
    @param train_data (list[tuple]): list of train (prem, hyp) pairs
    @param dev_data (lis(tuple)): list of dev (prem, hyp) pairs
    @param label (str): hyp label    
    """
    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    model_save_path = label + args['--save-model-to']

    model = NeuralModel(vocab, int(args['--embed-size']), embeddings,
                        hidden_size=int(args['--hidden-size']),
                        dropout_rate=float(args['--dropout']))
    model = model.to(device)

    init_lr = float(args['--lr'])
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

    total_loss = .0
    total_hyp_words = 0

    generated_hyp_path = label + args['--save-generated-hyp-to']

    hist_dev_losses = []
    patience = 0

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
        dev_hyps, dev_loss = evaluate(args, dev_data, model, vocab, embeddings)
        is_better = epoch == 0 or dev_loss < min(hist_dev_losses)
        hist_dev_losses.append(dev_loss)

        if is_better:
            #reset patience
            patience = 0
            #save model
            model.save(model_save_path)
            #save generated hyps
            save_generated_hyps(generated_hyp_path, dev_prems, dev_hyps)

        else:
            patience += 1
            if patience == int(args['--patience']):
                print('finishing training: dev loss = %.2f'
                    % (dev_loss))
                exit(0)

        print('validation: dev loss = %.2f'
            % (dev_loss))

        #update lr after every 2 epochs
        lr = init_lr / 2 ** (epoch // 2)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def train(args):
    """
    train neural models
    @param args (dict): command line args
    """
    vocab = Vocab.load(args['--vocab-file'])
    embeddings = loadEmbeddings(vocab, args['--word-embeddings'])
    embeddings = embeddings.to(device)

    #construct set of train sent pairs for each hyp class
    entail_pairs, neutral_pais, contradict_pairs = extractPairCorpus(args['--train-file'])
    #construct set of dev sent pars for each hyp class
    dev_entail_pairs, dev_neutral_pairs, dev_contradict_pairs = extractPairCorpus(args['--dev-file'])
    
    #train LG model for each hyp class
    train_lg_model(args, vocab, embeddings, train_data=entail_pairs, dev_data=dev_entail_pairs, label='entailment')

if __name__ == "__main__":
    args = docopt(__doc__)
    if args['train']:
        train(args)
