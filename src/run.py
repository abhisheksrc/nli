#!/usr/bin/python
"""
run.py: run script for NLI LG Model
Abhishek Sharma <sharm271@cs.purdue.edu>

Usage:
    run.py train EVAL_MODEL --train-file=<file> --dev-file=<file> [options]
    run.py test ENTAIL_MODEL NEUTRAL_MODEL CONTRADICT_MODEL EVAL_MODEL --test-file=<file> [options]

Options:
    -h --help                           show this screen.
    --train-file=<file>                 train corpus
    --dev-file=<file>                   dev corpus
    --save-generated-hyp-to=<file>      save generated hyp [default: _lg_result.txt]
    --vocab-file=<file>                 snli vocab dump [default: snli-vocab.pickle]
    --pretrained-embeddings=<file>      word embeddings [default: snli-embeddings.pickle]
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
    --save-model-to=<file>              save trained model [default: _model_att.pt]
"""
from __future__ import division

import time
import math
import pickle

import torch
import torch.nn.functional as F

from docopt import docopt
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from utils import extract_pair_corpus
from utils import batch_iter
from utils import save_generated_hyps
from vocab import Vocab

from neural_model import NeuralModel
from sts_bilstm import BiLSTMSim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(args, data, model):
    """ 
    Evaluate the model by generating hyps
    using Sim model to compute the sim score
    and using BLEU metric for word overlap 
    between actual vs generated sents
    @param args (dict): command line args
    @param data (list[tuple]): list of (prem, hyp) pairs
    @param model (NeuralModel): trained Neural Model
    @return gen_hyps (list[list[str]]): list of gen hyps
    @return sim_score (float): sim score on the corpus
    @return bleu_score (float): BLEU score on the corpus
    """
    #gen hyps
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
            data_hyps.append(hyp[1:-1]) #exclude <start> and <eos>

    if was_training:
        model.train()

    eval_data = []
    for gen_hyp, data_hyp in zip(gen_hyps, data_hyps):
        if len(gen_hyp) > 0 and len(data_hyp) > 0:
            eval_data.append((gen_hyp, data_hyp))

    num_empty_gen_hyp = len(gen_hyps) - len(eval_data)
    if num_empty_gen_hyp > 0:
        print('generated empty hypothesis = %d' %(num_empty_gen_hyp))

    #compute sim score
    sim_model = BiLSTMSim.load(args['EVAL_MODEL'])
    sim_model = sim_model.to(device)
    sim_model.eval()

    batch_size = int(args['--batch-size'])
    total_sim_score = .0
    with torch.no_grad():
        for sents1, sents2 in batch_iter(eval_data, batch_size):
            #sim score for this batch (torch.tensor(b,))
            sim_scores = sim_model(sents1, sents2)
            total_sim_score += torch.sum(sim_scores).item()

    sim_score = total_sim_score / len(eval_data)

    #compute bleu score
    bleu_score = compute_bleu_score(data_hyps, gen_hyps)

    return gen_hyps, sim_score, bleu_score
    
def compute_bleu_score(refs, hyps):
    """
    compute bleu score for the given references against generated hyps
    @param refs (list[list[str]]): list of reference sents
    @param hyps (list[list[str]]): list of generated candidate sents
    @return bleu_score (float): BLEU score for the word overlap
    """
    bleu_score = corpus_bleu([[ref] for ref in refs],
                            hyps, weights=(0.5, 0.5, 0, 0),
                            smoothing_function=SmoothingFunction().method4)
    return bleu_score

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

    dev_prems = [prem for (prem, hyp) in dev_data]
    save_gen_hyp_path = label + args['--save-generated-hyp-to']

    hist_dev_scores = []
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
        dev_hyps, sim_score, bleu_score = evaluate(args, dev_data, model)
        is_better = epoch == 0 or sim_score > max(hist_dev_scores)
        hist_dev_scores.append(sim_score)

        if is_better:
            #reset patience
            patience = 0
            #save model
            model.save(model_save_path)
            #save generated hyps
            save_generated_hyps(save_gen_hyp_path, dev_prems, dev_hyps)

        else:
            patience += 1
            if patience == int(args['--patience']):
                print('finishing training: dev sim score = %.2f, BLEU score = %.2f'
                    % (sim_score, bleu_score))
                return

        print('validation: dev sim score = %.2f, BLEU score = %.2f'
            % (sim_score, bleu_score))

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
    embeddings = pickle.load(open(args['--pretrained-embeddings'], 'rb'))
    embeddings = torch.tensor(embeddings, dtype=torch.float, device=device)

    #construct set of train sent pairs for each hyp class
    entail_pairs, neutral_pairs, contradict_pairs = extract_pair_corpus(args['--train-file'])
    #construct set of dev sent pars for each hyp class
    dev_entail_pairs, dev_neutral_pairs, dev_contradict_pairs = extract_pair_corpus(args['--dev-file'])
    
    #train LG model for each hyp class
    print('--------entailment--------')
    train_lg_model(args, vocab, embeddings, train_data=entail_pairs, dev_data=dev_entail_pairs, label='entailment')
    print('--------neutral--------')
    train_lg_model(args, vocab, embeddings, train_data=neutral_pairs, dev_data=dev_neutral_pairs, label='neutral')
    print('--------contradiction--------')
    train_lg_model(args, vocab, embeddings, train_data=contradict_pairs, dev_data=dev_contradict_pairs, label='contradiction')

def test(args):
    """
    test models for each label
    @param args (dict): command line args
    """
    entail_pairs, neutral_pairs, contradict_pairs = extract_pair_corpus(args['--test-file'])
    labels = ['entail', 'neutral', 'contradict']
    for i, label in enumerate(labels):
        if label == 'entail':
            data = entail_pairs
        elif label == 'neutral':
            data = neutral_pairs
        elif label == 'contradict':
            data = contradict_pairs
        
        save_gen_hyp_path = label + '_test' + args['--save-generated-hyp-to']
        prems = [prem for (prem, hyp) in data]
        model_path = label.upper() + '_MODEL'
        model = NeuralModel.load(args[model_path]) 
        model = model.to(device)
        gen_hyps, sim_score, bleu_score = evaluate(args, data, model)
        save_generated_hyps(save_gen_hyp_path, prems, gen_hyps)
        print('%s sim score = %.2f, BLEU score = %.2f' % (label, sim_score, bleu_score))

if __name__ == "__main__":
    args = docopt(__doc__)
    if args['train']:
        train(args)
    elif args['test']:
        test(args)
