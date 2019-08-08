#!/usr/bin/python
from __future__ import division
import torch
import numpy as np
import math

import nltk

def read_line(line):
    """
    read line of format result\tsent1\tsent2\n
    @param line (str)
    @return result, sent1, sent2 (tuple(str, list[str], list[str]))
    """
    line = line.split('\n')[0]
    line = line.split('\t')
    
    result = line[0]
    sent1 = nltk.word_tokenize(line[1])
    sent2 = nltk.word_tokenize(line[2])

    return result, sent1, sent2

def read_corpus(file_path):
    """
    read file to construct a list of sentences (list[str])
    @param file_path (str): path to file containing corpus
    @return data (list[list[str]]): list of sentences (containing tokens)
    """
    data = []
    for line in open(file_path, 'r'):
        result, sent1, sent2 = read_line(line)

        data.append(sent1)
        data.append(sent2)
        
    return data

def load_embeddings(vocab, embedding_file):
    """
    construct vector for word embeddings
    loads embedding from embedding_file
    @param vocab (Vocab): obj from Vocab class
    @param embedding_file (str): /path/file/containing_embedding
    @return embedding_weights (torch.tensor (len(vocab), word_dim))
    """
    embedding_words_vecs = np.loadtxt(embedding_file, dtype='str', comments=None)
    words = embedding_words_vecs[:, 0]
    vecs = embedding_words_vecs[:, 1:].astype('float').tolist()
    word_dim = len(vecs[0])

    #initialize weights
    init_vocab = len(vocab)
    weights = [None]*init_vocab
    weights[vocab[vocab.pad_tok]] = np.zeros(word_dim).tolist()

    for i, word in enumerate(words):
        if word in vocab.word2id:
            weights[vocab[word]] = vecs[i]

    #check if any word embedding is still None
    for i in range(init_vocab):
        if not weights[i]:
            weights[i] = np.random.rand(word_dim).tolist()

    embedding_weights = torch.tensor(weights, dtype=torch.float)
    return embedding_weights

def extract_pair_corpus(file_path):
    """
    build list of (prem, hyp) for each label
    @param file_path (str): /path/corpus
    @return entail_pairs, neutral_pairs, contradict_pairs (list[tuple(list[str])])
    """
    entail_pairs, neutral_pairs, contradict_pairs = [], [], []
    for line in open(file_path, 'r'):
        label, prem, hyp = read_line(line)
        hyp = ['<start>'] + hyp + ['<eos>']
        if label == 'entailment':
            entail_pairs.append((prem, hyp))
        elif label == 'neutral':
            neutral_pairs.append((prem, hyp))
        elif label == 'contradiction':
            contradict_pairs.append((prem, hyp))
    return entail_pairs, neutral_pairs, contradict_pairs

def extract_sents_result(file_path):
    """
    build list of (sent1, sent2, result)
    @param file_path (str): /path/corpus
    @return data (list[tuple(sent1, sent2, result)])
    """
    data = []
    for line in open(file_path, 'r'):
        score, sent1, sent2 = read_line(line)
        score = eval(score)
        data.append((sent1, sent2, score)
    return data

def batch_iter(data, batch_size, shuffle=True, result=False):
    """
    yield batches of sent1, sent2 and result(optional) reverse sorted by sent1 length
    @param data (list[tuple]): list of tuples (sent1, sent2, result(optional))
    @param batch_size (int)
    @param shuffle (boolean): option randomly shuffle data
    @param result (boolean): if optional result also present in data tuples
    """
    batches = int(math.ceil(len(data) / batch_size))
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batches):
        slice_ = index_array[i * batch_size: (i + 1) * batch_size]
        batch = [data[j] for j in slice_]

        batch = sorted(batch, key=lambda b: len(b[0]), reverse=True)
        sents1 = [b[0] for b in batch]
        sents2 = [b[1] for b in batch]
        if not result:
            yield sents1, sents2
        else:
            results = [b[2] for b in batch]
            yield sents1, sents2, results

def pad_sents(sents, pad_idx):
    """
    @param sents (list[list[int]]):
    @param pad_idx (int): Pad ID
    @return sents_padded (list[list[int]]): sents with padding according to max_sent_len
    """
    max_len = 0
    sents_padded = []
    for sent in sents: max_len = max(max_len, len(sent))
    for sent in sents:
        sent_padded = sent
        sent_padded.extend([pad_idx for i in range(max_len - len(sent))])
        sents_padded.append(sent_padded)

    return sents_padded

def sort_sents(sents):
    """
    reverse sort the sents, criteria: length
    @param sents (list[list[str]])
    @return sents_sorted (list[list[str]])
    @return orig_to_sorted (dict): mapping sents_indices_orig->sents_indices_sorted
    """
    sents_indices = []
    for i, sent in enumerate(sents):
        sents_indices.append((sent, i))
    sents_indices.sort(key=lambda (sent, index): len(sent), reverse=True)

    orig_to_sorted = {}
    sents_sorted = []
    for i, (sent, index) in enumerate(sents_indices):
        orig_to_sorted[index] = i
        sents_sorted.append(sent)

    return sents_sorted, orig_to_sorted

def save_generated_hyps(file_path, prems, hyps):
    """
    save each generated hyp by the Neural model for each given prem
    @param file_path (str): /path/save/generated/hyp
    @param prems (list[list[str]]): given prems
    @param hyps (list[list[str]]): generated hyps
    """
    with open(file_path, 'w') as file_obj:
        for prem, hyp in zip(prems, hyps):
            file_obj.write(' '.join(prem) + '\t' + ' '.join(hyp) + '\n')
