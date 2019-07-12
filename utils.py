#!/usr/bin/python
from __future__ import division
import torch
import numpy as np
import math

import nltk



def readLine(line):
    """
    read line of format label\tsentence1\tsentence2\n
    @param line (str)
    @return label, sent1, sent2 (tuple(str, List[str], List[str]))
    """
    line = line.split('\n')[0]
    line = line.split('\t')
    
    label = line[0]
    sent1 = nltk.word_tokenize(line[1])
    sent2 = nltk.word_tokenize(line[2])

    return label, sent1, sent2

def readCorpus(file_path):
    """
    read file to construct a List of sentences (List[str])
    @param file_path (str): path to file containing corpus
    @return data (List[List[str]]): list of sentences (containing tokens)
    """
    data = []
    for line in open(file_path, 'r'):
        label, sent1, sent2 = readLine(line)

        data.append(sent1)
        data.append(sent2)
        
    return data

def loadEmbeddings(vocab, embedding_file, device):
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

    embedding_weights = torch.tensor(weights, dtype=torch.float, device=device)
    return embedding_weights

def extractPairCorpus(file_path):
    """
    build list of (hyp, prem) for each label
    @param file_path (str): /path/corpus
    @return entail_pairs, neutral_pairs, contradict_pairs (List[tuple(List[str])])
    """
    entail_pairs, neutral_pairs, contradict_pairs = [], [], []
    for line in open(file_path, 'r'):
        label, sent1, sent2 = readLine(line)
        sent2 = ['<start>'] + sent2 + ['<eos>']
        if label == 'entailment':
            entail_pairs.append((sent1, sent2))
        elif label == 'neutral':
            neutral_pairs.append((sent1, sent2))
        else:
            contradict_pairs.append((sent1, sent2))
    return entail_pairs, neutral_pairs, contradict_pairs

def batch_iter(data, batch_size, shuffle=True):
    """
    yield batches of premise and hypothesis reverse sorted by length
    @param data (List[tuple]): list of tuples (premise, hypothesis)
    @param batch_size (int)
    @param shuffle (boolean): option randomly shuffle data
    """
    batches = int(math.ceil(len(data) / batch_size))
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batches):
        slice_ = index_array[i * batch_size: (i + 1) * batch_size]
        batch = [data[j] for j in slice_]

        batch = sorted(batch, key=lambda b: len(b[0]), reverse=True)
        prems = [b[0] for b in batch]
        hyps = [b[1] for b in batch]

        yield prems, hyps

def padSents(sents, pad_idx):
    """
    @param sents (List[List[int]]):
    @param pad_idx (int): Pad ID
    @return sents_padded (List[List[int]]): sents with padding according to max_sent_len
    """
    max_len = 0
    sents_padded = []
    for sent in sents: max_len = max(max_len, len(sent))
    for sent in sents:
        sent_padded = sent
        sent_padded.extend([pad_idx for i in range(max_len - len(sent))])
        sents_padded.append(sent_padded)

    return sents_padded

def save(model_dict, file_path):
    """
    @param model_dict (Dict): model.state_dict()
    @param file_path (str)
    """
    torch.save(model_dict, file_path)
