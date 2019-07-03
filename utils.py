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
    also adds new words to the vocab
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
        else:
            wid=vocab.add(word)
            weights.append(vecs[i])

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

def readTest(file_path):
    """
    read file containing sentences delineated by a '\n' to construct a List of words
    @param file_path (str): path to file containing test data
    @return data (List[str]): list of words
    """
    data = []
    for line in open(file_path):
        word = line.strip().split(' ')[0]
        data.append(word)

    return data

def readTestSent(file_path):
    """
    read file containing sentences delineated by a '\n' to construct a List of sents
    @param file_path (str): path to file containing test data
    @return data (List[List[str]]): list of sents
    """
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        data.append(sent)

    return data

def tagEOS(sents, eos_tok):
    """
    @param sents (List[List[str]]) : rev sorted sents (max_len)
    @param eos_tok (str) : EOS Token
    @return sents (List[List[str]]) : sents with EOS
    """
    temp = sents
    sents = []
    for sent in temp:
        sent.append(eos_tok)
        sents.append(sent)
    return sents

def padSents(sents, pad_tok):
    """
    @param sents (List[List[str]]) : rev sorted sents (max_len) with added EOS
    @param pad_tok (str) : Pad Token
    @return sents_padded (List[List[str]]) : sents with padding according to max_sent_len
    """
    sents_padded = []
    max_len = len(sents[0])
    for sent in sents:
        sent_padded = sent
        sent_padded.extend([pad_tok for i in range(max_len - len(sent))])
        sents_padded.append(sent_padded)

    return sents_padded

def save(model_dict, file_path):
    """
    @param model_dict (Dict) : model.state_dict()
    @param file_path (str)
    """
    torch.save(model_dict, file_path)
