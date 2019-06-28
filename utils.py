#!/usr/bin/python
import torch
import numpy as np

import nltk

def readCorpus(file_path):
    """
    read file containing label\tsentence1\tsentence2 followed by a '\n' to construct a List of sentences (List[str])
    @param file_path (str): path to file containing corpus
    @return data (List[List[str]]): list of sentences (containing tokens)
    """
    data = []
    for i, line in enumerate(open(file_path, 'r')):
        #skip header
        if i == 0:
            continue

        #handle new-line
        line = line.split('\n')[0]
        
        #extract sent
        sent1 = line.split('\t')[1]
        sent2 = line.split('\t')[2]

        #extract tokens
        sent1_tokens = nltk.word_tokenize(sent1)
        sent2_tokens = nltk.word_tokenize(sent2)

        data.append(sent1_tokens)
        data.append(sent2_tokens)
        
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
