#!/usr/bin/python

"""
vocab.py: Vocabulary Generation

Usage:
    vocab.py [--snli-corpus=<file> --sts-coprpus=<file>] [options]

Options:
    -h --help                   show this screen.
    --snli-corpus=<file>        [default: ../data/snli_train.txt]
    --sts-corpus=<file>         [default: ../data/sts_train.txt]
    --freq-cutoff=<int>         frequency cutoff [default: 1]
    --save-vocab-to=<file>      save vocab object [default: vocab.pickle]
"""

from docopt import docopt
from collections import Counter
from itertools import chain
import pickle
import torch

from utils import read_corpus
from utils import pad_sents

class Vocab(object):
    """
    structure of the vocabulary
    """
    def __init__(self, word2id=None):
        """
        @param word2id (dict): dictionary mapping words->indices
        """
        self.pad_tok = '<pad>'      #Pad Token
        self.unk_tok = '<unk>'      #Unknown Token
        self.start_tok = '<start>'  #Start Token
        self.eos_tok = '<eos>'      #End of sentence Token
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id[self.pad_tok] = 0   
            self.word2id[self.unk_tok] = 1
            self.word2id[self.start_tok] = 2
            self.word2id[self.eos_tok] = 3

        self.pad_id = self.word2id[self.pad_tok]
        self.unk_id = self.word2id[self.unk_tok]
        self.start_id = self.word2id[self.start_tok]
        self.eos_id = self.word2id[self.eos_tok]

        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __len__(self):
        return len(self.word2id)

    def id2word(self, w_id):
        return self.id2word[w_id]

    def add(self, word):
        if word not in self.word2id:
            w_id = self.word2id[word] = len(self)
            self.id2word[w_id] = word
            return w_id
        else:
            return self[word]

    def words2indices(self, sents):
        if type(sents[0]) == str:
            return [self[w] for w in sents]
        else:
            return [[self[w] for w in s] for s in sents]

    def indices2words(self, w_ids):
        return [self.id2word[w_id] for w_id in w_ids]

    def sents2Tensor(self, sents, device):
        """
        Convert list of sent to tensor by padding required sents
        @param sents (list[list[str]]): batch of sents in reverse sorted order
        @return torch.t(out_tensor) (torch.tensor (max_sent_len, batch_size))
        """
        word_ids = self.words2indices(sents)
        sents_padded = pad_sents(word_ids, self['<pad>'])
        out_tensor = torch.tensor(sents_padded, dtype=torch.long, device=device)
        return torch.t(out_tensor) #transpose since batch_first=False in our model

    def save(self, file_path):
        """
        save the vocab to a json file
        @param file_path (str): /path/file to save the vocab
        """
        pickle.dump(self.word2id, open(file_path, 'wb'))

    @staticmethod
    def build(corpus, freq_cutoff):
        """
        create Vocab object for the words in the corpus
        @param corpus (list[list[str]]): list of sentences
        @param freq_cutoff (int): min frequency cutoff
        @return vocab (Vocab): Vocab class obj
        """
        vocab = Vocab()
        word_freq = Counter(chain(*corpus))
        vocab_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        top_vocab_words = sorted(vocab_words, key=lambda w: word_freq[w], reverse=True)
        for word in top_vocab_words:
            vocab.add(word)
        return vocab

    @staticmethod
    def load(file_path):
        """
        load vocabulary from the json dump
        @param file_path (str): /path/file for vocab build on corpus
        @return Vocab class obj loaded from this file_path
        """
        word2id = pickle.load(open(file_path, 'rb'))
        return Vocab(word2id)

if __name__ == "__main__":
    args = docopt(__doc__)

    snli_sents = read_corpus(args['--snli-corpus'])
    sts_sents = read_corpus(args['--sts-corpus'])
    train_sents = snli_sents
    train_sents.extend(sts_sents)
    vocab = Vocab.build(train_sents, int(args['--freq-cutoff']))
    vocab.save(args['--save-vocab-to'])
