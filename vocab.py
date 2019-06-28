#!/usr/bin/python
from collections import Counter
from itertools import chain
import torch

from utils import tagEOS
from utils import padSents

class Vocab(object):
    """
    structure of the vocabulary
    """
    def __init__(self, word2id=None):
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

    def sents2Tensor(self, sents, device, test=False):
        """
        @param sents (List[List[str]]) if not test: list of variable sent lens
                else sents (List[str]) 
        pad to standardize sent lens according to max_sent_len
        convert words to indices and then form a tensor for List[List[int]]
        @return torch.t(out_tensor) (torch.tensor (max_sent_len, batch_size))
        """
        if not test:
            sents_padded = padSents(sents, self.pad_tok)
            word_ids = self.words2indices(sents_padded)
        else:
            word_ids = [self.words2indices(sents)]
        out_tensor = torch.tensor(word_ids, dtype=torch.long, device=device)
        return torch.t(out_tensor) #transpose since batch_first=False in our model

    @staticmethod
    def build(data, freq_cutoff=2):
        """
        create Vocab object build on the data
        @param data (List[List[str]])
        """
        vocab = Vocab()
        word_freq = Counter(chain(*data))
        vocab_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        top_vocab_words = sorted(vocab_words, key=lambda w: word_freq[w], reverse=True)
        for word in top_vocab_words:
            vocab.add(word)
        return vocab
