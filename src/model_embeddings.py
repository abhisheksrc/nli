#!/usr/bin/python

"""
Embeddings for the neural model
"""

import torch.nn as nn

class ModelEmbeddings(nn.Module):

    def __init__(self, vocab, embed_size, embeddings):
        """
        Init the embedding layer
        @param vocab (Vocab): vocab object
        @param embed_size (int): embedding size
        @param embeddings (torch.tensor (len(vocab), word_dim)): pretrained embeddings
        """
        super(ModelEmbeddings, self).__init__()
        pad_idx = vocab.pad_id
        self.embed_size = embed_size
        self.embedding = nn.Embedding(len(vocab), self.embed_size, padding_idx=pad_idx)
        self.embedding.weight = nn.Parameter(embeddings)
        #if want to freeze pretrained embeddings
        #self.embedding.weight.requires_grad = False
    
    def forward(self, inputs):
        return self.embedding(inputs)
