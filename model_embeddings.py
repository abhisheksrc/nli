#!/usr/bin/python

"""
Embeddings for the neural model
"""

import torch.nn as nn

class ModelEmbeddings(nn.Module):

    def __init__(self, vocab, embeddings):
        """
        Init the embedding layer
        @param vocab (Vocab): vocab object
        @param embeddings (torch.tensor (len(vocab), word_dim)): pretrained embeddings
        """
        super(ModelEmbeddings, self).__init__()
        pad_idx = vocab.pad_id
        self.embedding = nn.Embedding(embeddings.shape[0], embeddings.shape[1], padding_idx=pad_idx)
        self.embedding.weight = nn.Parameter(embeddings)
        
        #if want to freeze pretrained embeddings
        #self.embedding.weight.requires_grad = False
    
    def forward(self, inputs):
        return self.embedding(inputs)
