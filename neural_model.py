#!/usr/bin/python
import torch
import torch.nn as nn
from torch.nn.utils import rnn

from model_embeddings import ModelEmbeddings

class NeuralModel(nn.Module):
    """
    Bi-LSTM encoder
    Bi-LSTM decoder
    """
    def __init__(self, vocab, embeddings, hidden_size, num_layers, dropout_rate):
        """
        @param vocab (Vocab): vocab object
        @param embeddings (torch.tensor (len(vocab), embed_dim)): pretrained word embeddings
        @param hidden_size (int): hidden size dim
        @param num_layers (int): num layers
        @param dropout_rate (float): dropout prob
        """
        super(NeuralModel, self).__init__()
        self.embedding_size = embeddings.shape[1]
        self.embeddings = ModelEmbeddings(vocab, embeddings)
        self.device = self.embeddings.embedding.weight.device
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.encoder = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.num_layers, bias=True, dropout=self.dropout_rate, bidirectional=True)

    def forward(self, prems, hyps):
        """
        given a batch of prems and hyps, run encoder on prems to get hidden state init for decoder
        run decoder to generate the hyps and calculate log-likelihood of the words in the hyps
        @param prems (List[List[str]]): batches of premise (List[str])
        @param hyps (List[List[str]]): batches of hypothesis (List[str])
        @return scores (torch.Tensor(batch-size, )): log-likelihod of generating the words in the hyps
        """
        prems_lengths = [len(prem) for prem in prems]
        
        prems_padded = self.vocab.sents2Tensor(prems, device=self.device)

        enc_hiddens, dec_init_state = self.encode(prems_padded, prems_lengths)
        return dec_init_state

    def encode(self, prems, prems_lens):
        """
        apply the encoder on the premises to obtain encoder hidden states
        @param prems (torch.tensor(max_prem_len, batch-size))
        @param prem_lens (List[int]): list of actual lengths of the prems
        @return enc_hiddens (torch.tensor(max_prem_len, batch-size, 2*hidden-size)): tensor of seq of hidden outs
        @return dec_init_state (tuple(torch.tensor(num-layers, batch, 2*hidden-size), torch.tensor(num-layers, batch, 2*hidden-size))): encoders final hidden state and cell i.e. decoders initial hidden state and cell
        """
        X = self.embeddings(prems)
        X = rnn.pack_padded_sequence(X, prems_lens)
        enc_hiddens, (h_e, c_e) = self.encoder(X)
        enc_hiddens, prems_lens_tensor = rnn.pad_packed_sequence(enc_hiddens)
        batch = prems.shape[1]
        h_e_expand = h_e.view(self.num_layers, 2, batch, self.hidden_size) 
        c_e_expand = c_e.view(self.num_layers, 2, batch, self.hidden_size)
        h_e_cat = torch.cat((h_e_expand[:, 0, :, :], h_e_expand[:, 1, :, :]), dim=2).to(self.device)
        c_e_cat = torch.cat((c_e_expand[:, 0, :, :], c_e_expand[:, 1, :, :]), dim=2).to(self.device)
        return enc_hiddens, (h_e_cat, c_e_cat)
        
