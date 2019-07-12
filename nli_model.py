#!/usr/bin/python
import torch
import torch.nn as nn
from torch.nn.utils import rnn
import torch.nn.functional as F

from model_embeddings import ModelEmbeddings

class NLIModel(nn.Module):
    """
    Bi-LSTM encoder
    max pooling
    classifier
    """
    def __init__(self, vocab, embed_size, embeddings, hidden_size, num_layers, dropout_rate, device):
        """
        @param vocab (Vocab): vocab object
        @param embed_size (int): embedding size
        @param embeddings (torch.tensor (len(vocab), embed_dim)): pretrained word embeddings
        @param hidden_size (int): hidden size
        @param num_layers (int): num layers
        @param dropout_rate (float): dropout prob
        """
        super(NLIModel, self).__init__()
        self.embed_size = embed_size
        self.embeddings = ModelEmbeddings(vocab, self.embed_size, embeddings)
        self.device = device
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.encoder = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, num_layers=self.num_layers, bias=True, bidirectional=True)

        #classifier for 3 possible labels
        self.classifier = nn.Linear(in_features=self.hidden_size*2, out_features=3)

    def forward(self, prems, hyps):
        """
        given a batch of prems and hyps, run encoder on prems and hyps
        followed by max-pooling
        and finally pass through the classifier
        @param prems (List[List[str]]): batches of premise (List[str])
        @param hyps (List[List[str]]): batches of hypothesis (List[str])
        @return outs (torch.Tensor(batch-size, 3)): log-likelihod of 3-labels
        """
        prems_lengths = [len(prem) for prem in prems]
        prems_padded = self.vocab.sents2Tensor(prems, device=self.device)

        prems_enc_outs = self.encode(prems_padded, prems_lengths)

        #reverse sort hyps and save original indices
        hyps_indices = []
        for i, hyp in enumerate(hyps):
            hyps_indices.append((hyp, i))
        hyps_indices.sort(key=lambda (hyp, index): len(hyp), reverse=True)

        index_map = {}
        hyps_sorted = []
        for i, (hyp, index) in enumerate(hyps_indices):
            index_map[i] = index
            hyps_sorted.append(hyp)

        hyps_lengths = [len(hyp) for hyp in hyps_sorted]
        hyps_padded = self.vocab.sents2Tensor(hyps_sorted, device=self.device)

        hyps_enc_outs = self.encode(hyps_padded, hyps_lengths)
        hyps_enc_outs_seq_orig = [hyps_enc_outs[:, index_map[i], :].unsqueeze(dim=1) 
                                    for i in range(len(hyps))]
        hyps_enc_outs_orig = torch.cat(hyps_enc_outs_seq_orig, dim=1)
        #TODO max pooling and classifier 

    def encode(self, sents, sents_lens):
        """
        apply the encoder on the sentences to obtain encoder hidden states
        @param prems (torch.tensor(max_sent_len, batch))
        @param sents_lens (List[int]): list of actual lengths of the sents
        @return enc_hiddens (torch.tensor(max_sent_len, batch, hidden*2)): tensor of seq of hidden outs
        """
        X = self.embeddings(sents)
        X = rnn.pack_padded_sequence(X, sents_lens)
        enc_hiddens, (h_n, c_n) = self.encoder(X)
        enc_hiddens, sents_lens_tensor = rnn.pad_packed_sequence(enc_hiddens)
        return enc_hiddens
