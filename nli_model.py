#!/usr/bin/python
import torch
import torch.nn as nn
from torch.nn.utils import rnn
import torch.nn.functional as F

from utils import sortHyps
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
        self.embeddings = ModelEmbeddings(vocab, embed_size, embeddings)
        self.device = device
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.encoder = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, num_layers=self.num_layers, bias=True, bidirectional=True)

        #classifier for 3 possible labels
        self.classifier = nn.Linear(in_features=self.hidden_size*4, out_features=3)

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

        #reverse sort hyps and save original lengths and index mapping:
        #   map: indices_sorted -> indices_orig
        hyps_lengths_orig = [len(hyp) for hyp in hyps]
        hyps_sorted, index_map = sortHyps(hyps)
        hyps_lengths_sorted = [len(hyp) for hyp in hyps_sorted]
        hyps_padded = self.vocab.sents2Tensor(hyps_sorted, device=self.device)

        hyps_enc_outs = self.encode(hyps_padded, hyps_lengths_sorted)
        hyps_enc_outs_seq_orig = [hyps_enc_outs[:, index_map[i], :].unsqueeze(dim=1) 
                                    for i in range(len(hyps))]
        hyps_enc_outs_orig = torch.cat(hyps_enc_outs_seq_orig, dim=1)

        #max pooling and classifier
        prems_encoding_final = self.maxPool(prems_enc_outs, prems_lengths)
        hyps_encoding_final = self.maxPool(hyps_enc_outs_orig, hyps_lengths_orig)
        
        ins_classifier = torch.cat((prems_encoding_final, hyps_encoding_final), dim=-1)
        
        outs = self.classifier(ins_classifier)
        return outs 

    def encode(self, sents, sents_lens):
        """
        apply the encoder on the sentences to obtain encoder hidden states
        @param prems (torch.tensor(max_sent_len, batch))
        @param sents_lens (List[int]): list of actual lengths of the sents
        @return enc_hiddens (torch.tensor(max_sent_len, batch, hidden*2)): 
            tensor of seq of hidden outs
        """
        X = self.embeddings(sents)
        X = rnn.pack_padded_sequence(X, sents_lens)
        enc_hiddens, (h_n, c_n) = self.encoder(X)
        enc_hiddens, sents_lens_tensor = rnn.pad_packed_sequence(enc_hiddens)
        return enc_hiddens

    def maxPool(self, encodings, lengths):
        """
        apply max pool to each encoding to extract the max element
        @param encodings (torch.tensor(max_encoding_len, batch, hidden*2)):
            the out sequence of encoder
        @param lengths (List[int]): list of actual lengths of the encodings
        @return outs_encoder_final (torch.tensor(batch, hidden*2)):
            final out of the encoder
        """
        seq_max_list = []
        for i, length in enumerate(lengths):
            seq_i = encodings[:length, i, :]
            seq_i_max, _ = seq_i.max(dim=0)
            seq_i_max = torch.squeeze(seq_i_max, dim=0)
            seq_max_list.append(seq_i_max)
        return torch.stack(seq_max_list)

    def save(self, file_path):
        """
        saving model to the file_path
        """
        params = {
            'vocab' : self.vocab,
            'args' : dict(embed_size=self.embeddings.embed_size, 
                        embeddings=self.embeddings,
                        hidden_size=self.hidden_size, num_layers=self.num_layers,
                        dropout_rate=self.dropout_rate, device=self.device),
            'state_dict': self.state_dict()      
        }
        torch.save(params, file_path)

    @staticmethod
    def load(model_path):
        """
        load a saved neural model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NLIModel(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])
