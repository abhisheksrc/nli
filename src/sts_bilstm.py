#!/usr/bin/python
import torch
import torch.nn as nn
from torch.nn.utils import rnn
import torch.nn.functional as F

from utils import sort_sents
from model_embeddings import ModelEmbeddings

class BiLSTMSim(nn.Module):
    """
    Bi-LSTM encoder
    """
    def __init__(self, vocab, embed_size, embeddings, hidden_size, num_layers, mlp_size, dropout_rate, sim_scale=5):
        """
        @param vocab (Vocab): vocab object
        @param embed_size (int): embedding size
        @param embeddings (torch.tensor (len(vocab), embed_dim)): pretrained word embeddings
        @param hidden_size (int): hidden size
        @param mlp_size (int): mlp size
        @param dropout_rate (float): dropout prob
        @param sim_scale (float): scale the sim score by this scalar
        """
        super(BiLSTMSim, self).__init__()
        self.pretrained_embeddings = embeddings
        self.embeddings = ModelEmbeddings(vocab, embed_size, self.pretrained_embeddings)
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.mlp_size = mlp_size
        self.dropout_rate = dropout_rate
        self.sim_scale = sim_scale

        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=self.hidden_size, num_layers=self.num_layers, bias=True, bidirectional=True)

        #in_features = h * 2 (bidirectional) * num_layers * 2 (sent1-sent2 feature concat)
        self.mlp = nn.Linear(in_features=self.hidden_size*2*self.num_layers*2, out_features=self.mlp_size)
        self.final_layer = nn.Linear(in_features=self.mlp_size, out_features=1)
        self.scoring_fn = nn.Sequential(*[self.mlp, nn.Tanh(), nn.Dropout(self.dropout_rate),
                                        self.final_layer, nn.Sigmoid()]) 

    def forward(self, sents1, sents2):
        """
        given a batch of sent1 and sent2, run encoder on sent1 and sent2
        con-cat final encodings of both sents
        and finally pass through the scoring function
        @param sents1 (list[list[str]]): batch of sent1 (list[str])
        @param sents2 (list[list[str]]): batch of sent2 (list[str])
        @return scores (torch.Tensor(batch-size,)): sim scores for the batch of (sent1, sent2)
        """
        sent1_lengths = [len(sent1) for sent1 in sents1]
        sent1_padded = self.vocab.sents2Tensor(sents1, device=self.device)

        sent1_enc_final = self.encode(sent1_padded, sent1_lengths)

        #reverse sort sents2 and save index mapping: orig -> sorted, and orig length
        sent2_lengths_orig = [len(sent2) for sent2 in sents2]
        sent2_sorted, orig_to_sorted = sort_sents(sents2)
        sent2_lengths_sorted = [len(sent2) for sent2 in sent2_sorted]
        sent2_padded = self.vocab.sents2Tensor(sent2_sorted, device=self.device)

        sent2_enc_out = self.encode(sent2_padded, sent2_lengths_sorted)
        sent2_enc_out_orig = [sent2_enc_out[orig_to_sorted[i], :].unsqueeze(dim=0)
                                    for i in range(len(sents2))]
        sent2_enc_final = torch.cat(sent2_enc_out_orig, dim=0)

        scoring_features = torch.cat((sent1_enc_final, sent2_enc_final), dim=-1)
        
        scores = self.scoring_fn(scoring_features).squeeze(dim=-1)
        #scale the scores by sim_scale
        scores = scores * self.sim_scale
        return scores 

    def encode(self, sents, sents_lens):
        """
        apply the encoder on the sentences to obtain final out from all layers
        @param sents (torch.tensor(max_sent_len, batch))
        @param sents_lens (list[int]): list of actual lengths of the sents
        @return final_out (torch.tensor(batch, hidden*2*num_layers)): 
            the con-cat last outs from all the layers
        """
        batch = X.shape[1]
        X = self.embeddings(sents)
        X = rnn.pack_padded_sequence(X, sents_lens)
        out_layer, (h_n, c_n) = self.encoder(X)

        #expand h_n
        h_n = h_n.view(self.num_layers, 2, batch, self.hidden_size)
        #con-cat last outs from all the layers
        last_outs = []
        for i in range(self.num_layers):
            h_n_i = torch.cat((h_n[i, 0, :, :], h_n[i, 1, :, :]), dim=-1)
            last_outs.append(h_n_i)

        final_out = torch.cat(last_outs, dim=-1)

        #con-cat last outs from the last layer
        #h_n = torch.cat((h_n[-1, 0, :, :], h_n[-1, 1, :, :]), dim=-1)

        return final_out

    def save(self, file_path):
        """
        saving model to the file_path
        """
        params = {
            'vocab' : self.vocab,
            'args' : dict(embed_size=self.embeddings.embed_size, 
                        embeddings=self.pretrained_embeddings,
                        hidden_size=self.hidden_size, 
                        num_layers=self.num_layers, mlp_size=self.mlp_size,
                        dropout_rate=self.dropout_rate),
            'state_dict': self.state_dict()      
        }
        torch.save(params, file_path)

    @property
    def device(self):
        """
        property decorator for devive
        """
        return self.embeddings.embedding.weight.device

    @staticmethod
    def load(model_path):
        """
        load a saved neural model
        @param model_path (str): path to model
        @return model (BiLSTMSim)
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = BiLSTMSim(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model
