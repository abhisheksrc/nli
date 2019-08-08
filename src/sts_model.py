#!/usr/bin/python
import torch
import torch.nn as nn
from torch.nn.utils import rnn
import torch.nn.functional as F

from utils import sort_sents
from model_embeddings import ModelEmbeddings

class NeuralSim(nn.Module):
    """
    Bi-LSTM encoder
    max pooling
    score function
    """
    def __init__(self, vocab, embed_size, embeddings, hidden_size, mlp_size, dropout_rate, sim_scale=5):
        """
        @param vocab (Vocab): vocab object
        @param embed_size (int): embedding size
        @param embeddings (torch.tensor (len(vocab), embed_dim)): pretrained word embeddings
        @param hidden_size (int): hidden size
        @param mlp_size (int): mlp size
        @param dropout_rate (float): dropout prob
        @param sim_scale (float): scale the sim score by this scalar
        """
        super(NeuralSim, self).__init__()
        self.pretrained_embeddings = embeddings
        self.embeddings = ModelEmbeddings(vocab, embed_size, self.pretrained_embeddings)
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.mlp_size = mlp_size
        self.dropout_rate = dropout_rate
        self.sim_scale = sim_scale

        #self.encoder = nn.LSTM(input_size=embed_size, hidden_size=self.hidden_size, num_layers=self.num_layers, bias=True, bidirectional=True)
        self.lstm_1 = nn.LSTM(input_size=embed_size, hidden_size=self.hidden_size, num_layers=1, bias=True, bidirectional=True)
        self.lstm_2 = nn.LSTM(input_size=(embed_size + self.hidden_size*2), hidden_size=self.hidden_size*2, num_layers=1, bias=True, bidirectional=True)
        self.lstm_3 = nn.LSTM(input_size=(embed_size + self.hidden_size*2 + self.hidden_size*4), hidden_size=self.hidden_size*4, num_layers=1, bias=True, bidirectional=True)

        #in_features = final lstm out_size * 2 (bidirectional) * 4 (sent1-sent2 feature concat)
        self.mlp_1 = nn.Linear(in_features=self.hidden_size*4*2*4, out_features=self.mlp_size)
        self.mlp_2 = nn.Linear(in_features=self.mlp_size, out_features=self.mlp_size)
        self.final_layer = nn.Linear(in_features=self.mlp_size, out_features=1)
        self.scoring_fn = nn.Sequential(*[self.mlp_1, nn.ReLU(), nn.Dropout(self.dropout_rate),
                                        self.mlp_2, nn.ReLU(), nn.Dropout(self.dropout_rate),
                                        self.final_layer, nn.Sigmoid()]) 

    def forward(self, sents1, sents2):
        """
        given a batch of sent1 and sent2, run encoder on sent1 and sent2
        followed by max-pooling
        and finally pass through the scoring function
        @param sents1 (list[list[str]]): batch of sent1 (list[str])
        @param sents2 (list[list[str]]): batch of sent2 (list[str])
        @return scores (torch.Tensor(batch-size,)): sim scores for the batch of (sent1, sent2)
        """
        sent1_lengths = [len(sent1) for sent1 in sents1]
        sent1_padded = self.vocab.sents2Tensor(sents1, device=self.device)

        sent1_enc_out = self.encode(sent1_padded, sent1_lengths)

        #reverse sort sents2 and save index mapping: orig -> sorted, and orig length
        sent2_lengths_orig = [len(sent2) for sent2 in sents2]
        sent2_sorted, orig_to_sorted = sort_sents(sents2)
        sent2_lengths_sorted = [len(sent2) for sent2 in sent2_sorted]
        sent2_padded = self.vocab.sents2Tensor(sent2_sorted, device=self.device)

        sent2_enc_out = self.encode(sent2_padded, sent2_lengths_sorted)
        sent2_enc_out_seq_orig = [sent2_enc_out[:, orig_to_sorted[i], :].unsqueeze(dim=1) 
                                    for i in range(len(sents2))]
        sent2_enc_out_orig = torch.cat(sent2_enc_out_seq_orig, dim=1)

        #applying max pooling and scoring function
        sent1_enc_final = self.max_pool(sent1_enc_out, sent1_lengths)
        sent2_enc_final = self.max_pool(sent2_enc_out_orig, sent2_lengths_orig)
        
        scoring_features = torch.cat([sent1_enc_final, sent2_enc_final, torch.abs(sent1_enc_final - sent2_enc_final), sent1_enc_final * sent2_enc_final], dim=-1)
        
        scores = self.scoring_fn(scoring_features)
        #scale the scores by sim_scale
        scores = scores * self.sim_scale
        return scores 

    def encode(self, sents, sents_lens):
        """
        apply the encoder on the sentences to obtain encoder hidden states
        @param sents (torch.tensor(max_sent_len, batch))
        @param sents_lens (list[int]): list of actual lengths of the sents
        @return out_layer_3 (torch.tensor(max_sent_len, batch, hidden*4*2)): 
            the final out from the laste lstm layer
        """
        X = self.embeddings(sents)
        out_layer_1 = self.run_BiLSTM(self.lstm_1, X, sents_lens)

        in_layer_2 = torch.cat([X, out_layer_1], dim=-1)
        out_layer_2 = self.run_BiLSTM(self.lstm_2, in_layer_2, sents_lens)

        in_layer_3 = torch.cat([X, out_layer_1, out_layer_2], dim=-1)
        out_layer_3 = self.run_BiLSTM(self.lstm_3, in_layer_3, sents_lens)

        return out_layer_3

    def run_BiLSTM(self, lstm_model, in_layer, sents_lens):
        """
        run BiLSTM on the input
        @param lstm_model (nn.LSTM): LSTM model
        @param in_layer (torch.tensor(max_sent_len, batch, embed + hidden*((2 ** i) - 2))): input layer for the model, i = layer#
        @param sents_lens (list[int]): list of actual lengths of the sents
        @return out_layer (torch.tensor(max_sent_len, batch, hidden*2*i)): output layer of the model, i = layer#
        """
        in_layer = rnn.pack_padded_sequence(in_layer, sents_lens)
        out_layer, (h_n, c_n) = lstm_model(in_layer)
        out_layer, sents_lens_tensor = rnn.pad_packed_sequence(out_layer)
        return out_layer

    def max_pool(self, encodings, lengths):
        """
        apply max pool to each encoding to extract the max element
        @param encodings (torch.tensor(max_encoding_len, batch, hidden*2)):
            the out sequence of encoder
        @param lengths (list[int]): list of actual lengths of the encodings
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
                        embeddings=self.pretrained_embeddings,
                        hidden_size=self.hidden_size, mlp_size=self.mlp_size,
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
        @return model (NeuralSim)
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NeuralSim(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model
