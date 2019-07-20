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
    def __init__(self, vocab, embed_size, embeddings, hidden_size, mlp_size, dropout_rate, device):
        """
        @param vocab (Vocab): vocab object
        @param embed_size (int): embedding size
        @param embeddings (torch.tensor (len(vocab), embed_dim)): pretrained word embeddings
        @param hidden_size (int): hidden size
        @param mlp_size (int): mlp size
        @param dropout_rate (float): dropout prob
        """
        super(NLIModel, self).__init__()
        self.pretrained_embeddings = embeddings
        self.embeddings = ModelEmbeddings(vocab, embed_size, self.pretrained_embeddings)
        self.device = device
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.mlp_size = mlp_size
        self.dropout_rate = dropout_rate

        #self.encoder = nn.LSTM(input_size=embed_size, hidden_size=self.hidden_size, num_layers=self.num_layers, bias=True, bidirectional=True)
        self.lstm_1 = nn.LSTM(input_size=embed_size, hidden_size=self.hidden_size, num_layers=1, bias=True, bidirectional=True)
        self.lstm_2 = nn.LSTM(input_size=(embed_size + self.hidden_size*2), hidden_size=self.hidden_size*2, num_layers=1, bias=True, bidirectional=True)
        self.lstm_3 = nn.LSTM(input_size=(embed_size + self.hidden_size*2 + self.hidden_size*4), hidden_size=self.hidden_size*4, num_layers=1, bias=True, bidirectional=True)

        #classifier: in_features = final lstm out_size * 2 (bidirectional) * 4 (prem-hyp feature concat)
        #           out_fearures = 3 labels
        self.mlp_1 = nn.Linear(in_features=self.hidden_size*4*2*4, out_features=self.mlp_size)
        self.mlp_2 = nn.Linear(in_features=self.mlp_size, out_features=self.mlp_size)
        self.sm = nn.Linear(in_features=self.mlp_size, out_features=3)
        self.classifier = nn.Sequential(*[self.mlp_1, nn.ReLU(), nn.Dropout(self.dropout_rate),
                                        self.mlp_2, nn.ReLU(), nn.Dropout(self.dropout_rate),
                                        self.sm])        

    def forward(self, prems, hyps):
        """
        given a batch of prems and hyps, run encoder on prems and hyps
        followed by max-pooling
        and finally pass through the classifier
        @param prems (list[list[str]]): batches of premise (list[str])
        @param hyps (list[list[str]]): batches of hypothesis (list[str])
        @return outs (torch.Tensor(batch-size, 3)): log-likelihod of 3-labels
        """
        prems_lengths = [len(prem) for prem in prems]
        prems_padded = self.vocab.sents2Tensor(prems, device=self.device)

        prems_enc_out = self.encode(prems_padded, prems_lengths)

        #reverse sort hyps and save original lengths and index mapping:
        #   map: indices_sorted -> indices_orig
        hyps_lengths_orig = [len(hyp) for hyp in hyps]
        hyps_sorted, orig_to_sorted = sortHyps(hyps)
        hyps_lengths_sorted = [len(hyp) for hyp in hyps_sorted]
        hyps_padded = self.vocab.sents2Tensor(hyps_sorted, device=self.device)

        hyps_enc_out = self.encode(hyps_padded, hyps_lengths_sorted)
        hyps_enc_out_seq_orig = [hyps_enc_out[:, orig_to_sorted[i], :].unsqueeze(dim=1) 
                                    for i in range(len(hyps))]
        hyps_enc_out_orig = torch.cat(hyps_enc_out_seq_orig, dim=1)

        #max pooling and classifier
        prems_enc_final = self.maxPool(prems_enc_out, prems_lengths)
        hyps_enc_final = self.maxPool(hyps_enc_out_orig, hyps_lengths_orig)
        
        classifier_features = torch.cat([prems_enc_final, hyps_enc_final, torch.abs(prems_enc_final - hyps_enc_final), prems_enc_final * hyps_enc_final], dim=-1)
        
        outs = self.classifier(classifier_features)
        return outs 

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

    def maxPool(self, encodings, lengths):
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
