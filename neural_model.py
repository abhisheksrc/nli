#!/usr/bin/python
import torch
import torch.nn as nn
from torch.nn.utils import rnn
import torch.nn.functional as F

from model_embeddings import ModelEmbeddings

class NeuralModel(nn.Module):
    """
    Bi-LSTM encoder
    LSTM decoder
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
        super(NeuralModel, self).__init__()
        self.embed_size = embed_size
        self.embeddings = ModelEmbeddings(vocab, self.embed_size, embeddings)
        self.device = device
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.vocab_projection = nn.Linear(self.hidden_size*2, len(self.vocab), bias=False)

        self.encoder = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, num_layers=self.num_layers, bias=True, dropout=self.dropout_rate, bidirectional=True)

        self.decoder = nn.LSTMCell(input_size=self.embed_size, hidden_size=self.hidden_size*2)

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
        hyps_padded = self.vocab.sents2Tensor(hyps, device=self.device)

        enc_hiddens, dec_init_state = self.encode(prems_padded, prems_lengths)
        #take the final layer of the decoder_init_state
        h_d_0, c_d_0 = dec_init_state[0], dec_init_state[1]
        dec_init_state = (h_d_0[-1], c_d_0[-1])
        hyps_predicted = self.decode(hyps_padded, dec_init_state)

        P = F.log_softmax(hyps_predicted, dim=-1)        
        
        #create mask to zero out probability for the pad tokens
        hyps_mask = (hyps_padded != self.vocab.pad_id).float()
        #compute cross-entropy between hyps_words and hyps_predicted_words
        hyps_predicted_words_log_prob = torch.gather(P, dim=-1, 
            index=hyps_padded[1:].unsqueeze(-1)).squeeze(-1) * hyps_mask[1:]
        scores = hyps_predicted_words_log_prob.sum(dim=0)
        return scores

    def encode(self, prems, prems_lens):
        """
        apply the encoder on the premises to obtain encoder hidden states
        @param prems (torch.tensor(max_prem_len, batch))
        @param prem_lens (List[int]): list of actual lengths of the prems
        @return enc_hiddens (torch.tensor(max_prem_len, batch, hidden*2)): tensor of seq of hidden outs
        @return dec_init_state (tuple(torch.tensor(num-layers, batch, hidden*2), torch.tensor(num-layers, batch, hidden*2))): encoders final hidden state and cell i.e. decoders initial hidden state and cell
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
        
    def decode(self, hyps, dec_init_state):
        """
        @param hyps (torch.tensor(max_hyp_len, batch))
        @param dec_init_state (tuple(torch.tensor(batch, hidden*2), torch.tensor(batch, hidden*2))): h_decoder_0, c_decoder_0
        @return hyps_predicted (torch.tensor(max_hyp_len-1, batch, vocab)): batch of seq of hyp words
        """
        #Chop the last token from all the hyps
        hyps = hyps[:-1]
        
        Y = self.embeddings(hyps)
        (h_t, c_t) = dec_init_state

        hidden_outs = []
        batch_size = hyps.shape[1]

        for y_t in torch.split(Y, split_size_or_sections=1, dim=0):
            y_t = torch.squeeze(y_t, dim=0)#shape(1, b, e)
            h_t, c_t = self.decoder(y_t, (h_t, c_t))
            hidden_outs.append(h_t)

        hidden_outs = torch.stack(hidden_outs, dim=0)
        hyps_predicted = self.vocab_projection(hidden_outs)
        return hyps_predicted
