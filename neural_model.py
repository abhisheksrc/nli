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
    def __init__(self, vocab, embed_size, embeddings, hidden_size, dropout_rate, device):
        """
        @param vocab (Vocab): vocab object
        @param embed_size (int): embedding size
        @param embeddings (torch.tensor (len(vocab), embed_dim)): pretrained word embeddings
        @param hidden_size (int): hidden size
        @param dropout_rate (float): dropout prob
        """
        super(NeuralModel, self).__init__()
        self.pretrained_embeddings = embeddings
        self.embeddings = ModelEmbeddings(vocab, embed_size, self.pretrained_embeddings)
        self.device = device
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        self.h_projection = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)
        self.c_projection = nn.Linear(self.hidden_size*2, self.hidden_size, bias=False)
        self.att_projection = nn.Linear(self.hidden_size*2, self.hidden_size,bias=False)
        self.combined_out_projection = nn.Linear(self.hidden_size*3, self.hidden_size, bias=False)
        self.vocab_projection = nn.Linear(self.hidden_size, len(self.vocab), bias=False)

        self.dropout = nn.Dropout(self.dropout_rate)

        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=self.hidden_size, bias=True, bidirectional=True)

        self.decoder = nn.LSTMCell(input_size=embed_size+self.hidden_size, hidden_size=self.hidden_size, bias=True)

    def forward(self, prems, hyps):
        """
        given a batch of prems and hyps, run encoder on prems to get hidden state init for decoder
        run decoder to generate the hyps and calculate log-likelihood of the words in the hyps
        @param prems (list[list[str]]): batches of premise (list[str])
        @param hyps (list[list[str]]): batches of hypothesis (list[str])
        @return scores (torch.Tensor(batch-size, )): log-likelihod of generating the words in the hyps
        """
        prem_lengths = [len(prem) for prem in prems]
        
        prem_padded = self.vocab.sents2Tensor(prems, device=self.device)
        hyp_padded = self.vocab.sents2Tensor(hyps, device=self.device)

        enc_hiddens, dec_init_state = self.encode(prem_padded, prem_lengths)
        enc_masks = self.generate_sent_masks(enc_hiddens, prem_lengths)
        
        hyp_predicted = self.decode(hyp_padded, dec_init_state, enc_hiddens, enc_masks)
        P = F.log_softmax(hyp_predicted, dim=-1)        
        
        #create mask to zero out probability for the pad tokens
        hyp_mask = (hyp_padded != self.vocab.pad_id).float()
        #compute cross-entropy between hyp_words and hyp_predicted_words
        hyp_predicted_words_log_prob = torch.gather(P, dim=-1, 
            index=hyp_padded[1:].unsqueeze(-1)).squeeze(-1) * hyp_mask[1:]
        scores = hyp_predicted_words_log_prob.sum(dim=0)
        return scores

    def encode(self, prems, prem_lens):
        """
        apply the encoder on the premises to obtain encoder hidden states
        @param prems (torch.tensor(max_prem_len, batch))
        @param prem_lens (list[int]): list of actual lengths of the prems
        @return enc_hiddens (torch.tensor(batch, max_prem_len, hidden*2)): tensor of seq of hidden outs
        @return dec_init_state (tuple(torch.tensor(batch, hidden), torch.tensor(batch, hidden))): encoders final hidden state and cell concatenated and projected i.e. decoders initial hidden state and cell
        """
        X = self.embeddings(prems)
        X = rnn.pack_padded_sequence(X, prem_lens)
        enc_hiddens, (h_e, c_e) = self.encoder(X)
        enc_hiddens, prem_lens_tensor = rnn.pad_packed_sequence(enc_hiddens)
        batch = prems.shape[1]
        #h_e.shape = (2, b, h)
        h_e_cat = torch.cat((h_e[0, :, :], h_e[1, :, :]), dim=-1).to(self.device)
        c_e_cat = torch.cat((c_e[0, :, :], c_e[1, :, :]), dim=-1).to(self.device)
        #permute dim of enc_hiddens for batch_first
        enc_hiddens = enc_hiddens.permute(1, 0, 2)
        h_d = self.h_projection(h_e_cat)
        c_d = self.c_projection(c_e_cat)
        return enc_hiddens, (h_d, c_d)
        
    def decode(self, hyps, dec_init_state, enc_hiddens, enc_masks):
        """
        @param hyps (torch.tensor(max_hyp_len, batch))
        @param dec_init_state (tuple(torch.tensor(batch, hidden), torch.tensor(batch, hidden))): h_decoder_0, c_decoder_0
        @param enc_hiddens (torch.tensor(b, max_enc_len, h*2))
        @param enc_masks (torch.tensor(b, max_enc_len))
        @return hyp_predicted (torch.tensor(max_hyp_len-1, batch, vocab)): batch of seq of hyp words
        """
        batch_size = enc_hiddens.shape[0]
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        #Chop the last token from all the hyps
        hyps = hyps[:-1]
        
        Y = self.embeddings(hyps)

        (h_t, c_t) = dec_init_state

        enc_hiddens_proj = self.att_projection(enc_hiddens)

        hidden_outs = []
        for y_t in torch.split(Y, split_size_or_sections=1, dim=0):
            y_t = torch.squeeze(y_t, dim=0)#shape(1, b, e) -> (b, e)
            i_t = torch.cat((y_t, o_prev), dim=-1)
            (h_t, c_t), o_t = self.step(i_t, (h_t, c_t), enc_hiddens, enc_hiddens_proj, enc_masks)
            hidden_outs.append(o_t)
            o_prev = o_t

        hidden_outs = torch.stack(hidden_outs, dim=0)
        hyp_predicted = self.vocab_projection(hidden_outs)
        return hyp_predicted

    def step(self, i_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks):
        """
        @param i_t (torch.tensor(b, e+h)): decoder input at t
        @param dec_state (tuple(torch.tensor(b, h), torch.tensor(b, h)))
        @param enc_hiddens (torch.tensor(b, max_enc_len, h*2))
        @param enc_hiddens_proj (torch.tensor(b, max_enc_len, h))
        @param enc_masks (torch.tensor(b, max_enc_len))
        @return dec_next_state (tuple(torch.tensor(b, h), torch.tensor(b, h))): decoder next hidden and cell state
        @return o_t (torch.tensor(b, h)): decoder output at t
        """
        dec_next_state = self.decoder(i_t, dec_state)
        (h_t, c_t) = dec_next_state
        #attention scores
        e_t = torch.bmm(enc_hiddens_proj, h_t.unsqueeze(-1)).squeeze(-1) #(b, max_enc_len)
        #filling -inf to e_t where enc_masks has 1
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.byte(), -float('inf'))
        
        a_t = F.softmax(e_t, dim=-1) #(b, max_enc_len)
        o_t = torch.bmm(a_t.unsqueeze(1), enc_hiddens).squeeze(1) #(b, h*2)
        o_t = torch.cat((h_t, o_t), dim=-1) #(b, h*3)
        
        o_t = self.combined_out_projection(o_t) #(b, h)
        o_t = self.dropout(torch.tanh(o_t))
        return dec_next_state, o_t

    def generate_sent_masks(self, enc_hiddens, sent_lens):
        """
        @param enc_hiddens (torch.tensor(b, max_enc_len, h*2))
        @param sent_lens (list[int]): original len of the sentences
        @return enc_masks (torch.tensor(b, max_enc_len))
        """
        enc_masks = torch.zeros(enc_hiddens.shape[0], enc_hiddens.shape[1], dtype=torch.float, device=self.device)
        for i, sent_len in enumerate(sent_lens):
            enc_masks[i, sent_len:] = 1
        return enc_masks

    def beam_search(self, prem, beam_size, max_decoding_time_step):
        """
        given a premise search possible hyps, from this LG model, up to the beam size
        @param prem (list[str]): premise
        @param beam_size (int)
        @param max_decoding_time_step (int): decode the hyp until <eos> or max decoding time step
        @return best_hyp (list[str]): best possible hyp
        """
        prems = [prem]
        prem_lens = [len(prem)]
        prem_padded = self.vocab.sents2Tensor(prems, device=self.device)

        enc_hiddens, dec_init_state = self.encode(prem_padded, prem_lens)
        enc_hiddens_proj = self.att_projection(enc_hiddens)

        t = 0

        hyps = [['<start>']]
        completed_hyps = []
        hyp_scores = torch.zeros(len(hyps), dtype=torch.float, device=self.device)

        o_t = torch.zeros(1, self.hidden_size, device=self.device)
        (h_t, c_t) = dec_init_state
        while len(completed_hyps) < beam_size and t < max_decoding_time_step:
            hyp_num = len(hyps)
            y_t = torch.tensor([self.vocab[hyp[-1]] for hyp in hyps], dtype=torch.long, device=self.device)
            y_t = self.embeddings(y_t)
            y_t = torch.cat((y_t, o_t), dim=-1)
            enc_hiddens_batch = enc_hiddens.expand(hyp_num, enc_hiddens.shape[1], enc_hiddens.shape[2])
            enc_hiddens_proj_batch = enc_hiddens_proj.expand(hyp_num, enc_hiddens_proj.shape[1], enc_hiddens_proj.shape[2])
            (h_t, c_t), o_t = self.step(y_t, (h_t, c_t), enc_hiddens_batch, enc_hiddens_proj_batch, enc_masks=None)

            log_p_t = F.log_softmax(self.vocab_projection(o_t), dim=-1)
            
            live_hyp_num = beam_size - len(completed_hyps)
            live_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1) #shape = live_hyp_num * len(vocab)
            top_word_scores, top_word_pos = torch.topk(live_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_word_pos / len(self.vocab)
            hyp_word_ids = top_word_pos % len(self.vocab)

            new_hyps = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, top_word_score in zip(prev_hyp_ids, hyp_word_ids, top_word_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                top_word_score = top_word_score.item()

                hyp_word = self.vocab.id2word[hyp_word_id]
                new_hyp_sent = hyps[prev_hyp_id] + [hyp_word]
                if hyp_word == '<eos>':
                    completed_hyps.append((new_hyp_sent[1:-1], top_word_score))
                else:
                    new_hyps.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(top_word_score)

            hyps = new_hyps
            
            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)

            o_t = o_t[live_hyp_ids]
            h_t, c_t = h_t[live_hyp_ids], c_t[live_hyp_ids]

            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

            t += 1
        #end-while
        #in this case best_hyp is not guaranteed
        if len(completed_hyps) == 0:
            completed_hyps.append((hyps[0][1:], hyp_scores[0].item()))

        completed_hyps.sort(key=lambda (hyp, score): score, reverse=True)
        return completed_hyps[0][0]

    def save(self, file_path):
        """
        saving model to the file_path
        """
        params = {
            'vocab' : self.vocab,
            'args' : dict(embed_size=self.embeddings.embed_size, 
                        embeddings=self.pretrained_embeddings,
                        hidden_size=self.hidden_size,
                        dropout_rate=self.dropout_rate, device=self.device),
            'state_dict': self.state_dict()      
        }
        torch.save(params, file_path)

    @staticmethod
    def load(model_path):
        """
        load a saved neural model
        @param model_path (str): path to model
        @return model (NeuralModel)
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NeuralModel(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model
