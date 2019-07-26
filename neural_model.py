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

        self.vocab_projection = nn.Linear(self.hidden_size*2, len(self.vocab), bias=False)

        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=self.hidden_size, bias=True, bidirectional=True)

        self.decoder = nn.LSTMCell(input_size=embed_size, hidden_size=self.hidden_size*2)

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
        hyp_predicted = self.decode(hyp_padded, dec_init_state)
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
        @return enc_hiddens (torch.tensor(max_prem_len, batch, hidden*2)): tensor of seq of hidden outs
        @return dec_init_state (tuple(torch.tensor(batch, hidden*2), torch.tensor(batch, hidden*2))): encoders final hidden state and cell i.e. decoders initial hidden state and cell
        """
        X = self.embeddings(prems)
        X = rnn.pack_padded_sequence(X, prem_lens)
        enc_hiddens, (h_e, c_e) = self.encoder(X)
        enc_hiddens, prem_lens_tensor = rnn.pad_packed_sequence(enc_hiddens)
        batch = prems.shape[1]
        #h_e.shape = (2, b, h)
        h_e_cat = torch.cat((h_e[0, :, :], h_e[1, :, :]), dim=1).to(self.device)
        c_e_cat = torch.cat((c_e[0, :, :], c_e[1, :, :]), dim=1).to(self.device)
        return enc_hiddens, (h_e_cat, c_e_cat)
        
    def decode(self, hyps, dec_init_state):
        """
        @param hyps (torch.tensor(max_hyp_len, batch))
        @param dec_init_state (tuple(torch.tensor(batch, hidden*2), torch.tensor(batch, hidden*2))): h_decoder_0, c_decoder_0
        @return hyp_predicted (torch.tensor(max_hyp_len-1, batch, vocab)): batch of seq of hyp words
        """
        #Chop the last token from all the hyps
        hyps = hyps[:-1]
        
        Y = self.embeddings(hyps)
        (h_t, c_t) = dec_init_state

        hidden_outs = []
        batch_size = hyps.shape[1]

        for y_t in torch.split(Y, split_size_or_sections=1, dim=0):
            y_t = torch.squeeze(y_t, dim=0)#shape(1, b, e) -> (b, e)
            h_t, c_t = self.decoder(y_t, (h_t, c_t))
            hidden_outs.append(h_t)

        hidden_outs = torch.stack(hidden_outs, dim=0)
        hyp_predicted = self.vocab_projection(hidden_outs)
        return hyp_predicted

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

        t = 0

        hyps = [['<start>']]
        completed_hyps = []
        hyp_scores = torch.zeros(len(hyps), dtype=torch.float, device=self.device)

        (h_t, c_t) = dec_init_state
        while len(completed_hyps) < beam_size and t < max_decoding_time_step:
            y_t = torch.tensor([self.vocab[hyp[-1]] for hyp in hyps], dtype=torch.long, device=self.device)
            y_t = self.embeddings(y_t)
            h_t, c_t = self.decoder(y_t, (h_t, c_t))

            log_p_t = F.log_softmax(self.vocab_projection(h_t), dim=-1)
            
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
