#!/usr/bin/python
import torch
import torch.nn as nn

from model_embeddings import ModelEmbeddings

class AvgSim(nn.Module):
    """
    word-averaging cosine sim
    """
    def __init__(self, vocab, embed_size, embeddings, sim_scale=5):
        """
        @param vocab (Vocab): vocab object
        @param embed_size (int): embedding size
        @param embeddings (torch.tensor (len(vocab), embed_dim)): pretrained word embeddings
        @param sim_scale (float): scale the sim score by this scalar
        """
        super(AvgSim, self).__init__()
        self.pretrained_embeddings = embeddings
        self.embeddings = ModelEmbeddings(vocab, embed_size, self.pretrained_embeddings)
        self.vocab = vocab
        self.sim_scale = sim_scale

        self.scoring_fn = nn.CosineSimilarity(dim=-1)

    def forward(self, sents1, sents2):
        """
        given a batch of sent1 and sent2, avg word embeddings
        and finally pass through the scoring function
        @param sents1 (list[list[str]]): batch of sent1 (list[str])
        @param sents2 (list[list[str]]): batch of sent2 (list[str])
        @return scores (torch.Tensor(batch-size,)): sim scores for the batch of (sent1, sent2)
        """
        batch = len(sents1)
        sent1_lens = [len(sent1) for sent1 in sents1]
        sent1_lens = torch.tensor(sent1_lens, dtype=torch.float, device=self.device)
        sent1_padded = self.vocab.sents2Tensor(sents1, device=self.device)
        sent1_embeds = self.embeddings(sent1_padded)
        sent1_avg = torch.sum(sent1_embeds, dim=0) / sent1_lens.view(batch, 1)

        sent2_lens = [len(sent2) for sent2 in sents2]
        sent2_lens = torch.tensor(sent2_lens, dtype=torch.float, device=self.device)
        sent2_padded = self.vocab.sents2Tensor(sents2, device=self.device)
        sent2_embeds = self.embeddings(sent2_padded)
        sent2_avg = torch.sum(sent2_embeds, dim=0) / sent2_lens.view(batch, 1)

        scores = self.scoring_fn(sent1_avg, sent2_avg)
        #scale the scores by sim_scale
        scores = scores * self.sim_scale
        return scores 

    def save(self, file_path):
        """
        saving model to the file_path
        """
        params = {
            'vocab' : self.vocab,
            'args' : dict(embed_size=self.embeddings.embed_size, 
                        embeddings=self.pretrained_embeddings),
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
        model = AvgSim(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model
