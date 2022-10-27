import torch
import numpy as np

class Skipgram(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim
    ):
        super(Skipgram, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        # embedding layer
        self.embed = torch.nn.Embedding(vocab_size, embedding_dim)

        # initial weight will be sampled from P(x) = 1/(vocab_size-0)
        self.embed.weight.data.uniform_(0, vocab_size)

        self.fc = torch.nn.Linear(embedding_dim, vocab_size)

    def forward(self, input):

        # create embeddings for each word
        embedded_basis = self.embed(input)

        # create fully connected layer on top of embeddings
        scores = self.fc(embedded_basis)

        return scores