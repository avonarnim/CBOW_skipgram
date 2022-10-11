import torch


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

    def forward(self, input, context):
        embedded_basis = self.embed(input)
        # embedded_context = self.context_embedding(context)
        # embedded_product = torch.mul(embedded_basis, embedded_context)

        out = self.fc(embedded_basis)

        return out