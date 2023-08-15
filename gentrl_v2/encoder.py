import torch
from torch import nn
from gentrl_v2.tokenizer import encode, get_vocab_size


class RNNEncoder(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2, latent_size=50,
                 bidirectional=False):
        # Simple GRU and 2 linear layers
        super(RNNEncoder, self).__init__()

        self.hidden_size = hidden_size

        self.embs = nn.Embedding(get_vocab_size(), hidden_size)
        self.rnn = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional)

        if bidirectional:
            self.final_mlp = nn.Sequential(
                nn.Linear(2 * hidden_size, hidden_size), nn.LeakyReLU(),
                nn.Linear(hidden_size, 2 * latent_size),
            )
        else:
            self.final_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(),
                nn.Linear(hidden_size, 2 * latent_size),
            )

    def encode(self, sm_list):
        """
        Maps smiles onto a latent space
        """

        tokens, lens = encode(sm_list)
        to_feed = tokens.transpose(1, 0).to(self.embs.weight.device)

        outputs, _ = self.rnn(self.embs(to_feed))

        outputs = outputs[lens, torch.arange(len(lens))]

        return self.final_mlp(outputs)
