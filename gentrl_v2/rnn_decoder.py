import torch
import torch.nn as nn
import torch.nn.functional as F
from gentrl_v2.tokenizer import get_vocab_size, encode, decode


class RNNDecoder(nn.Module):
    '''
    Class for autoregressive model that works in WaveNet manner.
        It make conditinioning on previosly sampled tokens by running
        stack of dilation convolution on them.
    '''
    def __init__(self, latent_input_size, token_weights=None, split_len=75,
                rnn_hidden_size = 128, rnn_layers=3, dropout=0.2, bidirectional=False, num_channels=128):
        r'''
        Args:
            latent_input_size: int, size of latent code used in VAE-like models
            token_weights: Tensor of shape [num_tokens], where i-th element
                    contains the weight of i-th token. If None, then all
                    tokens has the same weight.
            split_len: int, maximum length of token sequence
            num_dilated_layers: int, how much dilated layers is in stack
            num_channels: int, num channels in convolutional layers
        '''
        super(RNNDecoder, self).__init__()
        self.vocab_size = get_vocab_size()
        self.latent_input_size = latent_input_size
        self.split_len = split_len
        self.num_channels = num_channels
        self.token_weights = token_weights
        self.rnn_hidden_size = rnn_hidden_size
        self.eos = 2

        self.input_embeddings = nn.Embedding(self.vocab_size,
                                             num_channels)

        self.RNN = nn.GRU(
                num_channels + latent_input_size,
                rnn_hidden_size,
                num_layers=rnn_layers,
                batch_first=True,
                dropout=dropout if rnn_layers > 1 else 0,
                bidirectional=bidirectional
            )

        self.latent_fc = nn.Linear(latent_input_size, rnn_hidden_size)
        self.decoder_fc = nn.Linear(num_channels, self.vocab_size)

        cur_parameters = []
        for layer in [self.input_embeddings, self.RNN, self.latent_fc, self.decoder_fc]:
            cur_parameters += list(layer.parameters())

        self.parameters = nn.ParameterList(cur_parameters)

    def get_logits(self, input_tensor, z, sampling=False):
        '''
        Computing logits for each token input_tensor by given latent code

        [WORKS ONLY IN TEACHER-FORCING MODE]

        Args:
            input_tensor: Tensor of shape [batch_size, max_seq_len]
            z: Tensor of shape [batch_size, lat_code_size]
        '''

        x_emb = self.input_embeddings(input_tensor)
        z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)

        x_input = torch.cat([x_emb, z_0], dim=-1)

        h_0 = self.latent_fc(z)
        h_0 = h_0.unsqueeze(0).repeat(self.RNN.num_layers, 1, 1)

        output, _ = self.RNN(x_input, h_0)

        if self.RNN.bidirectional:
            output = output[:, :, self.rnn_hidden_size:] \
                    + output[:, :, :self.rnn_hidden_size]


        y = self.decoder_fc(output)

        return F.log_softmax(y, dim=-1)

    def get_log_prob(self, x, z):
        '''
        Getting logits of SMILES sequences
        Args:
            x: tensor of shape [batch_size, seq_size] with tokens
            z: tensor of shape [batch_size, lat_size] with latents
        Returns:
            logits: tensor of shape [batch_size, seq_size]
        '''
        seq_logits = torch.gather(self.get_logits(x, z)[:, :-1, :],
                                  2,
                                  x[:, 1:].long().unsqueeze(-1))

        return seq_logits[:, :, 0]

    def forward(self, x, z):
        '''
        Getting logits of SMILES sequences
        Args:
            x: tensor of shape [batch_size, seq_size] with tokens
            z: tensor of shape [batch_size, lat_size] with latents
        Returns:
            logits: tensor of shape [batch_size, seq_size]
            None: since dilconv decoder doesn't have hidden state unlike RNN
        '''
        return self.get_log_prob(x, z), None

    def weighted_forward(self, sm_list, z):
        '''
        '''
        # SMILES -> numerical input
        x = encode(sm_list)[0].to(
            self.input_embeddings.weight.data.device
        )

        seq_logits = self.get_log_prob(x, z)

        if self.token_weights is not None:
            w = self.token_weights[x[:, 1:].long().contiguous().view(-1)]
            w = w.view_as(seq_logits)
            seq_logits = seq_logits * w

        # Normalization of the non end of string components
        non_eof = (x != self.eos)[:, :-1].float()
        ans_logits = (seq_logits * non_eof).sum(dim=-1)
        ans_logits /= non_eof.sum(dim=-1)

        return ans_logits

    def sample(self, latents, max_len=75, argmax=True):
        ''' Sample SMILES for given latents

        Args:
            latents: tensor of shape [n_batch, n_features]

        Returns:
            logits: tensor of shape [batch_size, seq_size], logits of tokens
            tokens: tensor of shape [batch_size, seq_size], sampled token
            None: since dilconv decoder doesn't have hidden state unlike RNN

        '''

        num_objects = latents.shape[0]

        ans_seqs = [[1] for _ in range(num_objects)]

        cur_tokens = torch.tensor(ans_seqs, device="cuda:0").long()

        # Latent in correct shape
        z_0 = latents.unsqueeze(1)

        h = self.latent_fc(latents)
        h = h.unsqueeze(0).repeat(self.RNN.num_layers, 1, 1)

        for i in range(max_len):
            x_emb = self.input_embeddings(cur_tokens)
            x_input = torch.cat([x_emb, z_0], dim=-1)

            o, h = self.RNN(x_input, h)
            y = self.decoder_fc(o.squeeze(1))
            y = F.log_softmax(y, dim=-1)
            y = y.detach()

            if argmax:
                cur_tokens = torch.argmax(y, dim=-1)[1].unsqueeze(-1)
            else:
                cur_tokens = torch.multinomial(F.softmax(y, dim=-1), 1)

            det_tokens = cur_tokens.cpu().detach().tolist()
            ans_seqs = [a + b for a, b in zip(ans_seqs, det_tokens)]

        ans_seqs = torch.tensor(ans_seqs)[:, 1:]
        return decode(ans_seqs)
