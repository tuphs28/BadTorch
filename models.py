import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import Embedding, RNN, Linear

class RecurrentModel:
    """
    Class for constructing recurrent character-level language models

    Args:
        vocab_size (int): number of unique chars in our data.
        emb_dim (int): dimensionality of space in which to embed chars before passing in to model.
        hidden_dim (int): dimensionality of hidden state (and cell state for LSTM).
        n_layers (int): number of recurrent layers to include.
        device (torch.device): device to house model on. Defaults to CPU.
        bias (bool): whether to include biases in layers. Defaults to True.
        cell_type (str): type of RNN cell to use out of:
            - RNN: standard RNN cell consisting of basic hidden-to-hidden updates.
            - GRU: Gated recurrent units with output and forget gates.
            - LSTM: Long short term memory units with input, output and forget gates (non-peephole config)
    """

    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, n_layers: int, device: torch.device = torch.device("cpu"), bias: bool = True, cell_type: str = "RNN"):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.cell_type = cell_type

        self.embedding = Embedding(vocab_size, emb_dim, device)
        self.rnn = RNN(emb_dim, hidden_dim, n_layers, device, bias, cell_type)
        self.linear = Linear(hidden_dim, vocab_size, device, bias)

        for parameter in self.parameters():
            parameter.requires_grad = True

    def __call__(self, x, h = None):

        embs = self.embedding(x)
        hs, h_final = self.rnn(embs, h)
        if self.cell_type == "LSTM": # if using LSTM cells hs contains both hidden and cell states, so need to select hidden state only to pass into next layer
            hs = hs[:,:,:,0]
        hs = hs.contiguous().view(-1, self.hidden_dim).to(self.device)
        logits = self.linear(hs)

        return logits, h_final

    def parameters(self) -> None:
        parameters = []
        parameters += self.embedding.parameters()
        parameters += self.rnn.parameters()
        parameters += self.linear.parameters()
        return parameters

    def sample(self, idx_to_char_dict: dict, sample_length: int = 250) -> str:
        """
        Sampled from recurrent language model (with random start idx) using current weights.

        Args:
            idx_to_char_dict (dict): dictionary mapping from index to char
            sample_length (int, optional): number of chars to sample. Defaults to 250.

        Returns:
            str: string of output text sampled from model
        """

        sampled_idxs = []
        current_idx = torch.randint(high=self.vocab_size, size=(1,1)).to(self.device)
        sampled_idxs.append(current_idx.item())
        hidden = None
        for i in range(sample_length):
            out, hidden = self(current_idx, hidden)
            with torch.no_grad():
                prob = F.softmax(out[-1], dim=0).data
            current_idx = torch.multinomial(prob, 1).view(1,1).to(self.device)
            sampled_idxs.append(current_idx.item())

        sampled_text = []
        for idx in sampled_idxs:
            sampled_text.append(idx_to_char_dict[idx])
        sampled_text = "".join(sampled_text)
        return sampled_text